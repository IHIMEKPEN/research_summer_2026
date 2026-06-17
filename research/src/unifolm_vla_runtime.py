"""
V100-optimized async runtime primitives for UnifoLM VLA + ESN integration.

Hardware targets (Tesla V100, PCIe Gen 3):
  - Native FP16 Tensor Core execution (avoid bfloat16 software emulation)
  - Pinned host memory + non-blocking H2D copies
  - Dedicated background CUDA stream for VLA (ESN samples at 100 Hz on default stream)
  - GPU-resident action register with lock-free double buffering
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, torch.Tensor]


# ── Pinned host → async GPU transfer ────────────────────────────────────────
class PinnedHostTransfer:
    """
    Converts NumPy / CPU tensors to pinned host memory, then copies to GPU
    without blocking the default stream (critical on PCIe Gen 3).
    """

    def __init__(self, device: torch.device):
        self.device = device

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pin (if CPU) and copy to ``device`` with ``non_blocking=True``."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        if tensor.device.type != "cpu":
            return tensor
        pinned = tensor.pin_memory()
        return pinned.to(self.device, non_blocking=True)

    def numpy_to_device(
        self,
        array: np.ndarray,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """NumPy array → pinned CPU tensor → async GPU tensor."""
        cpu = torch.from_numpy(np.ascontiguousarray(array)).to(dtype=dtype)
        return self.to_device(cpu)

    def transfer_batch(self, tensors: dict) -> dict:
        """Apply async transfer to every tensor value in a HuggingFace inputs dict."""
        out = {}
        for key, value in tensors.items():
            if isinstance(value, torch.Tensor):
                out[key] = self.to_device(value)
            else:
                out[key] = value
        return out


# ── GPU action register (double-buffered, stream-safe) ────────────────────────
class GPUActionRegister:
    """
    Holds the latest ``action_dim``-dimensional action vector entirely on GPU.

    Writer (VLA background stream) fills the back buffer; readers (ESN @ 100 Hz)
    consume the front buffer.  Buffer swap is a Python reference exchange after
    the writer's CUDA event reports completion — no ``cudaStreamSynchronize`` on
    the ESN read path.
    """

    def __init__(self, action_dim: int, device: torch.device, dtype: torch.dtype = torch.float32):
        self.action_dim = action_dim
        self.device = device
        self.dtype = dtype

        self._front = torch.zeros(action_dim, device=device, dtype=dtype)
        self._back = torch.zeros(action_dim, device=device, dtype=dtype)
        self._seq = 0
        self._write_event: Optional[torch.cuda.Event] = None
        self._swap_lock = threading.Lock()

        if device.type == "cuda":
            self._write_event = torch.cuda.Event(enable_timing=False)

    def write_async(self, action: torch.Tensor, stream: torch.cuda.Stream) -> None:
        """
        Copy ``action`` into the back buffer on ``stream`` (non-blocking).
        Call ``try_publish()`` from the writer thread to expose the new value.
        """
        if action.device.type != "cuda":
            raise ValueError("GPUActionRegister.write_async expects a CUDA tensor.")

        flat = action.detach().flatten()
        if flat.numel() < self.action_dim:
            flat = torch.nn.functional.pad(flat, (0, self.action_dim - flat.numel()))
        flat = flat[: self.action_dim].to(dtype=self.dtype)

        with torch.cuda.stream(stream):
            self._back.copy_(flat, non_blocking=True)
            if self._write_event is not None:
                self._write_event.record(stream)

    def try_publish(self) -> bool:
        """
        Non-blocking: swap front/back if the writer event has completed.
        Returns True when a new action became visible to readers.
        """
        if self._write_event is None:
            return False
        if not self._write_event.query():
            return False
        with self._swap_lock:
            self._front, self._back = self._back, self._front
            self._seq += 1
        return True

    def sample(self) -> torch.Tensor:
        """
        ESN hot path: attempt publish, then return the front buffer (GPU-resident).
        No CPU sync, no allocation — returns a view of the live register.
        """
        self.try_publish()
        return self._front

    @property
    def sequence(self) -> int:
        """Monotonic publish counter (CPU-side, for diagnostics only)."""
        return self._seq


# ── Background VLA worker (dedicated CUDA stream) ─────────────────────────────
class AsyncVLARuntime:
    """
    Runs heavy VLA inference on a dedicated background CUDA stream and publishes
    actions into a :class:`GPUActionRegister`.

    The main thread (or ESN loop @ 100 Hz) calls :meth:`register.sample` without
    ever waiting on VLA completion.
    """

    def __init__(
        self,
        infer_fn: Callable[[np.ndarray, str], Tuple[torch.Tensor, int, int]],
        register: GPUActionRegister,
        instruction: str,
        *,
        device: Optional[torch.device] = None,
        poll_interval_s: float = 0.0,
    ):
        self.infer_fn = infer_fn
        self.register = register
        self.instruction = instruction
        self.poll_interval_s = poll_interval_s

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

        self._obs_lock = threading.Lock()
        self._latest_image: Optional[np.ndarray] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._inflight = False
        self._ticks = 0

    def submit_observation(self, image: np.ndarray, joint_state: Optional[np.ndarray] = None) -> None:
        """Thread-safe: robot/sim pushes the freshest camera frame (and optional joints)."""
        with self._obs_lock:
            self._latest_image = np.ascontiguousarray(image)
            # joint_state reserved for future proprioceptive conditioning
            _ = joint_state

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker_loop, name="AsyncVLARuntime", daemon=True)
        self._thread.start()
        logger.info("AsyncVLARuntime started on dedicated CUDA stream.")

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    @property
    def ticks(self) -> int:
        return self._ticks

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            with self._obs_lock:
                image = self._latest_image

            if image is None:
                time.sleep(0.001)
                continue

            if self._inflight:
                if self.poll_interval_s > 0:
                    time.sleep(self.poll_interval_s)
                continue

            self._inflight = True
            try:
                if self.stream is not None:
                    with torch.cuda.stream(self.stream):
                        action_gpu, _, _ = self.infer_fn(image, self.instruction)
                        self.register.write_async(action_gpu, self.stream)
                    self.register.try_publish()
                else:
                    action_gpu, _, _ = self.infer_fn(image, self.instruction)
                    # CPU fallback: still populate register semantics via direct copy
                    self.register._front.copy_(action_gpu.flatten()[: self.register.action_dim])
                    self.register._seq += 1
                self._ticks += 1
            finally:
                self._inflight = False

            if self.poll_interval_s > 0:
                time.sleep(self.poll_interval_s)


# ── 100 Hz ESN sampler stub (main-thread, synchronous) ─────────────────────
def esn_control_tick(
    register: GPUActionRegister,
    esn_step_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    One 100 Hz control tick: sample latest VLA action from GPU register, run ESN.

    ``esn_step_fn`` must accept a GPU tensor and return a GPU tensor without
    forcing a device sync (no ``.item()``, ``.cpu()``, or ``print(tensor)``).
    """
    vla_action = register.sample()
    return esn_step_fn(vla_action)
