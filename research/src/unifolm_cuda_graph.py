"""
CUDA Graph capture/replay for UnifoLM VLA on Tesla V100 (Volta).

Dual-graph design (memory-safe on 32 GB V100):

  1. ``prefill_graph`` — vision + language prefill → first token
  2. ``decode_graph``  — **one** decode step, replayed with ``copy_``-updated
     ``cache_position`` / ``write_slot`` buffers (no tensor indexing inside graph)

Decode steps omit explicit ``position_ids`` so Qwen2.5-VL computes mRoPE from
``cache_position + rope_deltas`` (graph-safe, correct head_dim).
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

_MIN_CAPTURE_HEADROOM_MIB = 512


def cuda_free_mib(device: torch.device | int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    idx = device if isinstance(device, int) else device.index
    idx = idx if idx is not None else 0
    free, _ = torch.cuda.mem_get_info(idx)
    return free / (1024 ** 2)


class VLACUDAGraphEngine:
    """Dual-graph VLA inference: prefill graph + single-step decode graph replay."""

    def __init__(
        self,
        model: Any,
        *,
        device: torch.device,
        stream: torch.cuda.Stream,
        max_new_tokens: int = 64,
        warmup_iters: int = 2,
        use_fp16: bool = True,
        joint_state_dim: int = 29,
        min_capture_headroom_mib: float = _MIN_CAPTURE_HEADROOM_MIB,
    ):
        if warmup_iters < 1:
            raise ValueError("CUDA Graph capture requires at least 1 warmup iteration.")

        self.model = model
        self.device = device
        self.stream = stream
        self.max_new_tokens = max_new_tokens
        self.warmup_iters = warmup_iters
        self.use_fp16 = use_fp16
        self.joint_state_dim = joint_state_dim
        self.min_capture_headroom_mib = min_capture_headroom_mib

        self.static_inputs: Dict[str, torch.Tensor] = {}
        self.static_joint_state: Optional[torch.Tensor] = None
        self.static_output_ids: Optional[torch.Tensor] = None
        self.static_decode_input: Optional[torch.Tensor] = None
        self.static_decode_cache_position: Optional[torch.Tensor] = None
        self.static_write_slot: Optional[torch.Tensor] = None
        self.static_image_mask_expanded: Optional[torch.Tensor] = None
        self.static_rope_deltas: Optional[torch.Tensor] = None
        self.static_prefill_position_ids: Optional[torch.Tensor] = None
        self.static_prefill_cache_position: Optional[torch.Tensor] = None
        self.static_cache: Optional[Any] = None
        self._decode_cache_positions: List[torch.Tensor] = []

        self.prefill_graph: Optional[torch.cuda.CUDAGraph] = None
        self.decode_graph: Optional[torch.cuda.CUDAGraph] = None
        self._sorted_input_keys: List[str] = []
        self._image_token_id: int = 0

        self.input_len: int = 0
        self.captured: bool = False
        self.capture_mode: str = "none"

    def capture_from_template(self, template_inputs: Dict[str, Any]) -> None:
        free_mib = cuda_free_mib(self.device)
        if free_mib < self.min_capture_headroom_mib:
            raise RuntimeError(
                f"Insufficient VRAM for CUDA Graph capture: {free_mib:.0f} MiB free, "
                f"need >= {self.min_capture_headroom_mib:.0f} MiB headroom."
            )

        tensor_inputs = {k: v for k, v in template_inputs.items() if isinstance(v, torch.Tensor)}
        for key in ("input_ids", "pixel_values", "image_grid_thw"):
            if key not in tensor_inputs:
                raise ValueError(f"template_inputs must contain '{key}'.")

        self._prepare_static_buffers(tensor_inputs)
        self._bootstrap_static_metadata()

        if not self._try_capture_dual_graphs():
            self._cleanup_failed_capture()
            raise RuntimeError("Dual CUDA Graph capture failed — see logs above.")

    def replay(
        self,
        inputs: Dict[str, Any],
        joint_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.captured or self.prefill_graph is None or self.decode_graph is None:
            raise RuntimeError("CUDA Graph not captured.")

        with torch.cuda.stream(self.stream):
            for key in self._sorted_input_keys:
                value = inputs.get(key)
                if isinstance(value, torch.Tensor):
                    self.static_inputs[key].copy_(value, non_blocking=True)

            if joint_state is not None and self.static_joint_state is not None:
                self.static_joint_state.copy_(joint_state, non_blocking=True)

            self.prefill_graph.replay()

            for step in range(1, self.max_new_tokens):
                assert self.static_decode_cache_position is not None
                assert self.static_write_slot is not None
                self.static_decode_cache_position.copy_(self._decode_cache_positions[step - 1])
                self.static_write_slot.fill_(self.input_len + step)
                self.decode_graph.replay()

        return self.static_output_ids  # type: ignore[return-value]

    def _prepare_static_buffers(self, tensor_inputs: Dict[str, torch.Tensor]) -> None:
        self._sorted_input_keys = sorted(tensor_inputs.keys())
        self.static_inputs = {
            key: tensor_inputs[key].detach().to(self.device).contiguous()
            for key in self._sorted_input_keys
        }
        self.input_len = int(self.static_inputs["input_ids"].shape[-1])
        out_len = self.input_len + self.max_new_tokens
        self._image_token_id = int(getattr(self.model.config, "image_token_id", 0))

        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.static_joint_state = torch.zeros(self.joint_state_dim, device=self.device, dtype=dtype)
        self.static_output_ids = torch.zeros((1, out_len), device=self.device, dtype=torch.long)
        self.static_decode_input = torch.zeros((1, 1), device=self.device, dtype=torch.long)
        self.static_decode_cache_position = torch.zeros(1, device=self.device, dtype=torch.long)
        self.static_write_slot = torch.zeros(1, device=self.device, dtype=torch.long)

    def _bootstrap_static_metadata(self) -> None:
        from transformers.cache_utils import StaticCache

        input_ids = self.static_inputs["input_ids"]
        image_grid_thw = self.static_inputs["image_grid_thw"]
        attention_mask = self.static_inputs.get("attention_mask")

        self.model.rope_deltas = None
        with torch.inference_mode():
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )

        self.static_rope_deltas = rope_deltas.clone()
        self.static_prefill_position_ids = position_ids.clone()
        self.static_prefill_cache_position = torch.arange(
            0, self.input_len, device=self.device, dtype=torch.long
        )

        hidden_size = int(self.model.config.hidden_size)
        image_mask = (input_ids == self._image_token_id).unsqueeze(-1)
        self.static_image_mask_expanded = image_mask.expand(-1, -1, hidden_size).contiguous()

        self._decode_cache_positions = [
            torch.tensor([self.input_len + step], device=self.device, dtype=torch.long)
            for step in range(self.max_new_tokens)
        ]

        cache_len = self.input_len + self.max_new_tokens
        cache_dtype = torch.float16 if self.use_fp16 else torch.float32
        self.static_cache = StaticCache(
            config=self.model.model.config,
            max_batch_size=1,
            max_cache_len=cache_len,
            device=self.device,
            dtype=cache_dtype,
        )

    def _build_inputs_embeds(self) -> torch.Tensor:
        pixel_values = self.static_inputs["pixel_values"]
        image_grid_thw = self.static_inputs["image_grid_thw"]
        input_ids = self.static_inputs["input_ids"]

        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        token_embeds = self.model.model.embed_tokens(input_ids)
        image_embeds = image_embeds.to(token_embeds.device, token_embeds.dtype)
        return token_embeds.masked_scatter(self.static_image_mask_expanded, image_embeds)

    def _prefill_body(self) -> None:
        assert self.static_cache is not None
        assert self.static_output_ids is not None
        assert self.static_decode_input is not None

        self.static_cache.reset()
        self.static_output_ids.zero_()
        self.static_output_ids[:, : self.input_len].copy_(self.static_inputs["input_ids"])

        inputs_embeds = self._build_inputs_embeds()
        attention_mask = self.static_inputs.get("attention_mask")

        prefill_out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=self.static_cache,
            use_cache=True,
            cache_position=self.static_prefill_cache_position,
            position_ids=self.static_prefill_position_ids,
            rope_deltas=self.static_rope_deltas,
            return_dict=True,
        )
        next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        self.static_output_ids[:, self.input_len : self.input_len + 1].copy_(next_token)
        self.static_decode_input.copy_(next_token)

    def _decode_step_body(self) -> None:
        """
        Single decode step. ``position_ids`` omitted — model derives mRoPE from
        ``cache_position`` + ``rope_deltas`` (avoids split_with_sizes shape bug).
        """
        assert self.static_cache is not None
        assert self.static_output_ids is not None
        assert self.static_decode_input is not None
        assert self.static_decode_cache_position is not None
        assert self.static_write_slot is not None

        decode_out = self.model(
            input_ids=self.static_decode_input,
            past_key_values=self.static_cache,
            use_cache=True,
            cache_position=self.static_decode_cache_position,
            rope_deltas=self.static_rope_deltas,
            return_dict=True,
        )
        next_token = decode_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        self.static_output_ids.index_copy_(1, self.static_write_slot.long(), next_token)
        self.static_decode_input.copy_(next_token)

    def _run_full_eager_once(self) -> None:
        """One full prefill + decode chain for warmup / cache priming."""
        self._prefill_body()
        for step in range(1, self.max_new_tokens):
            self.static_decode_cache_position.copy_(self._decode_cache_positions[step - 1])
            self.static_write_slot.fill_(self.input_len + step)
            self._decode_step_body()

    def _try_capture_dual_graphs(self) -> bool:
        try:
            self._release_memory_before_capture()
            self.stream.wait_stream(torch.cuda.current_stream())

            with torch.inference_mode():
                for i in range(self.warmup_iters):
                    with torch.cuda.stream(self.stream):
                        self._run_full_eager_once()
                    if i < self.warmup_iters - 1:
                        self._release_memory_before_capture()
                self.stream.synchronize()

                self.prefill_graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(self.stream):
                    with torch.cuda.graph(self.prefill_graph, stream=self.stream):
                        self._prefill_body()

                with torch.cuda.stream(self.stream):
                    self._prefill_body()
                    self.static_decode_cache_position.copy_(self._decode_cache_positions[0])
                    self.static_write_slot.fill_(self.input_len + 1)

                self.decode_graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(self.stream):
                    with torch.cuda.graph(self.decode_graph, stream=self.stream):
                        self._decode_step_body()
                self.stream.synchronize()

            self.captured = True
            self.capture_mode = "dual_CUDAGraph"
            logger.info(
                f"CUDA Graph captured (prefill + decode replay, warmup={self.warmup_iters}, "
                f"max_new_tokens={self.max_new_tokens})"
            )
            return True
        except Exception as exc:
            logger.warning(f"Dual CUDA Graph capture failed: {exc}")
            self.prefill_graph = None
            self.decode_graph = None
            self.captured = False
            return False

    @staticmethod
    def _release_memory_before_capture() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @staticmethod
    def _cleanup_failed_capture() -> None:
        VLACUDAGraphEngine._release_memory_before_capture()
