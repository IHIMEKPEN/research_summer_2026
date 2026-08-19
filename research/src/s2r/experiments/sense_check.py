"""Stage −1: probe Unitree G1 camera (+ optional LiDAR) and measure rates.

Run on the robot (or laptop with a USB camera) before any motion / oracle deploy:

  python3 -m s2r.cli sense-check --seconds 10 --camera 0 --show
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class CameraReport:
    ok: bool
    source: str
    backend: str = "opencv"
    width: int = 0
    height: int = 0
    channels: int = 0
    frames: int = 0
    capture_hz: float = 0.0
    process_hz: float = 0.0
    mean_grab_ms: float = 0.0
    mean_process_ms: float = 0.0
    sample_path: Optional[str] = None
    error: Optional[str] = None
    devices: list[str] = field(default_factory=list)


@dataclass
class LidarReport:
    ok: bool
    backend: str = "none"
    topic_or_channel: Optional[str] = None
    samples: int = 0
    hz: float = 0.0
    notes: list[str] = field(default_factory=list)
    error: Optional[str] = None


def list_video_devices() -> list[str]:
    root = Path("/dev")
    if not root.exists():
        return []
    return sorted(str(p) for p in root.glob("video*") if p.exists())


def _open_capture(source: str, width: int, height: int):
    import cv2

    src: Any = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def probe_camera(
    source: str = "0",
    seconds: float = 10.0,
    width: int = 640,
    height: int = 480,
    show: bool = False,
    out_dir: Optional[Path] = None,
    resize_for_process: tuple[int, int] = (224, 224),
) -> CameraReport:
    devices = list_video_devices()
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover
        return CameraReport(ok=False, source=source, devices=devices, error=f"opencv import failed: {exc}")

    cap = _open_capture(source, width, height)
    if not cap.isOpened():
        return CameraReport(
            ok=False,
            source=source,
            devices=devices,
            error=f"failed to open camera source={source!r}; devices={devices}",
        )

    grab_ms: list[float] = []
    proc_ms: list[float] = []
    t0 = time.perf_counter()
    frames = 0
    sample = None
    window = "s2r-sense-check"
    try:
        while (time.perf_counter() - t0) < seconds:
            t_grab = time.perf_counter()
            ok, frame = cap.read()
            grab_ms.append((time.perf_counter() - t_grab) * 1e3)
            if not ok or frame is None:
                continue
            t_proc = time.perf_counter()
            # Cheap stand-in for "pipeline can touch the image" (BGR→RGB + resize).
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _ = cv2.resize(rgb, resize_for_process)
            proc_ms.append((time.perf_counter() - t_proc) * 1e3)
            frames += 1
            sample = frame
            if show:
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if show:
            try:
                cv2.destroyWindow(window)
            except Exception:
                pass

    elapsed = max(1e-6, time.perf_counter() - t0)
    sample_path = None
    if sample is not None and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_path = str(out_dir / "camera_sample.jpg")
        cv2.imwrite(sample_path, sample)

    h, w = (int(sample.shape[0]), int(sample.shape[1])) if sample is not None else (0, 0)
    c = int(sample.shape[2]) if sample is not None and sample.ndim == 3 else 0
    return CameraReport(
        ok=frames > 0,
        source=source,
        width=w,
        height=h,
        channels=c,
        frames=frames,
        capture_hz=frames / elapsed,
        process_hz=frames / elapsed,
        mean_grab_ms=float(np.mean(grab_ms)) if grab_ms else 0.0,
        mean_process_ms=float(np.mean(proc_ms)) if proc_ms else 0.0,
        sample_path=sample_path,
        devices=devices,
        error=None if frames > 0 else "opened but got zero frames",
    )


def _ros2_topic_list() -> list[str]:
    try:
        out = subprocess.run(
            ["ros2", "topic", "list"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def probe_lidar(seconds: float = 5.0) -> LidarReport:
    """Best-effort LiDAR discovery (ROS2 topics and/or Unitree DDS hints)."""
    _ = seconds  # reserved for a future timed subscriber
    notes: list[str] = []
    topics = _ros2_topic_list()
    lidarish = [
        t
        for t in topics
        if any(k in t.lower() for k in ("lidar", "utlidar", "livox", "points", "pointcloud", "scan"))
    ]
    if lidarish:
        topic = lidarish[0]
        notes.append(f"ros2 topics: {', '.join(lidarish[:8])}")
        notes.append(f"for precise rate on robot: ros2 topic hz {topic}")
        echo_ok = False
        try:
            out = subprocess.run(
                ["ros2", "topic", "echo", topic, "--once"],
                capture_output=True,
                text=True,
                timeout=max(3.0, seconds),
                check=False,
            )
            echo_ok = out.returncode == 0 and bool(out.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            notes.append(f"topic listed but echo failed: {exc}")
        return LidarReport(
            ok=True,
            backend="ros2",
            topic_or_channel=topic,
            samples=1 if echo_ok else 0,
            hz=0.0,
            notes=notes,
        )

    # Unitree SDK2 python: optional import — do not hard-require.
    try:
        import unitree_sdk2py  # noqa: F401

        notes.append("unitree_sdk2py importable; start Unitree lidar DDS example if L1 is installed")
        return LidarReport(
            ok=False,
            backend="unitree_sdk2py_available",
            topic_or_channel="rt/utlidar/cloud (typical)",
            samples=0,
            hz=0.0,
            notes=notes
            + [
                "No ROS2 lidar topic found. If G1 has Unitree L1, run vendor lidar subscriber "
                "on the robot iface and re-run sense-check.",
            ],
        )
    except Exception:
        notes.append("unitree_sdk2py not installed in this env")

    if not topics:
        notes.append("ros2 not available or no topics; LiDAR may be off or not on this Edu image")
    else:
        notes.append(f"ros2 up ({len(topics)} topics) but none matched lidar/points naming")

    return LidarReport(ok=False, backend="none", notes=notes, error="no lidar stream detected")


def run_sense_check(
    camera: str = "0",
    seconds: float = 10.0,
    width: int = 640,
    height: int = 480,
    show: bool = False,
    out: Optional[str] = None,
    skip_lidar: bool = False,
) -> dict[str, Any]:
    out_path = Path(out) if out else Path("results/sense_check/latest.json")
    out_dir = out_path.parent
    cam = probe_camera(
        source=camera,
        seconds=seconds,
        width=width,
        height=height,
        show=show,
        out_dir=out_dir,
    )
    lidar = LidarReport(ok=False, backend="skipped", notes=["--skip-lidar"]) if skip_lidar else probe_lidar(
        seconds=min(5.0, max(2.0, seconds * 0.5))
    )
    report = {
        "ts": time.time(),
        "stage": -1,
        "camera": asdict(cam),
        "lidar": asdict(lidar),
        "data_flow": {
            "rgb": "OpenCV VideoCapture → BGR frame → (RGB+resize timing) → sample JPEG + JSON",
            "later_s2r": "camera_node → ZMQ camera_pub → vision_node → perception_pub (no joints yet)",
        },
        "exit_criteria": {
            "camera_ok": cam.ok,
            "capture_hz": cam.capture_hz,
            "lidar_ok_or_documented": lidar.ok or bool(lidar.notes),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["out"] = str(out_path)
    return report
