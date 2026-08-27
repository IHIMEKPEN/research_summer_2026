"""DAgger-style visited-state teacher batching for the independent wipe ESN.

Collection and IK run on macOS. Real frozen UnifoLM inference is intentionally
strict and refuses mock fallback; run the ``label`` command in Unitree's
Python-3.10/CUDA-12.4 environment, then bring the NPZ cache back to the Mac.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

INSTRUCTION = "Wipe the water off the table with the cloth."


def save_visited_bundle(captures: dict[str, np.ndarray], output: Path, *, episode: int, seed: int):
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **captures)
    manifest = {
        "schema": "unifolm_visited_observations_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "episode": episode,
        "policy_seed": seed,
        "instruction": INSTRUCTION,
        "anchors": int(len(captures["time_s"])),
        "rgb_shape": list(captures["rgb"].shape),
        "proprio_semantics": "23-D [L_xyz,L_rot6d,R_xyz,R_rot6d,waist5]",
        "teacher_period_s": 0.570,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def label_bundle(bundle_path: Path, output: Path, *, mjcf_path: Path, model_id: str, unnorm_key: str):
    """Run real frozen UnifoLM then validated EE→joint IK at each visited state."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Real UnifoLM labeling requires the official CUDA worker. Unitree's release "
            "targets Python 3.10, CUDA 12.4 and FlashAttention2; mock labels are forbidden."
        )
    from src.g1_dex1 import materialize_g1_dex1_mjcf
    from src.step1_profile_unifolm_vla0 import UnifoLMVLAWrapper
    from src.vla_ee_bridge import ee_action_to_joint_target

    batch = np.load(bundle_path)
    robot = materialize_g1_dex1_mjcf(mjcf_path)
    try:
        model = mujoco.MjModel.from_xml_path(str(robot))
    finally:
        if robot.name.startswith("_g1_dex1_runtime"):
            robot.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    teacher = UnifoLMVLAWrapper(
        model_id=model_id, action_dim=23, allow_mock_fallback=False,
        use_fp16=True, use_cuda_graph=False, unnorm_key=unnorm_key,
    )
    ee_actions, joint_targets = [], []
    for image, ee_state, q in zip(batch["rgb"], batch["ee_proprio"], batch["q"]):
        action_gpu, _, _ = teacher.infer_gpu(image, INSTRUCTION, joint_state=ee_state)
        ee = action_gpu.detach().float().cpu().numpy()[:23]
        target = ee_action_to_joint_target(model, data, ee, q)
        ee_actions.append(ee)
        joint_targets.append(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, time_s=batch["time_s"], visited_q=batch["q"],
        ee_action_23d=np.asarray(ee_actions), joint_target_29d=np.asarray(joint_targets),
    )
    meta = {
        "schema": "unifolm_teacher_cache_v1", "model_id": model_id,
        "unnorm_key": unnorm_key, "instruction": INSTRUCTION,
        "source_bundle": str(bundle_path), "mock": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "anchors": len(joint_targets),
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def validate_cache(cache_path: Path, *, expected_anchors: int | None = None):
    cache = np.load(cache_path)
    q = cache["joint_target_29d"]
    ee = cache["ee_action_23d"]
    if q.ndim != 2 or q.shape[1] != 29 or not np.isfinite(q).all():
        raise ValueError(f"Invalid 29-D joint targets: {q.shape}")
    if ee.ndim != 2 or ee.shape[1] != 23 or not np.isfinite(ee).all():
        raise ValueError(f"Invalid 23-D EE actions: {ee.shape}")
    if expected_anchors is not None and len(q) != expected_anchors:
        raise ValueError(f"Expected {expected_anchors} anchors, got {len(q)}")
    meta = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("mock") is not False:
        raise ValueError("Final teacher cache must declare mock=false")
    return {"anchors": len(q), "ee_shape": list(ee.shape), "joint_shape": list(q.shape)}


def mac_capability_report():
    import torch
    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "torch": torch.__version__, "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(), "cuda_available": torch.cuda.is_available(),
        "mujoco": mujoco.__version__,
        "unifolm_official_local_supported": bool(torch.cuda.is_available()),
        "reason": "Official UnifoLM release requires Python 3.10/CUDA 12.4/FlashAttention2.",
    }


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    lab = sub.add_parser("label")
    lab.add_argument("bundle", type=Path); lab.add_argument("output", type=Path)
    lab.add_argument("--mjcf", required=True, type=Path)
    lab.add_argument("--model", default="unitreerobotics/UnifoLM-VLA-Base")
    lab.add_argument("--unnorm-key", default="g1_wipe_table")
    val = sub.add_parser("validate"); val.add_argument("cache", type=Path)
    args = p.parse_args()
    if args.command == "audit": print(json.dumps(mac_capability_report(), indent=2))
    elif args.command == "label": print(json.dumps(label_bundle(args.bundle, args.output, mjcf_path=args.mjcf, model_id=args.model, unnorm_key=args.unnorm_key), indent=2))
    else: print(json.dumps(validate_cache(args.cache), indent=2))


if __name__ == "__main__":
    main()
