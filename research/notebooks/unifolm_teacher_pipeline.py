"""DAgger-style visited-state teacher batching for the independent wipe ESN.

Collection and IK run on macOS or the lab workstation. Real frozen UnifoLM
inference is intentionally strict and refuses mock fallback; run the ``label``
command in Unitree's Python-3.10/CUDA environment, then consume the NPZ cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

INSTRUCTION = "Wipe the water off the table with the cloth."
# G1_EE_6D UnifoLM-VLA emits NUM_ACTIONS_CHUNK=25 steps of ACTION_DIM=23.
# Sparse 570 ms teacher loss needs one EE pose per anchor: use chunk step 0
# (the immediate next action), matching closed-loop Step-3 consumption of the
# first frame of each predicted chunk rather than silently flattening/truncating.
TEACHER_CHUNK_STEP = 0
EE_ACTION_DIM = 23


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_visited_bundle(
    captures: dict[str, np.ndarray],
    output: Path,
    *,
    episode: int,
    seed: int,
    policy_id: str | None = None,
    policy_checkpoint: str | None = None,
    dagger_round: int | None = None,
):
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **captures)
    manifest = {
        "schema": "unifolm_visited_observations_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "episode": episode,
        "policy_seed": seed,
        "policy_id": policy_id,
        "policy_checkpoint": policy_checkpoint,
        "dagger_round": dagger_round,
        "instruction": INSTRUCTION,
        "anchors": int(len(captures["time_s"])),
        "rgb_shape": list(captures["rgb"].shape),
        "proprio_semantics": "23-D [L_xyz,L_rot6d,R_xyz,R_rot6d,waist5]",
        "teacher_period_s": 0.570,
        "bundle_sha256": _sha256_file(output),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _select_ee_action(action_flat: np.ndarray) -> tuple[np.ndarray, dict]:
    """Reshape UnifoLM action chunk and select a documented EE step."""
    flat = np.asarray(action_flat, dtype=np.float32).reshape(-1)
    if flat.size < EE_ACTION_DIM:
        raise ValueError(f"UnifoLM action shorter than {EE_ACTION_DIM}: {flat.size}")
    if flat.size == EE_ACTION_DIM:
        chunk = flat.reshape(1, EE_ACTION_DIM)
    elif flat.size % EE_ACTION_DIM == 0:
        chunk = flat.reshape(-1, EE_ACTION_DIM)
    else:
        raise ValueError(
            f"UnifoLM action length {flat.size} is not a multiple of {EE_ACTION_DIM}; "
            "refusing silent truncation of a malformed chunk."
        )
    step = TEACHER_CHUNK_STEP
    if step >= len(chunk):
        raise ValueError(f"Requested chunk step {step} but chunk length is {len(chunk)}")
    ee = chunk[step].copy()
    meta = {
        "raw_action_numel": int(flat.size),
        "chunk_len": int(len(chunk)),
        "chunk_step_selected": int(step),
        "chunk_step_reason": (
            "Select immediate next EE action (chunk index 0) for the sparse 570 ms "
            "teacher loss; later chunk steps are open-loop futures unused by the "
            "state-only ESN teacher term."
        ),
        "ee_dim": EE_ACTION_DIM,
    }
    return ee, meta


def label_bundle(bundle_path: Path, output: Path, *, mjcf_path: Path, model_id: str, unnorm_key: str):
    """Run real frozen UnifoLM then validated EE→joint IK at each visited state."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Real UnifoLM labeling requires the official CUDA worker. Unitree's release "
            "targets Python 3.10, CUDA 12.4 and FlashAttention2; mock labels are forbidden."
        )
    from src.g1_dex1 import materialize_g1_dex1_mjcf
    from src.step1_profile_unifolm_vla0 import UnifoLMVLAWrapper, _download_unifolm_vla_snapshot
    from src.vla_ee_bridge import ee_action_to_joint_target

    batch = np.load(bundle_path)
    source_meta = {}
    meta_path = bundle_path.with_suffix(".json")
    if meta_path.is_file():
        source_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    robot = materialize_g1_dex1_mjcf(mjcf_path)
    try:
        model = mujoco.MjModel.from_xml_path(str(robot))
    finally:
        if robot.name.startswith("_g1_dex1_runtime"):
            robot.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    teacher = UnifoLMVLAWrapper(
        model_id=model_id, action_dim=EE_ACTION_DIM, allow_mock_fallback=False,
        use_fp16=True, use_cuda_graph=False, unnorm_key=unnorm_key,
    )
    if teacher.model is None:
        raise RuntimeError("UnifoLM loaded as mock despite allow_mock_fallback=False")

    ckpt_path = _download_unifolm_vla_snapshot(model_id)
    ckpt_sha = _sha256_file(ckpt_path)

    from PIL import Image as PILImage

    ee_actions, joint_targets, chunk_meta = [], [], []
    for image, ee_state, q in zip(batch["rgb"], batch["ee_proprio"], batch["q"]):
        # Call the pre-reshape path so chunk length is observable and documented.
        pil_img = PILImage.fromarray(image)
        action_gpu, _ = teacher._infer_unifolm_vla_action_gpu(pil_img, INSTRUCTION, joint_state=ee_state)
        raw = action_gpu.detach().float().cpu().numpy()
        ee, cmeta = _select_ee_action(raw)
        target = ee_action_to_joint_target(model, data, ee, q)
        ee_actions.append(ee)
        joint_targets.append(target)
        chunk_meta.append(cmeta)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        time_s=batch["time_s"],
        visited_q=batch["q"],
        ee_action_23d=np.asarray(ee_actions, dtype=np.float32),
        joint_target_29d=np.asarray(joint_targets, dtype=np.float32),
    )
    meta = {
        "schema": "unifolm_teacher_cache_v1",
        "model_id": model_id,
        "unnorm_key": unnorm_key,
        "instruction": INSTRUCTION,
        "source_bundle": str(bundle_path.resolve()),
        "source_bundle_sha256": source_meta.get("bundle_sha256") or _sha256_file(bundle_path),
        "source_episode": source_meta.get("episode"),
        "source_policy_id": source_meta.get("policy_id"),
        "source_policy_checkpoint": source_meta.get("policy_checkpoint"),
        "source_dagger_round": source_meta.get("dagger_round"),
        "mock": False,
        "allow_mock_fallback": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "anchors": len(joint_targets),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "chunk_step_selected": TEACHER_CHUNK_STEP,
        "chunk_step_reason": chunk_meta[0]["chunk_step_reason"] if chunk_meta else None,
        "chunk_len_observed": [m["chunk_len"] for m in chunk_meta],
        "raw_action_numel": [m["raw_action_numel"] for m in chunk_meta],
        "torch_cuda": torch.version.cuda,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def validate_cache(cache_path: Path, *, expected_anchors: int | None = None, expected_times: np.ndarray | None = None):
    cache = np.load(cache_path)
    q = cache["joint_target_29d"]
    ee = cache["ee_action_23d"]
    times = cache["time_s"]
    if q.ndim != 2 or q.shape[1] != 29 or not np.isfinite(q).all():
        raise ValueError(f"Invalid 29-D joint targets: {q.shape}")
    if ee.ndim != 2 or ee.shape[1] != 23 or not np.isfinite(ee).all():
        raise ValueError(f"Invalid 23-D EE actions: {ee.shape}")
    if expected_anchors is not None and len(q) != expected_anchors:
        raise ValueError(f"Expected {expected_anchors} anchors, got {len(q)}")
    if expected_times is not None:
        if len(times) != len(expected_times) or not np.allclose(times, expected_times, atol=1e-5):
            raise ValueError("Teacher cache timestamps do not match visited bundle")
    meta = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("mock") is not False:
        raise ValueError("Final teacher cache must declare mock=false")
    required = ["model_id", "unnorm_key", "instruction", "source_bundle", "checkpoint_sha256"]
    missing = [k for k in required if not meta.get(k)]
    if missing:
        raise ValueError(f"Teacher cache metadata missing: {missing}")
    if "chunk_step_selected" not in meta:
        raise ValueError("Teacher cache must document chunk_step_selected")
    return {
        "ready": True,
        "anchors": len(q),
        "ee_shape": list(ee.shape),
        "joint_shape": list(q.shape),
        "mock": False,
        "model_id": meta["model_id"],
        "unnorm_key": meta["unnorm_key"],
        "instruction": meta["instruction"],
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "source_bundle": meta["source_bundle"],
        "chunk_step_selected": meta["chunk_step_selected"],
        "chunk_step_reason": meta.get("chunk_step_reason"),
        "times_match_bundle": expected_times is not None,
    }


def mac_capability_report():
    import torch
    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "torch": torch.__version__, "mps_built": getattr(torch.backends, "mps", None) and torch.backends.mps.is_built(),
        "mps_available": getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "mujoco": mujoco.__version__,
        "unifolm_official_local_supported": bool(torch.cuda.is_available()),
        "reason": "Official UnifoLM release requires Python 3.10/CUDA 12.4/FlashAttention2 (SDPA fallback acceptable).",
    }


def workstation_audit(results_dir: Path) -> dict:
    """Collect GPU/package/checkpoint/MJCF/visited-bundle inventory for the lab box."""
    import importlib
    import subprocess
    import sys
    import torch

    def _ver(name):
        try:
            mod = importlib.import_module(name)
            return getattr(mod, "__version__", "present")
        except Exception as exc:
            return f"MISSING:{type(exc).__name__}"

    gpus = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader"],
            text=True,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({"index": parts[0], "name": parts[1], "memory_total": parts[2],
                             "memory_used": parts[3], "memory_free": parts[4]})
    except Exception as exc:
        gpus = [{"error": str(exc)}]

    research = Path(__file__).resolve().parent.parent
    mjcf_candidates = [
        research / "unitree_mujoco/unitree_robots/g1/g1_29dof.xml",
        Path("/home/aihimekpen/research/unitree_ros/robots/g1_description/g1_29dof.xml"),
    ]
    mjcf = [{"path": str(p), "exists": p.is_file()} for p in mjcf_candidates]

    ckpt_roots = [
        Path("/raid/data/aihimekpen/hf_cache/hub/models--unitreerobotics--UnifoLM-VLA-Base"),
        Path("/raid/credit/tariq/unifolm-vla/Unifolm_vla_base"),
    ]
    checkpoints = []
    for root in ckpt_roots:
        if not root.exists():
            continue
        for pt in root.rglob("pytorch_model.pt"):
            checkpoints.append({"path": str(pt), "sha256": _sha256_file(pt), "bytes": pt.stat().st_size})

    visited = []
    for npz in sorted(results_dir.glob("visited*.npz")):
        meta_p = npz.with_suffix(".json")
        meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
        visited.append({"path": str(npz), "bytes": npz.stat().st_size, **{k: meta.get(k) for k in
                        ("episode", "policy_seed", "anchors", "dagger_round", "policy_id")}})

    report = {
        "schema": "workstation_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpus": gpus,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "packages": {
            "transformers": _ver("transformers"),
            "flash_attn": _ver("flash_attn"),
            "mujoco": _ver("mujoco"),
            "datasets": _ver("datasets"),
            "numpy": _ver("numpy"),
            "omegaconf": _ver("omegaconf"),
            "qwen_vl_utils": _ver("qwen_vl_utils"),
            "unifolm_vla": _ver("unifolm_vla"),
        },
        "hf_home": str(Path.home() / ".cache" / "huggingface"),
        "raid_hf_cache": "/raid/data/aihimekpen/hf_cache",
        "mjcf": mjcf,
        "unifolm_checkpoints": checkpoints,
        "visited_bundles": visited,
        "note": "FlashAttention2 preferred; QWen2_5 falls back to SDPA when flash_attn is absent.",
    }
    out = results_dir / "workstation_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    wa = sub.add_parser("workstation-audit")
    wa.add_argument("--results", type=Path, default=None)
    lab = sub.add_parser("label")
    lab.add_argument("bundle", type=Path)
    lab.add_argument("output", type=Path)
    lab.add_argument("--mjcf", required=True, type=Path)
    lab.add_argument("--model", default="unitreerobotics/UnifoLM-VLA-Base")
    lab.add_argument("--unnorm-key", default="g1_wipe_table")
    val = sub.add_parser("validate")
    val.add_argument("cache", type=Path)
    val.add_argument("--expected-anchors", type=int, default=None)
    args = p.parse_args()
    if args.command == "audit":
        print(json.dumps(mac_capability_report(), indent=2))
    elif args.command == "workstation-audit":
        results = args.results or (Path(__file__).resolve().parent.parent / "results/main_independent_esn")
        print(json.dumps(workstation_audit(results), indent=2))
    elif args.command == "label":
        print(json.dumps(label_bundle(args.bundle, args.output, mjcf_path=args.mjcf, model_id=args.model, unnorm_key=args.unnorm_key), indent=2))
    else:
        print(json.dumps(validate_cache(args.cache, expected_anchors=args.expected_anchors), indent=2))


if __name__ == "__main__":
    main()
