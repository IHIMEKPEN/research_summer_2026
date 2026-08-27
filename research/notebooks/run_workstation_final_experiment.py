#!/usr/bin/env python3
"""Frozen-UnifoLM teacher-guided independent ESN wipe experiment (workstation final).

Refuses mock / demonstration-proxy teacher labels in any final artifact path.
Large binaries land under /raid via the workstation_final symlink.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

# Notebooks dir on path for wipe_* modules; research root for src.*.
NOTEBOOKS = Path(__file__).resolve().parent
ROOT = NOTEBOOKS.parent  # research/
REPO = ROOT.parent
sys.path.insert(0, str(NOTEBOOKS))
sys.path.insert(0, str(ROOT))

from unifolm_teacher_pipeline import (  # noqa: E402
    label_bundle,
    save_visited_bundle,
    validate_cache,
    workstation_audit,
)
from wipe_esn_experiment import (  # noqa: E402
    ESN,
    TARGET_CONTACT_RATIO,
    TARGET_COVERAGE_RATIO,
    TARGET_TABLE_AREA_M2,
    fit_bc_initializer,
    pack_episodes,
    rollout,
)
from wipe_optimizers import cem, cmaes_lite, random_search, spsa_adapter  # noqa: E402

RESULTS = ROOT / "results/main_independent_esn"
FINAL = RESULTS / "workstation_final"
MJCF = ROOT / "unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
MODEL_ID = "unitreerobotics/UnifoLM-VLA-Base"
UNNORM_KEY = "g1_wipe_table"

TRAIN_RANGE = list(range(0, 160))
HELDOUT_EPS = list(range(160, 200))
BC_EPS = [0, 1, 2, 3]
VAL_EPS = [4, 5, 6]  # validation subset inside training range only
OPT_TRAIN_EP = 0
SEEDS = [0, 1, 2]
BUDGET = 8
DAGGER_ROUNDS = 3
METHODS = {
    "random": random_search,
    "spsa": spsa_adapter,
    "cem": cem,
    "cmaes": cmaes_lite,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _save_esn(esn: ESN, path: Path, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        Wout=esn.Wout, Win=esn.Win, W=esn.W, mean=esn.mean, scale=esn.scale,
        n=np.array([esn.n]), leak=np.array([esn.leak]), rho=np.array([esn.rho]), seed=np.array([esn.seed]),
    )
    meta = {**meta, "checkpoint_sha256": _sha256(path), "path": str(path.resolve())}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _load_esn(path: Path) -> ESN:
    z = np.load(path)
    esn = ESN(n=int(z["n"][0]), leak=float(z["leak"][0]), rho=float(z["rho"][0]), seed=int(z["seed"][0]))
    esn.Win, esn.W, esn.Wout = z["Win"], z["W"], z["Wout"]
    esn.mean, esn.scale = z["mean"], z["scale"]
    return esn


def _metric_row(result: dict, **extra) -> dict:
    keys = [
        "L_task", "L_teacher", "L_total", "teacher_weight", "teacher_source", "anchors",
        "grasp_success", "wipe_path_length_m", "table_contact_ratio", "wipe_coverage_m2",
        "max_cloth_jump_m", "joint_limit_violation", "task_success",
    ]
    row = {k: result.get(k) for k in keys if k in result}
    # joint-limit: rollout stores via metrics; fall back to limit fields if present
    if "joint_limit_violation" not in row:
        row["joint_limit_violation"] = result.get("joint_limit_failure", result.get("limit_violation", False))
    row.update(extra)
    return row


def _success_at_thresholds(result: dict, contact_thr: float, coverage_thr: float) -> bool:
    grasp = bool(result.get("grasp_success"))
    path_ok = float(result.get("wipe_path_length_m", 0.0)) >= 0.768
    contact_ok = float(result.get("table_contact_ratio", 0.0)) >= contact_thr
    cov_ratio = float(result.get("wipe_coverage_m2", 0.0)) / max(TARGET_TABLE_AREA_M2, 1e-9)
    coverage_ok = cov_ratio >= coverage_thr
    jump_ok = float(result.get("max_cloth_jump_m", 0.0)) <= 0.05
    limit_ok = not bool(result.get("joint_limit_violation", result.get("joint_limit_failure", False)))
    # Prefer explicit task_success for 90% gates when thresholds match defaults.
    if abs(contact_thr - TARGET_CONTACT_RATIO) < 1e-9 and abs(coverage_thr - TARGET_COVERAGE_RATIO) < 1e-9:
        return bool(result.get("task_success")) and jump_ok and limit_ok
    return grasp and path_ok and contact_ok and coverage_ok and jump_ok and limit_ok


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(p), float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def _ensure_env():
    os.environ.setdefault("HF_HOME", "/raid/data/aihimekpen/hf_cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/raid/data/aihimekpen/hf_cache/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/raid/data/aihimekpen/hf_cache/transformers")
    # Headless MuJoCo RGB capture for visited-state bundles.
    os.environ.setdefault("MUJOCO_GL", "egl")
    FINAL.mkdir(parents=True, exist_ok=True)
    if not MJCF.is_file():
        raise FileNotFoundError(f"MJCF missing: {MJCF}")


def _label_if_needed(bundle: Path, cache: Path) -> dict:
    if cache.is_file() and cache.with_suffix(".json").is_file():
        visited = np.load(bundle)
        status = validate_cache(cache, expected_anchors=len(visited["time_s"]), expected_times=visited["time_s"])
        if status.get("mock") is False:
            return status
    print(f"[label] {bundle.name} -> {cache.name}", flush=True)
    meta = label_bundle(bundle, cache, mjcf_path=MJCF, model_id=MODEL_ID, unnorm_key=UNNORM_KEY)
    visited = np.load(bundle)
    status = validate_cache(cache, expected_anchors=len(visited["time_s"]), expected_times=visited["time_s"])
    status["label_meta"] = meta
    return status


def _teacher_targets(cache: Path) -> np.ndarray:
    return np.load(cache)["joint_target_29d"].astype(np.float32)


def run_dagger(episodes: dict, started: float) -> dict:
    history = []
    esn = ESN(n=116, seed=0)
    fit_bc_initializer(esn, {ep: episodes[ep] for ep in BC_EPS})
    ckpt0 = FINAL / "esn_bc_init_seed0.npz"
    _save_esn(esn, ckpt0, {"stage": "bc_initializer", "episodes": BC_EPS, "seed": 0})

    # Round 0: existing visited bundle from BC policy, or collect fresh.
    bundle = RESULTS / "visited_ep0_seed0.npz"
    if not bundle.is_file():
        result = rollout(esn, episodes[OPT_TRAIN_EP], MJCF, capture_anchors=True, teacher_weight=0.0)
        save_visited_bundle(
            result["captures"], bundle, episode=OPT_TRAIN_EP, seed=0,
            policy_id="bc_initializer_seed0", policy_checkpoint=str(ckpt0), dagger_round=0,
        )
    else:
        # Annotate policy identity if missing.
        meta_p = bundle.with_suffix(".json")
        meta = json.loads(meta_p.read_text())
        if not meta.get("policy_checkpoint"):
            meta.update({"policy_id": "bc_initializer_seed0", "policy_checkpoint": str(ckpt0), "dagger_round": 0})
            meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cache = RESULTS / "teacher_ep0_seed0.npz"
    status = _label_if_needed(bundle, cache)
    teacher_q = _teacher_targets(cache)
    history.append({
        "round": 0, "bundle": str(bundle), "teacher_cache": str(cache),
        "anchors": status["anchors"], "checkpoint": str(ckpt0),
        "validate": {k: status[k] for k in status if k != "label_meta"},
        "elapsed_s": time.perf_counter() - started,
    })

    best_val = float("inf")
    best_ckpt = ckpt0
    for rnd in range(1, DAGGER_ROUNDS + 1):
        print(f"[dagger] round {rnd}/{DAGGER_ROUNDS}", flush=True)
        # Optimize with real sparse teacher + continuous task loss.
        _, train_metrics, opt_hist = spsa_adapter(
            esn, episodes[OPT_TRAIN_EP], MJCF, budget=BUDGET, seed=rnd,
            teacher_targets=teacher_q,
        )
        # Validation on train-range subset (never held-out).
        val_rows = []
        for ep in VAL_EPS:
            # Teacher cache is trajectory-specific (visited-state labels). Validate
            # with task loss only on other training-range episodes — never held-out.
            r = rollout(esn, episodes[ep], MJCF, teacher_weight=0.0)
            val_rows.append(_metric_row(r, episode=ep, round=rnd))
        val_mean = float(np.mean([r["L_task"] for r in val_rows]))
        accepted = val_mean < best_val
        ckpt = FINAL / f"esn_dagger_round{rnd:02d}_seed0.npz"
        _save_esn(esn, ckpt, {
            "stage": "dagger", "round": rnd, "accepted": accepted,
            "val_L_task_mean": val_mean, "train_metrics": _metric_row(train_metrics),
        })
        if accepted:
            best_val = val_mean
            best_ckpt = ckpt
        else:
            # Revert to previous best for next collection if not accepted.
            esn = _load_esn(best_ckpt)

        # Collect new visited bundle from accepted (or best) policy and relabel.
        pol_id = f"dagger_round{rnd:02d}_seed0"
        new_bundle = RESULTS / f"visited_round{rnd:02d}_ep0_seed0.npz"
        result = rollout(esn, episodes[OPT_TRAIN_EP], MJCF, capture_anchors=True, teacher_weight=0.0)
        save_visited_bundle(
            result["captures"], new_bundle, episode=OPT_TRAIN_EP, seed=0,
            policy_id=pol_id, policy_checkpoint=str(best_ckpt),
            dagger_round=rnd,
        )
        new_cache = RESULTS / f"teacher_round{rnd:02d}_ep0_seed0.npz"
        status = _label_if_needed(new_bundle, new_cache)
        teacher_q = _teacher_targets(new_cache)
        history.append({
            "round": rnd,
            "accepted": accepted,
            "val_L_task_mean": val_mean,
            "val_rows": val_rows,
            "opt_history_len": len(opt_hist),
            "bundle": str(new_bundle),
            "teacher_cache": str(new_cache),
            "anchors": status["anchors"],
            "checkpoint": str(ckpt),
            "best_checkpoint": str(best_ckpt),
            "validate": {k: status[k] for k in status if k != "label_meta"},
            "elapsed_s": time.perf_counter() - started,
        })
        # Early stop if validation clearly converged (no improvement for this round and prior).
        if rnd >= 2 and not accepted and history[-2].get("accepted") is False:
            print("[dagger] early stop: consecutive non-improving rounds", flush=True)
            break

    # Restore best policy for downstream stages.
    esn = _load_esn(best_ckpt)
    final_ckpt = FINAL / "esn_final_selected.npz"
    shutil.copy2(best_ckpt, final_ckpt)
    _save_esn(esn, final_ckpt, {"stage": "final_selected", "source": str(best_ckpt)})
    return {"history": history, "best_checkpoint": str(best_ckpt), "final_checkpoint": str(final_ckpt),
            "best_val_L_task": best_val, "teacher_cache": str(new_cache if len(history) > 1 else cache)}


def run_optimizer_comparison(episodes: dict, teacher_cache: Path) -> dict:
    teacher_q = _teacher_targets(teacher_cache)
    rows = []
    curves = {}
    for seed in SEEDS:
        base = ESN(n=116, seed=seed)
        fit_bc_initializer(base, {ep: episodes[ep] for ep in BC_EPS})
        base_w = base.Wout.copy()
        for name, fn in METHODS.items():
            print(f"[opt] seed={seed} method={name}", flush=True)
            base.Wout = base_w.copy()
            _, train, history = fn(base, episodes[OPT_TRAIN_EP], MJCF, budget=BUDGET, seed=seed, teacher_targets=teacher_q)
            curves[f"{name}_seed{seed}"] = [h.get("best_L", h.get("L_total")) for h in history]
            # Selection metric: validation subset mean L_task (train range only; teacher is ep0-specific).
            val_losses = []
            for ep in VAL_EPS:
                r = rollout(base, episodes[ep], MJCF, teacher_weight=0.0)
                val_losses.append(r["L_task"])
                rows.append({
                    **_metric_row(r, seed=seed, method=name, split="validation", episode=ep,
                                  budget=BUDGET, train_L_total=train["L_total"]),
                })
            # Stash checkpoint per method/seed for ablations.
            ckpt = FINAL / f"esn_opt_{name}_seed{seed}.npz"
            _save_esn(base, ckpt, {"method": name, "seed": seed, "val_L_task_mean": float(np.mean(val_losses))})
            rows.append({
                **_metric_row(train, seed=seed, method=name, split="train_opt", episode=OPT_TRAIN_EP,
                              budget=BUDGET, val_L_task_mean=float(np.mean(val_losses))),
            })

    # Select method by mean validation L_task across seeds (never held-out).
    method_scores = {}
    for name in METHODS:
        vals = [r["L_task"] for r in rows if r.get("method") == name and r.get("split") == "validation"]
        method_scores[name] = float(np.mean(vals)) if vals else float("inf")
    selected = min(method_scores, key=method_scores.get)
    summary = {
        "schema": "optimizer_comparison_v1",
        "budget": BUDGET,
        "seeds": SEEDS,
        "methods": list(METHODS),
        "selection_criterion": "mean_validation_L_task_over_VAL_EPS_and_seeds",
        "validation_episodes": VAL_EPS,
        "method_scores": method_scores,
        "selected_method": selected,
        "teacher_cache": str(teacher_cache),
        "teacher_mock": False,
        "note": "Real teacher used only on OPT_TRAIN_EP during optimization; validation is task-only because teacher caches are visited-state-specific.",
        "curves": {k: [float(x) for x in v] for k, v in curves.items()},
        "rows": rows,
    }
    (FINAL / "optimizer_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (FINAL / "optimizer_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader(); w.writerows(rows)
    # Plot curves
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in METHODS:
        ys = []
        for seed in SEEDS:
            key = f"{name}_seed{seed}"
            if key in curves:
                ys.append(curves[key])
        if not ys:
            continue
        m = int(min(map(len, ys)))
        arr = np.array([y[:m] for y in ys], dtype=np.float64)
        ax.plot(np.arange(m), arr.mean(0), label=name)
        ax.fill_between(np.arange(m), arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), alpha=0.15)
    ax.set_xlabel("evaluation"); ax.set_ylabel("best L_total"); ax.set_title("Optimizer curves (real teacher)")
    ax.legend(); fig.tight_layout()
    fig.savefig(FINAL / "optimizer_curves.png", dpi=140); plt.close(fig)
    return summary


def run_heldout(episodes: dict, method: str) -> dict:
    rows = []
    latencies = []
    for seed in SEEDS:
        ckpt = FINAL / f"esn_opt_{method}_seed{seed}.npz"
        esn = _load_esn(ckpt)
        for ep in HELDOUT_EPS:
            t0 = time.perf_counter()
            r = rollout(esn, episodes[ep], MJCF, teacher_weight=0.0)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            duration = float(episodes[ep]["t"][-1] - episodes[ep]["t"][0])
            rows.append(_metric_row(
                r, seed=seed, method=method, episode=ep, split="heldout",
                rollout_latency_s=elapsed,
                effective_control_hz=(duration / elapsed) * (1.0 / 0.01) if elapsed > 0 else None,
                # effective rate relative to realtime: sim_seconds / wall_seconds * control_hz is messy;
                # report wall latency and steps/s instead.
                sim_duration_s=duration,
                control_steps=int(duration / 0.01),
                steps_per_wall_s=(duration / 0.01) / elapsed if elapsed > 0 else None,
            ))
            print(f"[heldout] seed={seed} ep={ep} success={r.get('task_success')} L_task={r['L_task']:.3f}", flush=True)

    successes = sum(1 for r in rows if r.get("task_success"))
    n = len(rows)
    rate, lo, hi = _wilson_ci(successes, n)

    def _stat(key):
        vals = np.array([float(r[key]) for r in rows if r.get(key) is not None], dtype=np.float64)
        return {"mean": float(vals.mean()) if len(vals) else None, "std": float(vals.std()) if len(vals) else None}

    summary = {
        "schema": "heldout_summary_v1",
        "episodes": HELDOUT_EPS,
        "n_episodes": len(HELDOUT_EPS),
        "n_trials": n,
        "seeds": SEEDS,
        "method": method,
        "successes": successes,
        "success_rate": rate,
        "success_ci95": [lo, hi],
        "L_task": _stat("L_task"),
        "L_teacher": _stat("L_teacher"),
        "L_total": _stat("L_total"),
        "table_contact_ratio": _stat("table_contact_ratio"),
        "wipe_coverage_m2": _stat("wipe_coverage_m2"),
        "wipe_path_length_m": _stat("wipe_path_length_m"),
        "rollout_latency_s": _stat("rollout_latency_s"),
        "steps_per_wall_s": _stat("steps_per_wall_s"),
        "mean_wall_latency_s": float(np.mean(latencies)) if latencies else None,
    }
    with (FINAL / "heldout_episode_results.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader(); w.writerows(rows)
    (FINAL / "heldout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, key, title in zip(axes, ["table_contact_ratio", "wipe_coverage_m2", "wipe_path_length_m"],
                              ["Contact ratio", "Coverage m²", "Wipe path m"]):
        vals = [float(r[key]) for r in rows]
        ax.hist(vals, bins=12); ax.set_title(title)
    fig.suptitle(f"Held-out metrics ({method})")
    fig.tight_layout(); fig.savefig(FINAL / "heldout_metrics.png", dpi=140); plt.close(fig)
    return summary, rows


def run_ablations(episodes: dict, teacher_cache: Path, best_method: str) -> list[dict]:
    teacher_q = _teacher_targets(teacher_cache)
    rows = []
    # Cross-episode task eval uses VAL_EPS; teacher-conditioned rollouts stay on OPT_TRAIN_EP
    # because the UnifoLM cache is visited-state-specific to that trajectory.
    task_eps = VAL_EPS + [OPT_TRAIN_EP]
    teacher_eps = [OPT_TRAIN_EP]

    for seed in SEEDS:
        # BC only
        esn = ESN(n=116, seed=seed)
        fit_bc_initializer(esn, {ep: episodes[ep] for ep in BC_EPS})
        for ep in task_eps:
            r = rollout(esn, episodes[ep], MJCF, teacher_weight=0.0)
            rows.append(_metric_row(r, seed=seed, ablation="bc_initializer_only", episode=ep))

        # Task loss only
        esn = ESN(n=116, seed=seed)
        fit_bc_initializer(esn, {ep: episodes[ep] for ep in BC_EPS})
        spsa_adapter(esn, episodes[OPT_TRAIN_EP], MJCF, budget=BUDGET, seed=seed, teacher_targets=None)
        for ep in task_eps:
            r = rollout(esn, episodes[ep], MJCF, teacher_weight=0.0)
            rows.append(_metric_row(r, seed=seed, ablation="task_loss_only", episode=ep))

        # Task + real sparse teacher
        esn = ESN(n=116, seed=seed)
        fit_bc_initializer(esn, {ep: episodes[ep] for ep in BC_EPS})
        spsa_adapter(esn, episodes[OPT_TRAIN_EP], MJCF, budget=BUDGET, seed=seed + 17, teacher_targets=teacher_q)
        for ep in teacher_eps:
            r = rollout(esn, episodes[ep], MJCF, teacher_joint_targets=teacher_q, teacher_weight=1.0)
            rows.append(_metric_row(r, seed=seed, ablation="task_plus_real_sparse_teacher", episode=ep))
        for ep in VAL_EPS:
            r = rollout(esn, episodes[ep], MJCF, teacher_weight=0.0)
            rows.append(_metric_row(r, seed=seed, ablation="task_plus_real_sparse_teacher", episode=ep, teacher_on_eval=False))

        # Real sparse teacher only: minimize L_teacher via adapter search
        esn = ESN(n=116, seed=seed)
        fit_bc_initializer(esn, {ep: episodes[ep] for ep in BC_EPS})
        from wipe_optimizers import ArmReadoutAdapter
        adapter = ArmReadoutAdapter(esn.Wout.copy())
        rng = np.random.default_rng(seed + 99)
        best_t = np.zeros(adapter.dim)
        best = rollout(esn, episodes[OPT_TRAIN_EP], MJCF, teacher_joint_targets=teacher_q, teacher_weight=1.0)
        best_score = best["L_teacher"]
        for _ in range(BUDGET):
            theta = rng.normal(0, 0.05, adapter.dim)
            adapter.apply(esn, theta)
            cand = rollout(esn, episodes[OPT_TRAIN_EP], MJCF, teacher_joint_targets=teacher_q, teacher_weight=1.0)
            if cand["L_teacher"] < best_score:
                best_t, best_score = theta, cand["L_teacher"]
        adapter.apply(esn, best_t)
        for ep in teacher_eps:
            r = rollout(esn, episodes[ep], MJCF, teacher_joint_targets=teacher_q, teacher_weight=1.0)
            rows.append(_metric_row(r, seed=seed, ablation="real_sparse_teacher_only", episode=ep))

        # Best optimizer vs SPSA on validation (task metrics)
        for method in (best_method, "spsa"):
            ckpt = FINAL / f"esn_opt_{method}_seed{seed}.npz"
            if not ckpt.is_file():
                continue
            esn = _load_esn(ckpt)
            for ep in task_eps:
                r = rollout(esn, episodes[ep], MJCF, teacher_weight=0.0)
                rows.append(_metric_row(r, seed=seed, ablation=f"best_vs_spsa:{method}", episode=ep))

    with (FINAL / "ablation_results.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader(); w.writerows(rows)
    return rows


def run_threshold_sensitivity(heldout_rows: list[dict]) -> list[dict]:
    rows = []
    for contact in (0.80, 0.90, 0.95):
        for coverage in (0.80, 0.90, 0.95):
            flags = [_success_at_thresholds(r, contact, coverage) for r in heldout_rows]
            n = len(flags)
            s = sum(flags)
            rate, lo, hi = _wilson_ci(s, n)
            rows.append({
                "contact_threshold": contact,
                "coverage_threshold": coverage,
                "successes": s,
                "n": n,
                "success_rate": rate,
                "ci95_lo": lo,
                "ci95_hi": hi,
            })
    with (FINAL / "threshold_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def write_final_report(*, audit, dagger, opt, heldout, ablations, sensitivity, elapsed_s, large_artifacts):
    lines = [
        "# Independent ESN + frozen UnifoLM final report",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        f"Host: {platform.node()}",
        f"Total runtime: {elapsed_s / 3600:.2f} h ({elapsed_s:.1f} s)",
        "",
        "## Workstation",
        f"- GPUs: {audit.get('gpus')}",
        f"- Python/Torch/CUDA: {audit.get('python')} / {audit.get('torch')} / {audit.get('torch_cuda')}",
        f"- Packages: {json.dumps(audit.get('packages', {}), indent=2)}",
        "",
        "## UnifoLM teacher",
        f"- Model: `{MODEL_ID}`",
        f"- unnorm_key: `{UNNORM_KEY}`",
        f"- Mock forbidden: true",
        f"- DAgger rounds recorded: {len(dagger.get('history', []))}",
        f"- Best checkpoint: `{dagger.get('best_checkpoint')}`",
        "",
        "## Optimizer selection",
        f"- Selected: **{opt.get('selected_method')}**",
        f"- Criterion: {opt.get('selection_criterion')}",
        f"- Validation episodes (train range): {VAL_EPS}",
        f"- Method scores: {json.dumps(opt.get('method_scores', {}), indent=2)}",
        "",
        "## Splits",
        f"- BC init episodes: {BC_EPS}",
        f"- Training range: 0–159 ({len(TRAIN_RANGE)} episodes)",
        f"- Validation subset: {VAL_EPS}",
        f"- Held-out: 160–199 ({len(HELDOUT_EPS)} episodes)",
        "",
        "## Held-out (all 40 episodes × seeds)",
        f"- Trials: {heldout.get('n_trials')}",
        f"- Success rate: {heldout.get('success_rate'):.4f} (95% CI {heldout.get('success_ci95')})",
        f"- Contact: {heldout.get('table_contact_ratio')}",
        f"- Coverage: {heldout.get('wipe_coverage_m2')}",
        f"- Path: {heldout.get('wipe_path_length_m')}",
        f"- Losses: L_task={heldout.get('L_task')} L_teacher={heldout.get('L_teacher')} L_total={heldout.get('L_total')}",
        "",
        "## Ablations",
        "See `ablation_results.csv`. Conditions: bc_initializer_only, task_loss_only,",
        "real_sparse_teacher_only, task_plus_real_sparse_teacher, best_vs_spsa:*",
        "",
        "## Threshold sensitivity",
        "See `threshold_sensitivity.csv` for 80/90/95% contact×coverage gates on held-out trials.",
        "",
        "## Large artifacts (not committed to git)",
    ]
    for art in large_artifacts:
        lines.append(f"- `{art['path']}` sha256={art['sha256']} bytes={art['bytes']}")
    lines += [
        "",
        "## Artifact index",
        f"- `{FINAL / 'run_config.json'}`",
        f"- `{FINAL / 'environment_audit.json'}`",
        f"- `{FINAL / 'teacher_cache_manifest.json'}`",
        f"- `{FINAL / 'dagger_history.json'}`",
        f"- `{FINAL / 'optimizer_comparison.csv'}`",
        f"- `{FINAL / 'optimizer_comparison.json'}`",
        f"- `{FINAL / 'heldout_episode_results.csv'}`",
        f"- `{FINAL / 'heldout_summary.json'}`",
        f"- `{FINAL / 'ablation_results.csv'}`",
        f"- `{FINAL / 'threshold_sensitivity.csv'}`",
        f"- `{FINAL / 'esn_final_selected.npz'}`",
        f"- `{FINAL / 'optimizer_curves.png'}`",
        f"- `{FINAL / 'heldout_metrics.png'}`",
        f"- `{FINAL / 'FINAL_REPORT.md'}`",
        "",
        "## Limitations",
        "- Cloth contact is geometric (mocap), not calibrated force.",
        "- FlashAttention2 may be absent; SDPA fallback is used when needed.",
        "- Root disk is nearly full; large caches/checkpoints live under `/raid`.",
    ]
    (FINAL / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-heldout", action="store_true")
    parser.add_argument("--max-heldout", type=int, default=None, help="debug: limit held-out episode count")
    args = parser.parse_args()
    global HELDOUT_EPS
    if args.max_heldout is not None:
        HELDOUT_EPS = list(range(160, 160 + args.max_heldout))

    _ensure_env()
    started = time.perf_counter()
    print("[audit] workstation", flush=True)
    audit = workstation_audit(RESULTS)
    (FINAL / "environment_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    shutil.copy2(RESULTS / "workstation_audit.json", FINAL / "environment_audit.json")

    need = sorted(set(BC_EPS + VAL_EPS + [OPT_TRAIN_EP] + HELDOUT_EPS))
    print(f"[data] loading {len(need)} episodes from G1_Dex1_Wipe_Table", flush=True)
    ds = load_dataset("unitreerobotics/G1_Dex1_Wipe_Table")["train"]
    episodes = pack_episodes(ds, need)

    print("[dagger] starting", flush=True)
    dagger = run_dagger(episodes, started)
    (FINAL / "dagger_history.json").write_text(json.dumps(dagger, indent=2), encoding="utf-8")

    teacher_cache = Path(dagger["teacher_cache"])
    # Manifest of all teacher caches
    manifest = {"caches": [], "mock_any": False}
    for p in sorted(RESULTS.glob("teacher*.npz")):
        meta = json.loads(p.with_suffix(".json").read_text()) if p.with_suffix(".json").is_file() else {}
        if meta.get("mock") is not False:
            manifest["mock_any"] = True
            raise RuntimeError(f"Mock/proxy teacher cache forbidden in final run: {p}")
        manifest["caches"].append({
            "path": str(p.resolve()),
            "sha256": _sha256(p),
            "bytes": p.stat().st_size,
            "anchors": meta.get("anchors"),
            "model_id": meta.get("model_id"),
            "unnorm_key": meta.get("unnorm_key"),
            "checkpoint_sha256": meta.get("checkpoint_sha256"),
            "chunk_step_selected": meta.get("chunk_step_selected"),
            "source_bundle": meta.get("source_bundle"),
        })
    (FINAL / "teacher_cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[opt] comparison", flush=True)
    opt = run_optimizer_comparison(episodes, teacher_cache)
    selected = opt["selected_method"]

    print("[heldout] evaluating", flush=True)
    heldout_summary, heldout_rows = run_heldout(episodes, selected)

    print("[ablation]", flush=True)
    ablation_rows = run_ablations(episodes, teacher_cache, selected)

    print("[sensitivity]", flush=True)
    sensitivity = run_threshold_sensitivity(heldout_rows)

    large_artifacts = []
    for pattern in ("visited*.npz", "teacher*.npz", "esn_*.npz"):
        for p in list(RESULTS.glob(pattern)) + list(FINAL.glob(pattern)):
            if p.stat().st_size >= 100_000:  # >=100KB tracked as large-ish
                large_artifacts.append({"path": str(p.resolve()), "sha256": _sha256(p), "bytes": p.stat().st_size})

    elapsed = time.perf_counter() - started
    run_config = {
        "schema": "workstation_final_run_config_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "unnorm_key": UNNORM_KEY,
        "mjcf": str(MJCF.resolve()),
        "allow_mock_fallback": False,
        "seeds": SEEDS,
        "budget": BUDGET,
        "dagger_rounds": DAGGER_ROUNDS,
        "bc_episodes": BC_EPS,
        "validation_episodes": VAL_EPS,
        "heldout_episodes": HELDOUT_EPS,
        "train_range": [0, 159],
        "selected_method": selected,
        "selection_criterion": opt["selection_criterion"],
        "elapsed_s": elapsed,
        "repo": str(REPO),
        "python": sys.executable,
    }
    (FINAL / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    write_final_report(
        audit=audit, dagger=dagger, opt=opt, heldout=heldout_summary,
        ablations=ablation_rows, sensitivity=sensitivity, elapsed_s=elapsed,
        large_artifacts=large_artifacts,
    )
    print(json.dumps({"elapsed_s": elapsed, "selected": selected, "heldout_success": heldout_summary["success_rate"]}, indent=2))
    print(f"FINAL_REPORT: {FINAL / 'FINAL_REPORT.md'}")


if __name__ == "__main__":
    main()
