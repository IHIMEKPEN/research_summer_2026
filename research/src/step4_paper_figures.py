"""
Step 4 — ICRA paper figures from *measured* results only (no SUCCESS_PRIORS / mock grids).

Outputs (PDF + PNG) under:
  research/results/step4_paper_figures/
  papers/icra2027/figures/   (synced for \includegraphics)

Figures:
  fig1_architecture       Frozen UnifoLM → Jacobian IK hold → ESN+W_out → MuJoCo PD @ 100 Hz
  fig2_latency_gap        Step-1 PyTorch latency (n=100) + frequency gap
  fig3_dataset_distribution  Wipe-table episode lengths + train/held-out split
  fig4_offline_baselines  Held-out open-loop RMSE (ZOH / linear / PID / ESN)
  fig5_hyperparam_sweep   Leak × ridge heatmap (ep.0 wipe)
  fig6_contact_ladder     Oracle vs live contact (press_table disclosure)
  fig7_control_rates      Dual-process live bridge rates
  fig8_oracle_heldout     MuJoCo oracle RMSE / contact over 40 held-out eps

Usage (from research/):
  python3 -m src.step4_paper_figures
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.paths import RESEARCH_ROOT, result_file, results_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = results_path("step4_paper_figures")
PAPER_FIG_DIR = RESEARCH_ROOT.parent / "papers" / "icra2027" / "figures"
WRL_FIG_DIR = RESEARCH_ROOT.parent / "papers" / "neurips2026_wrl" / "figures"

# Frozen Aug-2025/2026 campaign paths (do not use smoke-overwritten latest stubs).
PROFILE_REPORT = result_file(
  "step1_profiling_unifolm_vla0",
  "pytorch_profiler_20260805_140730",
  "profiling_report.json",
)
PROFILE_LOG = result_file(
  "step1_profiling_unifolm_vla0",
  "pytorch_profiler_20260805_140730",
  "inference_log.json",
)
DATASET_CARD = result_file("step2_training", "dataset_card_wipe_table.json")
ESN_SEED = result_file("step2_training", "esn_task_registry_seed.json")
SWEEP_CSV = result_file("step2_training", "esn_hyperparam_sweep.csv")
BASELINE_JSON = result_file("step3_baselines", "baseline_comparison_summary.json")
DUAL_CSV = result_file("step3_dual_thread", "dual_thread_summary_all_live.csv")
LIVE_ESN = result_file("step3_live_wipe", "live_wipe_report_esn_live.json")
LIVE_ZOH = result_file("step3_live_wipe", "live_wipe_report_zoh_live.json")
MUJOCO_CSV = result_file("step4_mujoco_evaluation", "mujoco_eval_summary_heldout.csv")
MUJOCO_JSON = result_file("step4_mujoco_evaluation", "mujoco_eval_summary_heldout.json")

plt.rcParams.update({
  "font.family": "serif",
  "font.size": 9,
  "axes.titlesize": 10,
  "axes.labelsize": 9,
  "xtick.labelsize": 8,
  "ytick.labelsize": 8,
  "legend.fontsize": 8,
  "figure.dpi": 150,
  "savefig.dpi": 300,
  "savefig.bbox": "tight",
  "axes.spines.top": False,
  "axes.spines.right": False,
})

COLORS = {
  "vla": "#C62828",
  "esn": "#2E7D32",
  "zoh": "#EF6C00",
  "linear": "#1565C0",
  "pid": "#6A1B9A",
  "req": "#424242",
  "train": "#1565C0",
  "heldout": "#C62828",
}


def _load_json(path: Path) -> Dict[str, Any]:
  with open(path) as f:
    return json.load(f)


def _save(fig: plt.Figure, stem: str) -> Path:
  FIGURES_DIR.mkdir(parents=True, exist_ok=True)
  PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
  WRL_FIG_DIR.mkdir(parents=True, exist_ok=True)
  pdf = FIGURES_DIR / f"{stem}.pdf"
  png = FIGURES_DIR / f"{stem}.png"
  fig.savefig(pdf)
  fig.savefig(png)
  plt.close(fig)
  for ext in (".pdf", ".png"):
    src = FIGURES_DIR / f"{stem}{ext}"
    shutil.copy2(src, PAPER_FIG_DIR / f"{stem}{ext}")
  shutil.copy2(png, WRL_FIG_DIR / f"{stem}.png")
  logger.info("Saved %s (+ ICRA PDF/PNG, WRL PNG)", pdf)
  return pdf


def _esn_wipe_metrics() -> Dict[str, float]:
  """Prefer seeded paper wipe row; never use smoke-overwritten registry."""
  seed = _load_json(ESN_SEED)
  wipe = seed.get("wipe_table", seed) if isinstance(seed, dict) else {}
  return {
    "heldout_rmse_mean": float(wipe["heldout_rmse_mean"]),
    "heldout_rmse_std": float(wipe["heldout_rmse_std"]),
    "train_rmse": float(wipe.get("train_rmse", float("nan"))),
  }


# ── Fig 1: Architecture ───────────────────────────────────────
def fig1_architecture() -> Path:
  """Two-rate stack as implemented: frozen UnifoLM (Process A) → Jacobian IK
  hold → fixed ESN + ridge W_out (Process B) → MuJoCo PD @ 100 Hz.

  Cross-checked against step1 (UnifoLM-VLA-Base FP16, 23-D G1_EE_6D),
  vla_ee_bridge (xyz Jacobian IK, 23-D EE → 29-D q*), step2 (u=[q;q*]∈R^58,
  N=1000, ρ=0.95, α=0.3, λ=1, 50-tick hold), and step3_dual_thread
  (GIL-free shared q*, MuJoCo 29-DoF PD). Not hardware closed-loop.
  """
  readout = "#00695C"
  fig, ax = plt.subplots(figsize=(7.42, 3.92))
  ax.set_xlim(0.0, 14.55)
  ax.set_ylim(0.02, 6.72)
  ax.axis("off")
  fig.patch.set_facecolor("white")

  def box(x, y, w, h, title, sub, color, title_fs=7.6, sub_fs=5.7, lw=1.45):
    ax.add_patch(
      FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.05",
        linewidth=lw, edgecolor=color, facecolor=color + "16",
        mutation_aspect=0.3, zorder=2,
      )
    )
    ax.text(
      x + w / 2, y + h * 0.70, title, ha="center", va="center",
      fontsize=title_fs, fontweight="bold", color=color, clip_on=False, zorder=3,
    )
    ax.text(
      x + w / 2, y + h * 0.32, sub, ha="center", va="center",
      fontsize=sub_fs, color="#444444", linespacing=1.18, clip_on=False, zorder=3,
    )
    return (x, y, w, h)

  def h_arrow(x1, x2, y, label="", color="#333333", label_side="above"):
    ax.annotate(
      "", xy=(x2, y), xytext=(x1, y),
      arrowprops=dict(
        arrowstyle="-|>", lw=1.45, color=color,
        mutation_scale=10, shrinkA=0, shrinkB=0,
      ),
      clip_on=False, zorder=4,
    )
    if label:
      dy = 0.20 if label_side == "above" else -0.26
      ax.text(
        (x1 + x2) / 2, y + dy, label, ha="center", va="center",
        fontsize=6.0, color=color,
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.10"),
        clip_on=False, zorder=5,
      )

  def elbow_arrow(points, label="", color="#455A64", label_xy=None, lw=1.45):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=1)
    ax.annotate(
      "", xy=points[-1], xytext=points[-2],
      arrowprops=dict(
        arrowstyle="-|>", lw=lw, color=color,
        mutation_scale=10, shrinkA=0, shrinkB=0,
      ),
      clip_on=False, zorder=2,
    )
    if label and label_xy is not None:
      ax.text(
        label_xy[0], label_xy[1], label, ha="center", va="center",
        fontsize=6.0, color=color,
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.10"),
        clip_on=False, zorder=5,
      )

  # Lane bands
  ax.add_patch(
    FancyBboxPatch(
      (3.42, 3.58), 10.95, 2.22, boxstyle="round,pad=0.02",
      linewidth=0, facecolor="#FFEBEE", alpha=0.38, zorder=0,
    )
  )
  ax.add_patch(
    FancyBboxPatch(
      (3.42, 0.48), 10.95, 2.82, boxstyle="round,pad=0.02",
      linewidth=0, facecolor="#E8F5E9", alpha=0.42, zorder=0,
    )
  )
  ax.text(
    14.20, 5.58, r"Slow path  ($\sim$1.75 Hz)  ·  Process A",
    fontsize=6.6, color=COLORS["vla"], fontweight="bold", ha="right", va="center",
  )
  ax.text(
    14.20, 3.12, "Fast path  (100 Hz)  ·  Process B",
    fontsize=6.6, color=COLORS["esn"], fontweight="bold", ha="right", va="center",
  )

  # Observations
  ax.add_patch(
    FancyBboxPatch(
      (0.12, 0.48), 3.16, 5.32, boxstyle="round,pad=0.04",
      linewidth=1.05, edgecolor="#90A4AE", facecolor="#ECEFF1",
      linestyle="--", zorder=0,
    )
  )
  ax.text(1.70, 5.58, "Observations", ha="center", fontsize=7.1,
          color="#37474F", fontweight="bold")

  cam = box(0.30, 4.22, 2.80, 1.10, "RGB", "MuJoCo G1 camera", "#1565C0", 7.5, 5.6)
  lang = box(0.30, 2.90, 2.80, 1.10, "Language", "wipe / clean instruction", "#0277BD", 7.5, 5.6)
  state = box(
    0.30, 0.64, 2.80, 2.02, "Robot state",
    "23-D EE proprio → VLA\n29-DoF  $q_t$  → ESN",
    "#455A64", 7.5, 5.6,
  )

  vla = box(
    4.55, 3.72, 3.05, 1.78, "UnifoLM-VLA-Base",
    "frozen  ·  FP16\n" + r"$\sim$1.75 Hz  /  571 ms",
    COLORS["vla"], 7.5, 5.7,
  )
  ik = box(
    8.50, 3.72, 2.55, 1.78, "Jacobian IK",
    r"23-D EE $\rightarrow$ 29-D $q^{\star}$" + "\nxyz  ·  legs held",
    "#6A1B9A", 7.5, 5.7,
  )

  ax.add_patch(
    FancyBboxPatch(
      (4.42, 0.58), 6.38, 2.28, boxstyle="round,pad=0.03",
      linewidth=1.05, edgecolor=COLORS["esn"], facecolor="none",
      linestyle="--", zorder=1,
    )
  )
  ax.text(
    7.61, 2.94, r"ESN bridge   $u=[q_t;\,q^{\star}]\in\mathbb{R}^{58}$",
    ha="center", va="center", fontsize=6.2, color=COLORS["esn"], fontweight="bold",
    zorder=3,
  )
  esn = box(
    4.55, 0.70, 3.05, 1.78, "ESN reservoir",
    r"fixed  $W_{\mathrm{res}},W_{\mathrm{in}}$" + "\n" + r"$N{=}1000$, $\rho{=}0.95$, $\alpha{=}0.3$",
    COLORS["esn"], 7.5, 5.6,
  )
  wout = box(
    8.50, 0.70, 2.55, 1.78, r"Ridge $W_{\mathrm{out}}$",
    "trained (only)" + "\n" + r"$y=W_{\mathrm{out}}[x;u]$",
    readout, 7.5, 5.6, lw=1.7,
  )
  g1 = box(
    11.55, 0.70, 2.65, 1.78, "MuJoCo G1",
    "29-DoF PD targets\n100 Hz  (≤10 ms)",
    COLORS["req"], 7.5, 5.6,
  )

  # Obs → VLA
  h_arrow(cam[0] + cam[2], vla[0], 4.78, "image", "#1565C0", "above")
  elbow_arrow(
    [
      (lang[0] + lang[2], lang[1] + lang[3] / 2),
      (4.12, lang[1] + lang[3] / 2),
      (4.12, 4.28),
      (vla[0], 4.28),
    ],
    label="text",
    color="#0277BD",
    label_xy=(3.72, 3.72),
  )
  elbow_arrow(
    [
      (state[0] + state[2], state[1] + state[3] * 0.72),
      (4.12, state[1] + state[3] * 0.72),
      (4.12, 3.95),
      (vla[0], 3.95),
    ],
    label="23-D EE",
    color="#455A64",
    label_xy=(3.68, 2.28),
  )

  # VLA → IK
  h_arrow(vla[0] + vla[2], ik[0], 4.61, "", COLORS["vla"])
  ax.text(
    (vla[0] + vla[2] + ik[0]) / 2, 5.28, "23-D EE chunk",
    ha="center", va="center", fontsize=6.0, color=COLORS["vla"],
    bbox=dict(facecolor="#FFEBEE", edgecolor="none", boxstyle="round,pad=0.12"),
    clip_on=False, zorder=5,
  )

  # IK → ESN (latched shared register; 50 ticks at the 2 Hz training hold)
  elbow_arrow(
    [
      (ik[0], ik[1]),
      (ik[0], 3.22),
      (esn[0] + esn[2] * 0.50, 3.22),
      (esn[0] + esn[2] * 0.50, esn[1] + esn[3]),
    ],
    label=r"held $q^{\star}$  (50 ticks, shared)",
    color="#6A1B9A",
    label_xy=(7.55, 3.40),
  )

  # Fast proprio → reservoir
  h_arrow(
    state[0] + state[2], esn[0], esn[1] + esn[3] * 0.38,
    r"$q_t$  @ 100 Hz", "#455A64", "above",
  )

  # Reservoir → readout → MuJoCo
  h_arrow(esn[0] + esn[2], wout[0], wout[1] + wout[3] * 0.55, r"$x$", COLORS["esn"], "above")
  h_arrow(
    wout[0] + wout[2], g1[0], g1[1] + g1[3] * 0.55,
    r"$\hat{y}\in\mathbb{R}^{29}$  @ 100 Hz", readout, "above",
  )

  # Closed-loop proprio from MuJoCo back to q_t
  elbow_arrow(
    [
      (g1[0] + g1[2] / 2, g1[1]),
      (g1[0] + g1[2] / 2, 0.28),
      (state[0] + state[2] / 2, 0.28),
      (state[0] + state[2] / 2, state[1]),
    ],
    label=r"closed loop  $q_t$  (sim)",
    color=COLORS["req"],
    label_xy=(8.55, 0.14),
  )

  ax.text(
    8.90, 6.38,
    r"Frequency gap: UnifoLM $\sim$1.75 Hz  $\rightarrow$  ESN bridge  $\rightarrow$  100 Hz  ($\sim$57$\times$)",
    ha="center", fontsize=7.6, color=COLORS["vla"],
    bbox=dict(facecolor="#FFEBEE", edgecolor=COLORS["vla"], boxstyle="round,pad=0.26"),
  )
  ax.set_title(
    "VLA + ESN hybrid control for Unitree G1 (29-DoF, MuJoCo)",
    fontweight="bold", pad=7, fontsize=10,
  )
  return _save(fig, "fig1_architecture")


# ── Fig 2: Latency gap ────────────────────────────────────────
def fig2_latency_gap() -> Path:
  report = _load_json(PROFILE_REPORT)
  log = _load_json(PROFILE_LOG)
  latencies = np.array([r["latency_ms"] for r in log], dtype=float)
  mean_lat = float(report["mean_latency_ms"])
  p95 = float(report["p95_latency_ms"])
  mean_hz = float(report["mean_hz"])
  gap = float(report["frequency_gap"])
  dual = pd.read_csv(DUAL_CSV)
  esn_hz = float(dual.loc[dual["bridge"] == "esn", "achieved_control_hz"].iloc[0])

  fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
  ax = axes[0]
  ax.hist(latencies, bins=18, color=COLORS["vla"], edgecolor="white", alpha=0.9)
  ax.axvline(mean_lat, color="darkred", ls="--", lw=1.8, label=f"Mean {mean_lat:.1f} ms")
  ax.axvline(p95, color="#EF6C00", ls=":", lw=1.8, label=f"P95 {p95:.1f} ms")
  ax.axvline(10.0, color=COLORS["esn"], ls="-", lw=1.6, label="10 ms (100 Hz)")
  ax.set_xlabel("UnifoLM inference latency (ms)")
  ax.set_ylabel("Count (n=100)")
  ax.set_title("(a) Live UnifoLM-VLA-Base (G1_Clean_Table)")
  ax.legend(fontsize=7, loc="upper left")

  ax2 = axes[1]
  labels = ["UnifoLM\n(PyTorch)", "G1\nrequired", "Live ESN\nbridge"]
  vals = [mean_hz, 100.0, esn_hz]
  colors = [COLORS["vla"], COLORS["req"], COLORS["esn"]]
  bars = ax2.bar(labels, vals, color=colors, edgecolor="white", width=0.55)
  ax2.axhline(100, color=COLORS["req"], ls="--", lw=1.0, alpha=0.5)
  for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.1f} Hz",
             ha="center", fontsize=8, fontweight="bold")
  ax2.set_ylabel("Rate (Hz)")
  ax2.set_ylim(0, max(vals) * 1.2)
  ax2.set_title(f"(b) Frequency gap ≈ {gap:.0f}× vs 100 Hz")

  fig.suptitle("Measured frequency gap on DGX V100", fontweight="bold", y=1.02)
  fig.tight_layout()
  return _save(fig, "fig2_latency_gap")


# ── Fig 3: Dataset distribution ───────────────────────────────
def fig3_dataset_distribution() -> Path:
  card = _load_json(DATASET_CARD)
  durs = np.array(list(card["duration_per_episode_s"].values()), dtype=float)
  frames = np.array(list(card["frames_per_episode"].values()), dtype=float)
  split = card["recommended_split"]
  n_train = len(split["train_episodes"])
  n_hold = len(split["heldout_episodes"])

  fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
  ax = axes[0]
  ax.hist(durs[:n_train], bins=16, color=COLORS["train"], alpha=0.85,
          edgecolor="white", label=f"Train 0–159 (n={n_train})")
  ax.hist(durs[n_train:], bins=10, color=COLORS["heldout"], alpha=0.75,
          edgecolor="white", label=f"Held-out 160–199 (n={n_hold})")
  ax.axvline(durs.mean(), color="#333", ls="--", lw=1.2,
             label=f"Mean {durs.mean():.1f} s")
  ax.set_xlabel("Episode duration (s)")
  ax.set_ylabel("Count")
  ax.set_title("(a) G1_Dex1_Wipe_Table episode lengths")
  ax.legend(fontsize=7)

  ax2 = axes[1]
  cats = ["Episodes", "Frames\n(×10³)", "Duration\n(min)"]
  train_vals = [
    n_train,
    split["train_frames"] / 1000.0,
    (durs[:n_train].sum() / 60.0),
  ]
  hold_vals = [
    n_hold,
    split["heldout_frames"] / 1000.0,
    (durs[n_train:].sum() / 60.0),
  ]
  x = np.arange(len(cats))
  w = 0.38
  b1 = ax2.bar(x - w / 2, train_vals, w, color=COLORS["train"], label="Train", edgecolor="white")
  b2 = ax2.bar(x + w / 2, hold_vals, w, color=COLORS["heldout"], label="Held-out", edgecolor="white")
  for bars in (b1, b2):
    for b in bars:
      ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
               f"{b.get_height():.1f}", ha="center", fontsize=7)
  ax2.set_xticks(x)
  ax2.set_xticklabels(cats)
  ax2.set_ylabel("Value")
  ax2.set_title("(b) Episode-level 80/20 split (no frame shuffle)")
  ax2.legend(fontsize=7)
  ax2.set_ylim(0, max(train_vals) * 1.25)

  total_h = float(card["total_hours"])
  fig.suptitle(
    f"Wipe-table corpus: {card['n_episodes']} eps, {card['num_rows']:,} frames, "
    f"~{total_h:.2f} h @ ~{card['fps_estimate']:.0f} Hz",
    fontweight="bold", y=1.02,
  )
  fig.tight_layout()
  return _save(fig, "fig3_dataset_distribution")


# ── Fig 4: Offline baselines ──────────────────────────────────
def fig4_offline_baselines() -> Path:
  rows = _load_json(BASELINE_JSON)
  esn = _esn_wipe_metrics()
  methods = ["zoh", "linear", "pid", "esn"]
  labels = ["ZOH", "Linear", "PID", "ESN (ours)"]
  means, stds, colors = [], [], []
  lookup = {r["method"]: r for r in rows}
  for m, c in zip(methods, [COLORS["zoh"], COLORS["linear"], COLORS["pid"], COLORS["esn"]]):
    if m == "esn":
      means.append(esn["heldout_rmse_mean"])
      stds.append(esn["heldout_rmse_std"])
    else:
      means.append(float(lookup[m]["rmse_mean"]))
      stds.append(float(lookup[m]["rmse_std"]))
    colors.append(c)

  fig, ax = plt.subplots(figsize=(4.8, 3.0))
  x = np.arange(len(labels))
  bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
                capsize=4, error_kw={"lw": 1.2})
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.set_ylabel("Held-out joint RMSE (rad)")
  ax.set_title("Open-loop upsample RMSE (40 held-out wipe eps)")
  ax.set_yscale("log")
  for b, m, s in zip(bars, means, stds):
    ax.text(b.get_x() + b.get_width() / 2, m * 1.25,
            f"{m:.2e}", ha="center", fontsize=7)
  lin = means[1]
  ax.annotate(
    f"ESN ≈ {lin / means[3]:.1f}× vs linear",
    xy=(3, means[3]), xytext=(1.2, means[0] * 0.55),
    fontsize=8, color=COLORS["esn"],
    arrowprops=dict(arrowstyle="->", color=COLORS["esn"]),
  )
  fig.tight_layout()
  return _save(fig, "fig4_offline_baselines")


# ── Fig 5: Hyperparam sweep ───────────────────────────────────
def fig5_hyperparam_sweep() -> Path:
  df = pd.read_csv(SWEEP_CSV)
  alphas = sorted(df["leaky_rate"].unique())
  lambdas = sorted(df["ridge_alpha"].unique())
  grid = np.full((len(alphas), len(lambdas)), np.nan)
  for _, r in df.iterrows():
    i = alphas.index(r["leaky_rate"])
    j = lambdas.index(r["ridge_alpha"])
    grid[i, j] = r["rmse"]

  fig, ax = plt.subplots(figsize=(4.6, 3.4))
  im = ax.imshow(np.log10(grid), cmap="RdYlGn_r", aspect="auto")
  cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
  cbar.set_label(r"$\log_{10}$ RMSE (rad)")
  ax.set_xticks(range(len(lambdas)))
  ax.set_xticklabels([f"{l:g}" for l in lambdas])
  ax.set_yticks(range(len(alphas)))
  ax.set_yticklabels([f"{a:g}" for a in alphas])
  ax.set_xlabel(r"Ridge $\lambda$")
  ax.set_ylabel(r"Leaky rate $\alpha$")
  ax.set_title("ESN leak×ridge sweep (wipe ep.0)")

  best = df.loc[df["selection_score"].idxmin()]
  bi = alphas.index(best["leaky_rate"])
  bj = lambdas.index(best["ridge_alpha"])
  ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                              edgecolor="#0D47A1", lw=2.5))
  for i in range(len(alphas)):
    for j in range(len(lambdas)):
      ax.text(j, i, f"{grid[i, j]:.2e}", ha="center", va="center", fontsize=6.5)
  ax.text(
    0.02, -0.18,
    fr"Selected $(\alpha,\lambda)=({best['leaky_rate']:g},{best['ridge_alpha']:g})$, "
    fr"RMSE={best['rmse']:.2e} rad",
    transform=ax.transAxes, fontsize=7.5, color="#0D47A1",
  )
  fig.tight_layout()
  return _save(fig, "fig5_hyperparam_sweep")


# ── Fig 6: Contact ladder ─────────────────────────────────────
def fig6_contact_ladder() -> Path:
  mj = _load_json(MUJOCO_JSON)
  esn = _load_json(LIVE_ESN)
  zoh = _load_json(LIVE_ZOH)
  oracle_contact = float(mj["table_contact_ratio_mean"]) * 100.0
  live_esn = float(esn["task_metrics"]["table_contact_ratio"]) * 100.0
  live_zoh = float(zoh["task_metrics"]["table_contact_ratio"]) * 100.0
  # Paper-reported collapse without press_table (disclosure; not a new run here).
  live_no_press = 5.0

  labels = [
    "MuJoCo oracle\n(demo tokens)",
    "Live UnifoLM\nw/o press_table*",
    "Live ESN\n+ press_table",
    "Live ZOH\n+ press_table",
  ]
  vals = [oracle_contact, live_no_press, live_esn, live_zoh]
  colors = [COLORS["esn"], "#9E9E9E", COLORS["esn"], COLORS["zoh"]]

  fig, ax = plt.subplots(figsize=(6.2, 3.0))
  bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.65)
  ax.set_ylim(0, 115)
  ax.set_ylabel("Table-contact ratio (%)")
  ax.set_title("Contact ladder: oracle plan vs live UnifoLM (+ priors)")
  for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%",
            ha="center", fontsize=8, fontweight="bold")
  ax.text(
    0.5, -0.22,
    "* ~5% without press_table is the measured collapse mode reported in the paper "
    "(cloth lift); live rows use proximity-synthetic Dex1 + press_table.",
    transform=ax.transAxes, ha="center", fontsize=6.5, color="#555555",
  )
  fig.tight_layout()
  return _save(fig, "fig6_contact_ladder")


# ── Fig 7: Dual-process rates ─────────────────────────────────
def fig7_control_rates() -> Path:
  df = pd.read_csv(DUAL_CSV)
  order = ["zoh", "linear", "pid", "esn"]
  labels = ["ZOH", "Linear", "PID", "ESN"]
  colors = [COLORS["zoh"], COLORS["linear"], COLORS["pid"], COLORS["esn"]]
  means, p99s, hz = [], [], []
  for m in order:
    row = df.loc[df["bridge"] == m].iloc[0]
    means.append(float(row["mean_step_ms"]))
    p99s.append(float(row["p99_step_ms"]))
    hz.append(float(row["achieved_control_hz"]))

  fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
  ax = axes[0]
  x = np.arange(len(labels))
  ax.bar(x - 0.18, means, 0.36, color=colors, label="Mean", edgecolor="white")
  ax.bar(x + 0.18, p99s, 0.36, color=colors, alpha=0.45, label="P99", edgecolor="white")
  ax.axhline(10, color=COLORS["req"], ls="--", lw=1.2, label="10 ms budget")
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.set_ylabel("Step time (ms)")
  ax.set_title("(a) Live dual-process step latency")
  ax.legend(fontsize=7)

  ax2 = axes[1]
  bars = ax2.bar(labels, hz, color=colors, edgecolor="white")
  ax2.axhline(100, color=COLORS["req"], ls="--", lw=1.2)
  for b, v in zip(bars, hz):
    ax2.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}",
             ha="center", fontsize=8, fontweight="bold")
  ax2.set_ylabel("Achieved control Hz")
  ax2.set_title("(b) Achieved rate (10 s live UnifoLM)")
  ax2.set_ylim(0, max(hz) * 1.2)

  fig.suptitle("GIL-free dual-process MuJoCo + bridge under live UnifoLM", fontweight="bold", y=1.02)
  fig.tight_layout()
  return _save(fig, "fig7_control_rates")


# ── Fig 8: Oracle held-out ────────────────────────────────────
def fig8_oracle_heldout() -> Path:
  df = pd.read_csv(MUJOCO_CSV)
  fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
  ax = axes[0]
  ax.hist(df["tracking_rmse"], bins=12, color=COLORS["esn"], edgecolor="white", alpha=0.9)
  ax.axvline(df["tracking_rmse"].mean(), color="darkgreen", ls="--", lw=1.6,
             label=f"Mean {df['tracking_rmse'].mean():.3e} rad")
  ax.set_xlabel("Joint tracking RMSE (rad)")
  ax.set_ylabel("Episodes")
  ax.set_title("(a) Dataset-oracle ESN RMSE (n=40)")
  ax.legend(fontsize=7)

  ax2 = axes[1]
  ax2.scatter(df["wipe_path_m"], df["table_contact_ratio"] * 100.0,
              c=COLORS["esn"], s=28, alpha=0.85, edgecolors="white", lw=0.5)
  ax2.axhline(df["table_contact_ratio"].mean() * 100.0, color="#555", ls=":",
              label=f"Mean contact {df['table_contact_ratio'].mean()*100:.1f}%")
  ax2.set_xlabel("Wipe path length (m)")
  ax2.set_ylabel("Table contact (%)")
  ax2.set_title("(b) Coverage vs contact (held-out)")
  ax2.set_ylim(0, 105)
  ax2.legend(fontsize=7)

  grasp = 100.0 * df["grasp_success"].mean()
  task = 100.0 * df["task_success"].mean()
  fig.suptitle(
    f"MuJoCo wipe oracle · grasp {grasp:.0f}% · task {task:.0f}% · "
    f"eps 160–199",
    fontweight="bold", y=1.02,
  )
  fig.tight_layout()
  return _save(fig, "fig8_oracle_heldout")


def write_manifest(paths: List[Path]) -> Path:
  out = FIGURES_DIR / "FIGURE_MANIFEST.json"
  payload = {
    "generated_from_measured": True,
    "mock": False,
    "figures": [p.name for p in paths],
    "sources": {
      "profiling": str(PROFILE_REPORT),
      "dataset_card": str(DATASET_CARD),
      "esn_seed": str(ESN_SEED),
      "baselines": str(BASELINE_JSON),
      "dual_thread": str(DUAL_CSV),
      "live_wipe_esn": str(LIVE_ESN),
      "mujoco_heldout": str(MUJOCO_JSON),
    },
    "paper_figures_dir": str(PAPER_FIG_DIR),
  }
  out.write_text(json.dumps(payload, indent=2) + "\n")
  (PAPER_FIG_DIR / "FIGURE_MANIFEST.json").write_text(out.read_text())
  # Replace placeholder README
  readme = FIGURES_DIR / "README.md"
  readme.write_text(
    "# ICRA paper figures (measured)\n\n"
    "Generated by `python3 -m src.step4_paper_figures`.\n"
    "Synced to `papers/icra2027/figures/`.\n"
    "Do not commit mock grids; regenerate from `results/` JSON/CSV.\n"
  )
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate measured ICRA paper figures")
  parser.add_argument(
    "--mock",
    action="store_true",
    help="Rejected: figures must come from measured results (exit nonzero)",
  )
  parser.add_argument(
    "--only",
    nargs="+",
    metavar="STEM",
    help="Generate only these stems (e.g. fig1_architecture)",
  )
  args = parser.parse_args()
  if args.mock:
    raise SystemExit(
      "Refusing --mock: ICRA figures must be regenerated from measured results/ artifacts."
    )

  generators = {
    "fig1_architecture": fig1_architecture,
    "fig2_latency_gap": fig2_latency_gap,
    "fig3_dataset_distribution": fig3_dataset_distribution,
    "fig4_offline_baselines": fig4_offline_baselines,
    "fig5_hyperparam_sweep": fig5_hyperparam_sweep,
    "fig6_contact_ladder": fig6_contact_ladder,
    "fig7_control_rates": fig7_control_rates,
    "fig8_oracle_heldout": fig8_oracle_heldout,
  }

  logger.info("Generating measured paper figures → %s", FIGURES_DIR)
  if args.only:
    missing = [s for s in args.only if s not in generators]
    if missing:
      raise SystemExit(f"Unknown figure stem(s): {missing}")
    paths = [generators[s]() for s in args.only]
  else:
    paths = [fn() for fn in generators.values()]
  write_manifest(paths)
  print("\nGenerated:")
  for p in paths:
    print(f"  {p}")
  print(f"Paper sync: {PAPER_FIG_DIR}")
  print(f"WRL sync:   {WRL_FIG_DIR}")


if __name__ == "__main__":
  main()
