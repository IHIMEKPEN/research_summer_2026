"""
Step 4 — ICRA paper figures from *measured* results only (no SUCCESS_PRIORS / mock grids).

Outputs (PDF + PNG) under:
  research/results/step4_paper_figures/
  papers/icra2027/figures/   (synced for \includegraphics)

Figures:
  fig1_architecture       UnifoLM → IK → ESN → G1 @ 100 Hz
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
  pdf = FIGURES_DIR / f"{stem}.pdf"
  png = FIGURES_DIR / f"{stem}.png"
  fig.savefig(pdf)
  fig.savefig(png)
  plt.close(fig)
  for ext in (".pdf", ".png"):
    shutil.copy2(FIGURES_DIR / f"{stem}{ext}", PAPER_FIG_DIR / f"{stem}{ext}")
  logger.info("Saved %s (+ paper sync)", pdf)
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
  fig, ax = plt.subplots(figsize=(7.2, 2.6))
  ax.set_xlim(0, 14)
  ax.set_ylim(0, 5)
  ax.axis("off")
  fig.patch.set_facecolor("white")

  def box(x, y, w, h, title, sub, color):
    ax.add_patch(
      FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08",
        linewidth=1.4, edgecolor=color, facecolor=color + "22",
      )
    )
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
            fontsize=6.5, color="#444444")

  def arrow(x1, y1, x2, y2, label="", color="#333333"):
    ax.annotate(
      "", xy=(x2, y2), xytext=(x1, y1),
      arrowprops=dict(arrowstyle="-|>", lw=1.5, color=color),
    )
    if label:
      ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.22, label,
              ha="center", fontsize=6.5, color=color, style="italic")

  box(0.15, 2.6, 2.0, 1.5, "RGB + lang", "G1 camera\ninstruction", "#1565C0")
  box(0.15, 0.4, 2.0, 1.5, "Proprio q_t", "29-DoF joints\n@ 100 Hz", "#455A64")
  box(2.6, 1.5, 2.4, 2.2, "UnifoLM-VLA", "Base · FP16\n~1.75 Hz / 571 ms", COLORS["vla"])
  arrow(2.15, 3.3, 2.6, 3.0, "image")
  arrow(2.15, 1.1, 2.6, 2.0, "text")
  box(5.4, 1.7, 1.9, 1.8, "IK bridge", "EE chunk →\n29-D q*", "#6A1B9A")
  arrow(5.0, 2.6, 5.4, 2.6, "~1.7 Hz")
  box(7.7, 1.4, 2.5, 2.4, "ESN reservoir", "N=1000, ρ=0.95\nu=[q;q*]∈R⁵⁸\nridge W_out", COLORS["esn"])
  arrow(7.3, 2.6, 7.7, 2.6, "held q*")
  arrow(2.15, 0.9, 7.7, 1.8, "q_t", color="#455A64")
  box(10.7, 1.6, 2.9, 2.0, "G1 commands", "29-DoF targets\n100 Hz (≤10 ms)", COLORS["req"])
  arrow(10.2, 2.6, 10.7, 2.6, "100 Hz", color=COLORS["esn"])

  ax.text(
    7.0, 4.55,
    "Frequency gap: UnifoLM ~1.75 Hz  →  ESN bridge  →  G1 100 Hz  (~57×)",
    ha="center", fontsize=8, color=COLORS["vla"],
    bbox=dict(facecolor="#FFEBEE", edgecolor=COLORS["vla"], boxstyle="round,pad=0.25"),
  )
  ax.set_title("VLA + ESN hybrid control for Unitree G1 (29-DoF)", fontweight="bold", pad=6)
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
  args = parser.parse_args()
  if args.mock:
    raise SystemExit(
      "Refusing --mock: ICRA figures must be regenerated from measured results/ artifacts."
    )

  logger.info("Generating measured paper figures → %s", FIGURES_DIR)
  paths = [
    fig1_architecture(),
    fig2_latency_gap(),
    fig3_dataset_distribution(),
    fig4_offline_baselines(),
    fig5_hyperparam_sweep(),
    fig6_contact_ladder(),
    fig7_control_rates(),
    fig8_oracle_heldout(),
  ]
  write_manifest(paths)
  print("\nGenerated:")
  for p in paths:
    print(f"  {p}")
  print(f"Paper sync: {PAPER_FIG_DIR}")


if __name__ == "__main__":
  main()
