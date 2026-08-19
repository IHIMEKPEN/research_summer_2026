"""Typer CLI: deploy, run nodes, train, collect."""

from __future__ import annotations
from s2r.robot import G1_DOF

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="S2R ultra low-latency real-to-sim robotics pipeline")
console = Console()


def _ensure_path() -> None:
    # Prefer research/ + research/src on sys.path (Steps 1–4 + s2r).
    here = Path(__file__).resolve()
    research_root = here.parents[2]  # …/research
    src = here.parents[1]  # …/research/src
    for p in (str(research_root), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)


@app.command()
def deploy(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to YAML config"),
    only: Optional[str] = typer.Option(None, "--only", help="Comma-separated node names"),
) -> None:
    """Launch the full pipeline (or a subset of nodes)."""
    _ensure_path()
    from s2r.deploy.orchestrator import Orchestrator, launch_node
    import s2r.deploy.orchestrator as orch_mod

    research_root = Path(__file__).resolve().parents[2]
    src = Path(__file__).resolve().parents[1]
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = (
        str(research_root) + os.pathsep + str(src) + os.pathsep + base_env.get("PYTHONPATH", "")
    )

    def _launch(name, config_path, env_inner=None):
        merged = dict(base_env)
        if env_inner:
            merged.update(env_inner)
            # keep PYTHONPATH from base
            merged["PYTHONPATH"] = base_env["PYTHONPATH"]
        return launch_node(name, config_path, env=merged)

    orch_mod.launch_node = _launch  # type: ignore
    names = [x.strip() for x in only.split(",")] if only else None
    orch = Orchestrator(config)
    orch.start(only=names)
    orch.wait()


@app.command("run-node")
def run_node(
    name: str = typer.Argument(..., help="Node name"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Run a single node in the foreground."""
    _ensure_path()
    from s2r.deploy.orchestrator import NODE_ENTRY, NODE_MODULES

    if name not in NODE_MODULES:
        console.print(f"[red]Unknown node[/] {name}. Choices: {', '.join(NODE_MODULES)}")
        raise typer.Exit(1)
    module = NODE_MODULES[name]
    cls_name = NODE_ENTRY[name]
    mod = __import__(module, fromlist=[cls_name])
    cls = getattr(mod, cls_name)
    cls(config_path=config).run()


@app.command()
def train(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    raw_dir: Optional[str] = typer.Option(None, "--raw-dir"),
    out: Optional[str] = typer.Option(None, "--out"),
) -> None:
    """Train ESN upsampler from collected episodes."""
    _ensure_path()
    from s2r.training.train_esn import train_esn

    train_esn(config_path=config, raw_dir=raw_dir, out_path=out)


@app.command()
def synth(
    seconds: float = typer.Option(10.0, help="Seconds of synthetic demo data"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Generate synthetic paired token/command data for ESN bring-up."""
    _ensure_path()
    import json
    import math
    import time
    from s2r.core.config import load_config

    cfg = load_config(config)
    n = int(cfg.get("robot", {}).get("n_joints", G1_DOF))
    out_dir = Path(cfg.get("data_collection", {}).get("out_dir", "data/raw"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"episode_synth_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    t0 = time.time()
    rows = 0
    q = [0.0] * n
    with path.open("w", encoding="utf-8") as f:
        t = 0.0
        dt = 0.01
        last_action = [0.0] * n
        while t < seconds:
            if int(round(t * 100)) % 50 == 0:  # ~2 Hz tokens
                last_action = [0.4 * math.sin(2 * math.pi * 0.05 * t + 0.3 * i) for i in range(n)]
                f.write(
                    json.dumps(
                        {
                            "ts": t0 + t,
                            "topic": "action_token",
                            "source": "synth",
                            "seq": rows,
                            "payload": {"action": last_action, "confidence": 0.9, "goal": "synth"},
                        }
                    )
                    + "\n"
                )
                rows += 1
            alpha = 0.2
            q = [(1 - alpha) * cq + alpha * a for cq, a in zip(q, last_action)]
            f.write(
                json.dumps(
                    {
                        "ts": t0 + t,
                        "topic": "joint_cmd",
                        "source": "synth",
                        "seq": rows,
                        "payload": {"q": q, "source": "synth"},
                    }
                )
                + "\n"
            )
            rows += 1
            t += dt
    console.print(f"[green]Wrote[/] {rows} rows → {path}")


@app.command("inspect-data")
def inspect_data_cmd(
    source: Optional[str] = typer.Option(None, "--source", "-s", help="JSONL file or directory"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    out: str = typer.Option("data/processed/data_inspect.json", "--out"),
) -> None:
    """Inspect episode distributions and robotics fitness of a dataset."""
    _ensure_path()
    from pathlib import Path as P

    from s2r.core.config import load_config
    from s2r.experiments.inspect_data import inspect_benchmark_coverage, inspect_dataset, save_report
    from s2r.experiments.paths import RAW

    cfg = load_config(config)
    limits = cfg.get("robot", {}).get("joint_limits")
    src = P(source) if source else RAW
    report = inspect_dataset(src, joint_limits=limits)
    report["benchmark_coverage"] = inspect_benchmark_coverage()
    path = save_report(report, name=P(out).name)
    fit = report["robotics_fitness"]
    console.print(f"[bold]rows[/]={report['n_rows']} files={report['n_files']}")
    console.print(f"[bold]rates_hz[/]={report['rates_hz']}")
    console.print(f"[bold]robotics_applies[/]={fit['applies']} score={fit['score']:.2f}")
    for r in fit["reasons"]:
        console.print(f"  • {r}")
    for w in fit["warnings"]:
        console.print(f"  [yellow]warn[/] {w}")
    console.print(f"[green]wrote[/] {path}")


@app.command("compare-ablation")
def compare_ablation_cmd(
    with_esn: Optional[str] = typer.Option(None, "--with-esn", help="Episode/dir from ESN run"),
    no_esn: Optional[str] = typer.Option(None, "--no-esn", help="Episode/dir from passthrough run"),
    raw: Optional[str] = typer.Option(None, "--raw", help="Optional raw 2Hz run"),
    out: str = typer.Option("data/processed/esn_ablation_compare.json", "--out"),
) -> None:
    """Compare ESN vs no-ESN (passthrough) logged episodes for research."""
    _ensure_path()
    from pathlib import Path as P

    from s2r.experiments.ablation import compare_runs, save_ablation_report
    from s2r.experiments.paths import RAW

    paths = {}
    if with_esn:
        paths["esn"] = with_esn
    if no_esn:
        paths["passthrough"] = no_esn
    if raw:
        paths["raw"] = raw
    if not paths:
        # auto-discover common ablation folders if present
        for label, rel in [
            ("esn", RAW / "ablation_with_esn"),
            ("passthrough_zoh", RAW / "ablation_no_esn_zoh"),
            ("passthrough_raw", RAW / "ablation_no_esn_raw"),
        ]:
            if rel.exists() and any(rel.glob("*.jsonl")):
                paths[label] = rel
    if not paths:
        console.print("[red]No ablation logs found. Pass --with-esn/--no-esn or run ablation configs first.[/]")
        raise typer.Exit(1)
    report = compare_runs(paths)
    path = save_ablation_report(report, name=P(out).name)
    console.print("[bold]ranking (smoothest → roughest):[/]", ", ".join(report["ranking_smoothest_to_roughest"]))
    for label, run in report["runs"].items():
        console.print(
            f"  {label}: cmd_hz={run['rates_hz']['joint_cmd']:.1f} "
            f"token_hz={run['rates_hz']['action_token']:.1f} "
            f"jerk_p95={run['cmd_jerk_proxy'].get('p95')} "
            f"track_err={run['tracking_err_mean']}"
        )
    console.print(f"[green]wrote[/] {path}")


@app.command()
def profile(
    platform: str = typer.Option("auto", help="auto|v100|thor|generic"),
    backend: str = typer.Option("auto", help="auto|mock|real"),
    out: str = typer.Option("data/processed/profile_report.json"),
    skip_vlm: bool = typer.Option(False, "--skip-vlm"),
) -> None:
    """Profile YOLO / Qwen / ESN on the current GPU (V100, Thor, etc.)."""
    _ensure_path()
    import subprocess
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "profile_models.py"
    # parents[2] from src/s2r/cli.py = project root? parents[0]=s2r, [1]=src, [2]=root — yes
    cmd = [
        sys.executable,
        str(script),
        "--platform",
        platform,
        "--backend",
        backend,
        "--out",
        out,
    ]
    if skip_vlm:
        cmd.append("--skip-vlm")
    raise SystemExit(subprocess.call(cmd))


@app.command("sense-check")
def sense_check_cmd(
    camera: str = typer.Option("0", "--camera", help="OpenCV index or path (/dev/video0)"),
    seconds: float = typer.Option(10.0, "--seconds", help="How long to sample the camera"),
    width: int = typer.Option(640, "--width"),
    height: int = typer.Option(480, "--height"),
    show: bool = typer.Option(False, "--show", help="Open a live OpenCV window (needs display)"),
    out: str = typer.Option("results/sense_check/latest.json", "--out"),
    skip_lidar: bool = typer.Option(False, "--skip-lidar"),
) -> None:
    """Stage −1: live camera (+ optional LiDAR) and measure ingest / process rates."""
    _ensure_path()
    from s2r.experiments.sense_check import run_sense_check

    report = run_sense_check(
        camera=camera,
        seconds=seconds,
        width=width,
        height=height,
        show=show,
        out=out,
        skip_lidar=skip_lidar,
    )
    cam = report["camera"]
    lidar = report["lidar"]
    if cam.get("ok"):
        console.print(
            f"[green]camera ok[/] source={cam['source']} "
            f"{cam['width']}x{cam['height']} "
            f"capture_hz={cam['capture_hz']:.1f} "
            f"grab_ms={cam['mean_grab_ms']:.1f} process_ms={cam['mean_process_ms']:.1f}"
        )
        if cam.get("sample_path"):
            console.print(f"  sample → {cam['sample_path']}")
        if cam.get("devices"):
            console.print(f"  /dev video devices: {', '.join(cam['devices'])}")
    else:
        console.print(f"[red]camera failed[/] {cam.get('error')}")
        if cam.get("devices"):
            console.print(f"  devices seen: {', '.join(cam['devices'])}")
    if lidar.get("ok"):
        console.print(
            f"[green]lidar ok[/] backend={lidar['backend']} "
            f"topic={lidar.get('topic_or_channel')} hz≈{lidar.get('hz', 0):.1f}"
        )
    else:
        console.print(f"[yellow]lidar[/] backend={lidar.get('backend')} — {lidar.get('error') or 'not streaming'}")
    for note in lidar.get("notes") or []:
        console.print(f"  • {note}")
    console.print(f"[green]wrote[/] {report.get('out', out)}")
    if not cam.get("ok"):
        raise typer.Exit(1)


@app.command()
def version() -> None:
    from s2r import __version__

    console.print(__version__)


if __name__ == "__main__":
    app()
