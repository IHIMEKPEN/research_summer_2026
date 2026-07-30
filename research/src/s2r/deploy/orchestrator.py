"""Process orchestrator for easy multi-node deploy."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from s2r.core.config import load_config

console = Console()

NODE_MODULES = {
    "state_publisher": "s2r.nodes.state_publisher",
    "vla": "s2r.nodes.vla_node",
    "esn": "s2r.nodes.esn_engine",
    "passthrough": "s2r.nodes.passthrough_control",
    "reasoning": "s2r.nodes.reasoning_node",
    "sim_bridge": "s2r.nodes.sim_bridge",
    "robot_bridge": "s2r.nodes.robot_bridge",
    "g1_bridge": "s2r.nodes.g1_bridge",
    "camera": "s2r.nodes.camera_node",
    "vision": "s2r.nodes.vision_node",
    "mission": "s2r.nodes.mission_node",
    "data_collector": "s2r.nodes.data_collector",
    "gui": "s2r.gui.monitor",
}

# Entry attribute to call .run() on a Node subclass instance
NODE_ENTRY = {
    "state_publisher": "StatePublisherNode",
    "vla": "VLANode",
    "esn": "ESNUpsampleNode",
    "passthrough": "PassthroughControlNode",
    "reasoning": "ReasoningNode",
    "sim_bridge": "SimBridgeNode",
    "robot_bridge": "RobotBridgeNode",
    "g1_bridge": "G1BridgeNode",
    "camera": "CameraNode",
    "vision": "VisionNode",
    "mission": "MissionNode",
    "data_collector": "DataCollectorNode",
    "gui": "GUINode",
}


def _python() -> str:
    return sys.executable


def launch_node(name: str, config_path: str | None, env: dict[str, str] | None = None) -> subprocess.Popen:
    module = NODE_MODULES[name]
    cls = NODE_ENTRY[name]
    cfg_arg = repr(config_path) if config_path else "None"
    code = (
        f"from {module} import {cls}; "
        f"{cls}(config_path={cfg_arg}).run()"
    )
    cmd = [_python(), "-c", code]
    console.print(f"[green]launch[/] {name}")
    return subprocess.Popen(cmd, env=env or os.environ.copy())


def _engine_env(cfg: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    engine = str(cfg.get("pipeline", {}).get("control_engine", "esn")).lower()
    env["S2R_CONTROL_ENGINE"] = engine
    if engine in {"zoh", "linear", "raw"}:
        env["S2R_PASSTHROUGH_MODE"] = engine
    else:
        mode = str(cfg.get("passthrough", {}).get("mode", "zoh"))
        env["S2R_PASSTHROUGH_MODE"] = mode
    return env


class Orchestrator:
    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.procs: dict[str, subprocess.Popen] = {}

    def node_list(self) -> list[str]:
        nodes = list(self.cfg.get("deploy", {}).get("nodes", []))
        mode = self.cfg.get("pipeline", {}).get("mode", "sim")
        engine = str(self.cfg.get("pipeline", {}).get("control_engine", "esn")).lower()
        # Research ablation: swap ESN dynamic engine for passthrough baselines
        if engine in {"passthrough", "none", "no_esn", "zoh", "linear", "raw"}:
            nodes = [n for n in nodes if n != "esn"]
            if "passthrough" not in nodes:
                nodes.append("passthrough")
            # allow shorthand engine names to set passthrough mode if unset
            if engine in {"zoh", "linear", "raw"}:
                self.cfg.setdefault("passthrough", {})
                self.cfg["passthrough"].setdefault("mode", engine)
        elif engine == "esn":
            nodes = [n for n in nodes if n != "passthrough"]
            if "esn" not in nodes:
                nodes.append("esn")

        if mode == "sim":
            nodes = [n for n in nodes if n not in {"robot_bridge", "g1_bridge"}]
        if mode == "real":
            nodes = [n for n in nodes if n != "sim_bridge"]
        if mode == "g1":
            nodes = [n for n in nodes if n not in {"sim_bridge", "robot_bridge"}]
            if "g1_bridge" not in nodes:
                nodes.append("g1_bridge")
        return nodes

    def start(self, only: list[str] | None = None) -> None:
        names = only or self.node_list()
        # Prefer GUI bind first, then perception stack, then control
        priority = {
            "gui": 0,
            "camera": 1,
            "vision": 2,
            "mission": 3,
            "reasoning": 4,
            "state_publisher": 5,
        }
        ordered = sorted(names, key=lambda n: priority.get(n, 10))
        env = _engine_env(self.cfg)
        console.print(
            f"[cyan]control_engine[/]={env.get('S2R_CONTROL_ENGINE')} "
            f"passthrough_mode={env.get('S2R_PASSTHROUGH_MODE')}"
        )
        for name in ordered:
            if name not in NODE_MODULES:
                console.print(f"[red]unknown node[/] {name}")
                continue
            self.procs[name] = launch_node(name, self.config_path, env=env)
            time.sleep(0.15)
        self._print_status()

    def _print_status(self) -> None:
        table = Table(title="S2R Deploy")
        table.add_column("Node")
        table.add_column("PID")
        table.add_column("Alive")
        for name, p in self.procs.items():
            table.add_row(name, str(p.pid), "yes" if p.poll() is None else f"exit={p.returncode}")
        console.print(table)
        gui = self.cfg.get("gui", {})
        console.print(
            f"Monitor UI: [bold cyan]http://127.0.0.1:{gui.get('port', 8080)}[/]"
        )

    def stop(self) -> None:
        for name, p in self.procs.items():
            if p.poll() is None:
                console.print(f"[yellow]stop[/] {name} pid={p.pid}")
                p.send_signal(signal.SIGTERM)
        deadline = time.time() + 5
        for p in self.procs.values():
            while p.poll() is None and time.time() < deadline:
                time.sleep(0.05)
            if p.poll() is None:
                p.kill()
        self.procs.clear()

    def wait(self) -> None:
        try:
            while True:
                dead = [n for n, p in self.procs.items() if p.poll() is not None]
                if dead:
                    for n in dead:
                        console.print(f"[red]node exited[/] {n} code={self.procs[n].returncode}")
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            console.print("[yellow]shutting down…[/]")
        finally:
            self.stop()


def deploy(config_path: str | None = None, only: list[str] | None = None) -> None:
    # Ensure research/ and research/src on path for subprocess -c imports
    root = Path(__file__).resolve().parents[3]
    src = root / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(root) + os.pathsep + str(src) + os.pathsep + env.get("PYTHONPATH", "")
    )
    orch = Orchestrator(config_path)
    # monkeypatch launch to pass env
    global launch_node
    _orig = launch_node

    def _launch(name: str, config_path: str | None, env_inner: dict[str, str] | None = None) -> subprocess.Popen:
        return _orig(name, config_path, env=env)

    launch_node = _launch  # type: ignore
    orch.start(only=only)
    orch.wait()
