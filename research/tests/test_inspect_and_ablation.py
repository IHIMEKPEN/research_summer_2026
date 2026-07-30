from pathlib import Path

from s2r.experiments.inspect_data import inspect_dataset, robotics_fitness, extract_series, load_jsonl_rows
from s2r.experiments.ablation import analyze_run, compare_runs
from s2r.experiments.paths import RAW
from s2r.deploy.orchestrator import Orchestrator


def test_inspect_raw_if_present():
    if not any(RAW.glob("*.jsonl")) and not any(RAW.rglob("*.jsonl")):
        return
    report = inspect_dataset(RAW)
    assert "rates_hz" in report
    assert "robotics_fitness" in report
    assert "distributions" in report


def test_robotics_fitness_checklist_keys():
    rows = [
        {"topic": "state", "ts": 1.0, "payload": {"joint_pos": [0.1] * 29, "joint_vel": [0.0] * 29}},
        {"topic": "action_token", "ts": 1.5, "payload": {"action": [0.2] * 29}},
        {"topic": "joint_cmd", "ts": 1.51, "payload": {"q": [0.2] * 29, "source": "esn"}},
        {"topic": "joint_cmd", "ts": 1.52, "payload": {"q": [0.21] * 29, "source": "esn"}},
        {"topic": "joint_cmd", "ts": 1.53, "payload": {"q": [0.22] * 29, "source": "esn"}},
        {"topic": "decision", "ts": 1.5, "payload": {"intent": "explore", "latency_ms": 10.0}},
    ]
    series = extract_series(rows)
    fit = robotics_fitness(series, joint_limits={"min": [-3.14] * 29, "max": [3.14] * 29})
    assert "has_state" in fit.checklist
    assert fit.score >= 0.0


def test_orchestrator_swaps_esn_for_passthrough(tmp_path: Path):
    cfg = tmp_path / "no_esn.yaml"
    cfg.write_text(
        """
pipeline: {mode: sim, control_engine: zoh}
zmq: {state_pub: tcp://127.0.0.1:5991, action_token_pub: tcp://127.0.0.1:5992, joint_cmd_pub: tcp://127.0.0.1:5993, decision_pub: tcp://127.0.0.1:5994, map_pub: tcp://127.0.0.1:5995, data_push: tcp://127.0.0.1:5996, gui_pub: tcp://127.0.0.1:5997, perception_pub: tcp://127.0.0.1:5998, mission_pub: tcp://127.0.0.1:5999, camera_pub: tcp://127.0.0.1:6000}
deploy:
  nodes: [vla, esn, sim_bridge]
passthrough: {mode: zoh, target_hz: 50}
robot: {n_joints: 29, control_hz: 50}
"""
    )
    orch = Orchestrator(str(cfg))
    nodes = orch.node_list()
    assert "passthrough" in nodes
    assert "esn" not in nodes


def test_compare_runs_smoke(tmp_path: Path):
    # tiny synthetic logs
    a = tmp_path / "esn.jsonl"
    b = tmp_path / "raw.jsonl"
    a.write_text(
        "\n".join(
            [
                '{"topic":"action_token","ts":1.0,"payload":{"action":[0,0,0,0,0,0,0]}}',
                '{"topic":"joint_cmd","ts":1.01,"payload":{"q":[0.1,0,0,0,0,0,0],"source":"esn"}}',
                '{"topic":"joint_cmd","ts":1.02,"payload":{"q":[0.11,0,0,0,0,0,0],"source":"esn"}}',
                '{"topic":"joint_cmd","ts":1.03,"payload":{"q":[0.12,0,0,0,0,0,0],"source":"esn"}}',
                '{"topic":"state","ts":1.03,"payload":{"joint_pos":[0.12,0,0,0,0,0,0]}}',
            ]
        )
    )
    b.write_text(
        "\n".join(
            [
                '{"topic":"action_token","ts":1.0,"payload":{"action":[0,0,0,0,0,0,0]}}',
                '{"topic":"joint_cmd","ts":1.0,"payload":{"q":[0.0,0,0,0,0,0,0],"source":"passthrough_raw"}}',
                '{"topic":"joint_cmd","ts":1.5,"payload":{"q":[1.0,0,0,0,0,0,0],"source":"passthrough_raw"}}',
                '{"topic":"state","ts":1.5,"payload":{"joint_pos":[0.5,0,0,0,0,0,0]}}',
            ]
        )
    )
    report = compare_runs({"esn": a, "raw": b})
    assert "esn" in report["runs"]
    assert len(report["ranking_smoothest_to_roughest"]) == 2
