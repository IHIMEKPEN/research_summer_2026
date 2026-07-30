from s2r.experiments.benchmark import load_tasks, score_episode, save_score, summarize_results, make_leaderboard
from s2r.experiments.paths import ensure_experiment_dirs, BENCHMARK_RESULTS


def test_load_twelve_tasks():
    tasks = load_tasks()
    assert len(tasks) == 12
    assert tasks[0].id.startswith("01_")
    assert tasks[-1].id.startswith("12_")
    assert tasks[4].hf_dataset.endswith("G1_Pack_PencilBox")


def test_score_roundtrip(tmp_path, monkeypatch):
    ensure_experiment_dirs()
    # redirect results into tmp
    monkeypatch.setattr("s2r.experiments.benchmark.BENCHMARK_RESULTS", tmp_path)
    s = score_episode("01_stack_block", "unit_test_model", "e1", success=True, esn_hz=100)
    path = save_score(s, out_dir=tmp_path)
    assert path.exists()
    rows = summarize_results(results_dir=tmp_path)
    assert rows[0]["success_rate"] == 1.0
    board = make_leaderboard([s])
    assert board[0]["model"] == "unit_test_model"
