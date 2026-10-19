"""CPU tests for tools/compare_summarize.py on synthetic run directories."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import compare_summarize as cs  # noqa: E402


def _make_run(d, backend="vggt", forwards=(2.0, 2.2, 0.5), frames=(17, 17, 4), status="complete", with_gt=False):
    d.mkdir(parents=True)
    updates = [{"update": i, "n_frames": n, "forward_seconds": f, "lc_forward_seconds": None, "optimize_seconds": 0.01,
                "input_shape": [n, 3, 294, 518], "peak_cuda_allocated_bytes": 5 * 1024 ** 3,
                "peak_cuda_reserved_bytes": 6 * 1024 ** 3, "loop_candidates": 0} for i, (f, n) in enumerate(zip(forwards, frames))]
    meta = {"status": status, "args": {"submap_size": 16, "overlap": 1}, "input": {"n_images_used": 100, "groundtruth": None},
            "updates": updates, "loop_log": [], "similarity": {"layer": 21}, "seed": 0, "peak_rss_kib": 4 * 1024 ** 2,
            "summary": {"keyframes": 38, "submaps_processed": 3, "loop_candidates_total": 0, "loop_closures_accepted": 0,
                        "total_seconds": 10.0, "optimize_seconds_total": 0.03, "peak_cuda_allocated_bytes": 5 * 1024 ** 3,
                        "peak_cuda_reserved_bytes": 6 * 1024 ** 3, "upstream_style_timers": {"keyframe_selection_total": 1.0}}}
    if status != "complete":
        meta["error"] = "RuntimeError: boom"
    (d / "metadata.json").write_text(json.dumps(meta))
    t = np.arange(10, dtype=float)
    rows = [f"{s} {np.cos(s / 9 * 6.28):.4f} {np.sin(s / 9 * 6.28):.4f} 0 0 0 0 1" for s in t]
    (d / "poses.tum").write_text("\n".join(rows) + "\n")


def test_run_record_timing_and_proxy(tmp_path):
    _make_run(tmp_path / "office_loop_vggt_loops1")
    rec = cs.run_record(tmp_path, "office_loop", "vggt", 1)
    assert rec["status"] == "complete" and rec["keyframes"] == 38
    assert rec["forward_total_s"] == pytest.approx(4.7)
    assert rec["forward_median_per_submap_s"] == pytest.approx(2.0)
    assert rec["forward_median_per_full_submap_s"] == pytest.approx(2.1)  # median of the two 17-frame submaps
    assert rec["fps_keyframes_forward"] == pytest.approx(38 / 4.7)
    assert rec["peak_cuda_allocated_gib"] == pytest.approx(5.0)
    assert rec["start_end"]["gap_fraction"] < 0.2  # closed circle


def test_failed_and_missing_runs_are_reported(tmp_path):
    _make_run(tmp_path / "fr1_desk_omega_loops1", status="failed")
    bad = cs.run_record(tmp_path, "fr1_desk", "omega", 1)
    assert bad["status"] == "failed" and "boom" in bad["error"]
    assert cs.run_record(tmp_path, "fr1_desk", "vggt", 1)["status"] == "missing"


def test_paper_numbers_match_table_one():
    assert (cs.PAPER["fr1_desk"], cs.PAPER["fr1_desk2"], cs.PAPER["fr1_room"]) == (0.025, 0.029, 0.063)
    assert np.mean(list(cs.PAPER["all_9_sequences"].values())) == pytest.approx(0.041, abs=5e-4)
