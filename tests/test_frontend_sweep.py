"""Frontend sweep tests generate commands/fixtures only; never run inference."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools import sweep_frontend as sweep


@pytest.fixture
def signature(monkeypatch):
    record = {"runner_sha256": "runner", "checkpoint_sha256_expected": "checkpoint",
              "similarity": {"layer": 15, "tau": .3764, "config_file": "fixed.json", "config_sha256": "fixed"}}
    monkeypatch.setattr(sweep, "runtime_signature", lambda backend: copy.deepcopy(record))
    return record


def arguments(tmp_path, *extra):
    return sweep.parse_args(["--data-root", str(tmp_path / "data"), "--out", str(tmp_path / "sweep"), *extra])


def completed_metadata(config, design):
    return {"status": "complete", "backend": design["backend"],
            "args": {**sweep.CORE_DEFAULTS, "input": design["sequence_path"], "backend": design["backend"],
                     "submap_size": config["submap_size"], "min_disparity": config["min_disparity"],
                     "prediction_cache": config["prediction_cache"], "max_cloud_points": 2000000},
            "checkpoints": {design["backend"]: {"sha256_expected": "checkpoint"}},
            "similarity": {"layer": 15, "tau": .3764}, "summary": {"keyframes": 3},
            "updates": [{"frame_names": ["1.0.png", "2.0.png", "3.0.png"],
                         "graph_nodes": 3, "loops_accepted_in_update": 0}],
            "ate": {"status": "ok", "ate_rmse_m": .04, "n_associated": 3}}


def write_completed(config, design):
    output, cache = Path(config["output"]), Path(config["prediction_cache"])
    output.mkdir(parents=True)
    cache.mkdir(parents=True)
    metadata = completed_metadata(config, design)
    (output / "metadata.json").write_text(json.dumps(metadata))
    (output / "poses.tum").write_text("1 0 0 0 0 0 0 1\n2 1 0 0 0 0 0 1\n3 1 1 0 0 0 0 1\n")
    np.savez_compressed(cache / "00000.npz", __metadata__=np.asarray(json.dumps({
        "schema": "vos-prediction-cache-v1", "submap_id": 0,
        "frame_names": metadata["updates"][0]["frame_names"], "values": {"detected_loops": []}})))


def selection_records():
    return [{**c, "status": "complete", "ate_rmse_m": .04, "eligible_for_accuracy_selection": True}
            for c in sweep.GRID]


def test_offload_changes_storage_command_without_changing_selected_geometry(tmp_path, signature):
    ordinary = sweep.make_design(arguments(tmp_path))
    offloaded = sweep.make_design(arguments(tmp_path, "--offload-keyframes"))
    assert offloaded["offload_keyframes"] is True
    for before, after in zip(ordinary["configs"], offloaded["configs"]):
        assert after["argv"] == before["argv"] + ["--offload-keyframes"]
        assert (before["submap_size"], before["min_disparity"]) == (after["submap_size"], after["min_disparity"])
    baseline = completed_metadata(offloaded["configs"][1], offloaded)
    baseline["args"]["offload_keyframes"] = False
    assert sweep.validate_metadata(baseline, offloaded["configs"][1], offloaded) == .04


def test_six_commands_keep_backend_fixed_and_record_resource_limits(tmp_path, signature):
    design = sweep.make_design(arguments(tmp_path))
    assert len(design["configs"]) == 6
    assert {(c["submap_size"], c["min_disparity"]) for c in design["configs"]} == {
        (n, d) for n in (16, 32) for d in (25, 50, 75)}
    for c in design["configs"]:
        command = c["argv"]
        def value(flag):
            return command[command.index(flag) + 1]
        assert value("--backend-policy") == "legacy"
        assert value("--seed") == "0" and value("--max-loops") == "1"
        assert value("--max-cloud-points") == "100000"
        assert value("--gpu-need-mb") == ("9000" if c["submap_size"] == 16 else "9500")
        assert value("--prediction-cache") == c["prediction_cache"]


def test_default_design_never_launches_subprocess(tmp_path, signature, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Design-only execution attempted to launch inference")
    monkeypatch.setattr(sweep.subprocess, "run", forbidden)
    assert sweep.main(["--data-root", str(tmp_path / "data"), "--out", str(tmp_path / "plan")]) == 0
    result = json.loads((tmp_path / "plan/results.json").read_text())
    assert result["executed"] is False and len(result["commands"]) == 6
    assert not (tmp_path / "plan/frozen_selection.json").exists()


def test_environment_preserves_vos_and_explicit_thread_limits():
    source = {"VOS_OMEGA_CKPT": "/custom/model.pt", "OMP_NUM_THREADS": "2", "HOME": "/original"}
    result = sweep.inherited_environment(4, source)
    assert result["VOS_OMEGA_CKPT"] == source["VOS_OMEGA_CKPT"]
    assert result["OMP_NUM_THREADS"] == "2" and result["MKL_NUM_THREADS"] == "4"
    assert result["HOME"] == "/original" and "MKL_NUM_THREADS" not in source


def test_selection_excludes_oom_scores_and_ties_prefer_baseline():
    records = selection_records()
    records[0].update(status="hardware_infeasible", ate_rmse_m=0., eligible_for_accuracy_selection=False)
    selected, baseline = sweep.choose_configuration(records)
    assert selected["id"] == baseline["id"] == "s16_d50"
    records[-1]["ate_rmse_m"] = .035
    assert sweep.choose_configuration(records)[0]["id"] == "s32_d75"
    records[-1]["status"] = "planned"
    with pytest.raises(ValueError, match="terminal"):
        sweep.choose_configuration(records)


def test_failed_baseline_prevents_selection():
    records = selection_records()
    next(r for r in records if r["id"] == sweep.BASELINE_ID).update(
        status="failed", ate_rmse_m=None, eligible_for_accuracy_selection=False)
    with pytest.raises(ValueError, match="baseline"):
        sweep.choose_configuration(records)


def test_completed_metadata_rejects_different_frontend_conditions(tmp_path, signature):
    design = sweep.make_design(arguments(tmp_path))
    config = next(c for c in design["configs"] if c["id"] == sweep.BASELINE_ID)
    metadata = completed_metadata(config, design)
    assert sweep.validate_metadata(metadata, config, design) == .04
    for key, value in (("min_disparity", 25), ("seed", 1), ("backend_policy", "omega_confidence"), ("max_frames", 100)):
        changed = copy.deepcopy(metadata)
        changed["args"][key] = value
        with pytest.raises(ValueError, match="mismatch"):
            sweep.validate_metadata(changed, config, design)
    changed = copy.deepcopy(metadata)
    changed["similarity"]["tau"] = .95
    with pytest.raises(ValueError, match="calibration"):
        sweep.validate_metadata(changed, config, design)


def test_cache_verifies_loop_ids_frames_and_temporary_files(tmp_path, signature):
    design = sweep.make_design(arguments(tmp_path))
    config = design["configs"][0]
    write_completed(config, design)
    metadata = completed_metadata(config, design)
    assert sweep.validate_cache(config["prediction_cache"], metadata)["n_updates"] == 1
    changed = copy.deepcopy(metadata)
    changed["updates"][0]["frame_names"][0] = "wrong.png"
    with pytest.raises(ValueError, match="frame identities"):
        sweep.validate_cache(config["prediction_cache"], changed)
    (Path(config["prediction_cache"]) / "00001.tmp.npz").write_bytes(b"unfinished")
    with pytest.raises(ValueError, match="temporary"):
        sweep.validate_cache(config["prediction_cache"], metadata)


def test_completed_resume_is_frozen_and_launches_nothing(tmp_path, signature, monkeypatch):
    args = arguments(tmp_path, "--run")
    design = sweep.make_design(args)
    sequence = Path(design["sequence_path"])
    (sequence / "rgb").mkdir(parents=True)
    (sequence / "groundtruth.txt").write_text("fixture")
    for config in design["configs"]:
        write_completed(config, design)
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: pytest.fail("Completed output was relaunched"))
    first = sweep.execute_sweep(args)
    frozen_file = args.out / "frozen_selection.json"
    before = frozen_file.read_bytes()
    second = sweep.execute_sweep(args)
    assert first["status"] == second["status"] == "complete"
    assert frozen_file.read_bytes() == before
    baseline = json.loads(before)["baseline"]
    assert baseline["n_keyframes"] == baseline["n_pose_rows"] == 3
    assert baseline["first_timestamp"] == 1 and baseline["last_timestamp"] == 3


def test_transient_launch_fields_do_not_change_frozen_selection(tmp_path, signature):
    design = sweep.make_design(arguments(tmp_path))
    records = selection_records()
    original = sweep.frozen_selection(design, records)
    for record in records:
        record.update(returncode=0, launch_log="console.log", command="python run_slam.py")
    assert sweep.frozen_selection(design, records) == original


def test_validation_uses_frozen_configuration_and_cannot_reselect(tmp_path, signature):
    design = sweep.make_design(arguments(tmp_path))
    records = selection_records()
    records[-1]["ate_rmse_m"] = .035
    frozen = sweep.frozen_selection(design, records)
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(frozen))
    args = arguments(tmp_path, "--sequence", "room", "--frozen-selection", str(path))
    validation = sweep.make_design(args)
    assert validation["phase"] == "validation"
    assert {c["id"] for c in validation["configs"] if c["scheduled"]} == {"s16_d50", "s32_d75"}
    with pytest.raises(ValueError, match="Validation"):
        sweep.frozen_selection(validation, records)
    assert json.loads(path.read_text()) == frozen


@pytest.mark.parametrize("text,code,status", [
    ("torch.OutOfMemoryError: CUDA out of memory", 1, "hardware_infeasible"),
    ("MemoryError", 1, "hardware_infeasible"),
    ("killed with no OOM evidence", -9, "terminated"),
    ("invalid input", 1, "failed"),
])
def test_resource_failures_are_not_accuracy_scores(text, code, status):
    assert sweep.classify_failure(text, code) == status
