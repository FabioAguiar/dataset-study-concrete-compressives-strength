import copy
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts.select_models import (
    ArtifactConflictError,
    REGRESSION_ARTIFACT_FILENAMES,
    load_and_validate_model_selection_handoff,
    validate_regression_model_selection_contract,
    write_regression_model_selection_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/model-selection/concrete-compressive-strength"


def _artifacts():
    result = {}
    for name in REGRESSION_ARTIFACT_FILENAMES:
        path = SOURCE / name
        result[name] = (pd.read_csv(path) if name.endswith(".csv")
                        else json.loads(path.read_text()))
    return result


def test_contract_unit_is_generic():
    validated = validate_regression_model_selection_contract({
        "problem_type": "continuous_regression",
        "target_semantics": "Continuous / quantitative",
        "target_unit": "kWh",
        "primary_metric": "mae",
        "primary_metric_direction": "lower_is_better",
        "refit_metric": "mae",
        "cv": {"strategy": "KFold", "n_splits": 5, "shuffle": True, "random_state": 42},
        "test_partition_sealed": True,
        "test_partition_evaluated": False,
    })
    assert validated["target_unit"] == "kWh"


def test_valid_write_reload_and_equivalent_reuse(tmp_path):
    out = tmp_path / "artifacts/model-selection/concrete-compressive-strength"
    first = write_regression_model_selection_artifacts(output_directory=out, artifacts=_artifacts())
    assert not first.idempotent
    second = write_regression_model_selection_artifacts(output_directory=out, artifacts=_artifacts())
    assert second.idempotent
    # The public loader needs the authenticated preparation lineage too.
    shutil.copytree(ROOT / "artifacts/preparation", tmp_path / "artifacts/preparation")
    shutil.copytree(ROOT / "artifacts/exploration", tmp_path / "artifacts/exploration")
    shutil.copytree(ROOT / "data", tmp_path / "data")
    loaded = load_and_validate_model_selection_handoff(
        project_root=tmp_path,
        handoff_path="artifacts/model-selection/concrete-compressive-strength/model-selection-handoff.json",
    )
    assert loaded["selected_model_id"] == "hist_gradient_boosting"


@pytest.mark.parametrize("mutation", ["target", "metric", "cv", "family", "feature", "winner"])
def test_writer_fails_closed_for_scientific_manifest_changes(tmp_path, mutation):
    artifacts = _artifacts()
    out = tmp_path / "selection"
    write_regression_model_selection_artifacts(output_directory=out, artifacts=artifacts)
    changed = _artifacts()
    manifest = changed["model-selection-manifest.json"]
    if mutation == "target": manifest["target_contract"]["unit"] = "kWh"
    elif mutation == "metric": manifest["model_selection_contract"]["primary_metric"] = "rmse"
    elif mutation == "cv": manifest["cv_contract"]["n_splits"] = 4
    elif mutation == "family": manifest["candidate_families"][0]["search_space"] = {"model__alpha": [7]}
    elif mutation == "feature": manifest["feature_contract"]["selected_feature_policy"] = "subset"
    else: manifest["final_model_trained"] = True
    with pytest.raises(ArtifactConflictError):
        write_regression_model_selection_artifacts(output_directory=out, artifacts=changed)


def test_writer_reuses_volatile_metadata_only(tmp_path):
    artifacts = _artifacts(); out = tmp_path / "selection"
    write_regression_model_selection_artifacts(output_directory=out, artifacts=artifacts)
    changed = _artifacts()
    changed["model-selection-manifest.json"]["generated_at"] = "2099-01-01T00:00:00Z"
    assert write_regression_model_selection_artifacts(
        output_directory=out, artifacts=changed
    ).idempotent


def test_writer_rejects_partial_set(tmp_path):
    out = tmp_path / "selection"; out.mkdir()
    (out / "candidate-results.json").write_text("{}")
    with pytest.raises(ArtifactConflictError):
        write_regression_model_selection_artifacts(output_directory=out, artifacts=_artifacts())


def test_runtime_loader_is_defensive():
    one = load_and_validate_model_selection_handoff(project_root=ROOT, handoff_path=SOURCE.relative_to(ROOT) / "model-selection-handoff.json")
    one["target_contract"]["unit"] = "kg"
    two = load_and_validate_model_selection_handoff(project_root=ROOT, handoff_path=SOURCE.relative_to(ROOT) / "model-selection-handoff.json")
    assert two["target_contract"]["unit"] == "MPa"
