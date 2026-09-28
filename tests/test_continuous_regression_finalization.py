import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from scripts.finalize_model import (
    FinalizationContractError,
    RegressionFrozenFinalizationContract,
    assemble_regression_final_training_data,
    compute_regression_model_state_fingerprint,
    describe_regression_fitted_pipeline,
    reconstruct_regression_selected_pipeline,
)
from scripts.select_models import compute_regression_metrics


FEATURES = ["a", "b"]


def contract(family="HistGradientBoostingRegressor", *, scale=False):
    fixed = {
        "Ridge": {"alpha": 1.0},
        "DecisionTreeRegressor": {"random_state": 42, "max_depth": None},
        "RandomForestRegressor": {"random_state": 42, "n_estimators": 5, "n_jobs": 1},
        "HistGradientBoostingRegressor": {"random_state": 42, "max_iter": 10},
    }[family]
    selected = {"model__alpha": 2.0} if family == "Ridge" else {}
    payload = {
        "selected_model_family": family,
        "selected_estimator_fixed_constructor_parameters": fixed,
        "selected_hyperparameters": selected,
        "feature_order": FEATURES,
        "preprocessing": {"type": "Pipeline", "categorical_features": [],
                          "numerical_features": FEATURES, "scale_numerical": scale,
                          "scaler": "StandardScaler" if scale else None},
    }
    return RegressionFrozenFinalizationContract(tuple(payload.items()))


@pytest.mark.parametrize("family", ["Ridge", "DecisionTreeRegressor",
    "RandomForestRegressor", "HistGradientBoostingRegressor"])
def test_exact_supported_family_reconstruction_is_unfitted(family):
    pipeline = reconstruct_regression_selected_pipeline(contract(family, scale=family == "Ridge"))
    assert pipeline.named_steps["model"].__class__.__name__ == family
    with pytest.raises(Exception):
        check_is_fitted(pipeline)
    assert (pipeline.named_steps["preprocess"].transformers[0][1].__class__.__name__ == "StandardScaler") == (family == "Ridge")


def test_selected_parameters_override_fixed_values():
    pipeline = reconstruct_regression_selected_pipeline(contract("Ridge", scale=True))
    assert pipeline.named_steps["model"].alpha == 2.0


@pytest.mark.parametrize("mutation", ["family", "fixed", "selected", "preprocess"])
def test_reconstruction_fails_closed(mutation):
    data = contract().as_dict()
    if mutation == "family": data["selected_model_family"] = "SVR"
    elif mutation == "fixed": data["selected_estimator_fixed_constructor_parameters"]["bogus"] = 1
    elif mutation == "selected": data["selected_hyperparameters"]["model__bogus"] = 1
    else: data["preprocessing"]["categorical_features"] = ["a"]
    bad = RegressionFrozenFinalizationContract(tuple(data.items()))
    with pytest.raises(FinalizationContractError):
        reconstruct_regression_selected_pipeline(bad)


def test_training_assembly_has_no_test_argument_and_preserves_order():
    assert "test" not in inspect.signature(assemble_regression_final_training_data).parameters
    train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "y": [5.0, 6.0]})
    validation = pd.DataFrame({"a": [7.0], "b": [8.0], "y": [9.0]})
    x, y = assemble_regression_final_training_data(train=train, validation=validation,
        feature_columns=FEATURES, target_column="y")
    assert x.to_dict("records") == [{"a": 1.0, "b": 3.0}, {"a": 2.0, "b": 4.0}, {"a": 7.0, "b": 8.0}]
    assert y.tolist() == [5.0, 6.0, 9.0]


@pytest.mark.parametrize("target", [[1.0, np.nan], [1.0, np.inf], ["x", "y"]])
def test_training_target_must_be_numeric_complete_finite(target):
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "y": target})
    with pytest.raises(FinalizationContractError):
        assemble_regression_final_training_data(train=frame, validation=frame,
            feature_columns=FEATURES, target_column="y")


def test_known_regression_metrics_and_diagnostics():
    metrics = compute_regression_metrics([1, 2, 4], [0, 2, 7])
    residuals = np.array([1.0, 0.0, -3.0]); absolute = np.abs(residuals)
    assert metrics["mae"] == pytest.approx(4 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(10 / 3))
    assert metrics["medae"] == 1.0
    assert metrics["residual_mean"] == pytest.approx(residuals.mean())
    assert metrics["residual_standard_deviation"] == pytest.approx(residuals.std(ddof=1))
    assert metrics["max_absolute_error"] == 3.0
    for q in (50, 90, 95): assert metrics[f"absolute_error_p{q}"] == pytest.approx(np.quantile(absolute, q/100))


def test_descriptor_and_semantic_fingerprint_survive_equivalent_fit():
    c = contract("Ridge", scale=True)
    frame = pd.DataFrame({"a": [1., 2., 3.], "b": [3., 2., 1.]})
    pipeline = reconstruct_regression_selected_pipeline(c).fit(frame, [1., 2., 4.])
    descriptor = describe_regression_fitted_pipeline(pipeline=pipeline, contract=c)
    assert descriptor["model_class"] == "Ridge"
    assert compute_regression_model_state_fingerprint(descriptor) == compute_regression_model_state_fingerprint(json.loads(json.dumps(descriptor)))


def test_official_notebook_is_continuous_only():
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/04_final_model_and_bundle.ipynb").read_text())
    code = "\n".join("".join(c.get("source", [])) for c in notebook["cells"] if c["cell_type"] == "code").lower()
    assert "run_regression_finalization" in code
    assert "argmax" not in code and "positive_class" not in code and "threshold" not in code


# ---------------------------------------------------------------------------
# End-to-end finalization over the synthetic continuous project
# ---------------------------------------------------------------------------

from tests._continuous_model_selection_fixtures import (  # noqa: E402
    SELECTION_HANDOFF,
    build_synthetic_preparation,
    synthetic_selection_artifacts,
)
from scripts.select_models import write_regression_model_selection_artifacts  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

FINAL_DIR = Path("artifacts/models/synthetic-regression")


def _finalization_project(root, *, declare_group_diagnostic=True):
    preparation = build_synthetic_preparation(root)
    artifacts = synthetic_selection_artifacts(preparation)
    handoff = artifacts["model-selection-handoff.json"]
    handoff["selected_estimator_fixed_constructor_parameters"] = {
        k: v for k, v in Ridge().get_params(deep=False).items() if v is not None
    }
    handoff["selected_preprocessing_contract"] = {
        "type": "Pipeline", "numerical_features": ["input_a", "input_b"], "categorical_features": [],
        "scale_numerical": False, "scaler": None, "fit_scope": "training_partition_or_training_fold_only"}
    handoff["random_seeds"] = {"cv": 42, "estimators": 42}
    handoff["selected_validation_evidence"] = {"mae": 0.1, "rmse": 0.12, "r2": 0.99, "medae": 0.1}
    if declare_group_diagnostic:
        handoff["evaluation_diagnostics"] = {"group_overlap_columns": ["input_a"], "used_for_selection": False}
    write_regression_model_selection_artifacts(output_directory=root / SELECTION_HANDOFF.parent, artifacts=artifacts)
    return root


def _run(root):
    from scripts.finalize_model import run_regression_finalization
    return run_regression_finalization(project_root=root, model_selection_handoff_path=SELECTION_HANDOFF,
                                       output_directory=FINAL_DIR)


def _load(root, name):
    return json.loads((root / FINAL_DIR / name).read_text())


def test_finalization_records_effective_parameters_runtime_and_test_state(tmp_path):
    from scripts.finalize_model import regression_runtime_versions
    root = _finalization_project(tmp_path)
    result = _run(root)
    assert result["status"] == "created" and result["test_evaluation_count"] == 1
    manifest, bundle = _load(root, "final-model-manifest.json"), _load(root, "inference-bundle.json")
    evidence = _load(root, "final-test-evidence.json")
    effective = manifest["selected_estimator_effective_parameters"]
    assert effective["alpha"] == 0.1 and manifest["selected_estimator_fixed_constructor_parameters"]["alpha"] == 1.0
    assert bundle["model_contract"]["effective_parameters"] == effective
    assert manifest["runtime_versions"] == bundle["runtime_compatibility"] == regression_runtime_versions()
    assert "numpy" in manifest["runtime_versions"]
    assert evidence["partition_state"] == {"sealed_until_final_fit": True, "evaluated": True, "evaluation_count": 1}
    assert evidence["frozen_partition_reference"] == manifest["frozen_test_reference"]
    assert "partition_reference" not in evidence and "test_reference" not in manifest


def test_group_overlap_diagnostic_is_descriptive_and_partitions_test_rows(tmp_path):
    root = _finalization_project(tmp_path)
    _run(root)
    evidence = _load(root, "final-test-evidence.json")
    diagnostic = evidence["group_overlap_diagnostic"]
    assert diagnostic["diagnostic_only"] is True and diagnostic["used_for_selection"] is False
    assert diagnostic["group_columns"] == ["input_a"]
    assert diagnostic["reference_partitions"] == ["train", "validation"]
    assert diagnostic["seen_group_row_count"] + diagnostic["unseen_group_row_count"] == evidence["row_count"]
    assert diagnostic["full_metrics"] == evidence["metrics"]


def test_group_overlap_diagnostic_is_absent_when_not_declared(tmp_path):
    root = _finalization_project(tmp_path, declare_group_diagnostic=False)
    _run(root)
    assert _load(root, "final-test-evidence.json")["group_overlap_diagnostic"] is None


def test_finalization_reuses_equivalent_set_without_second_test_access(tmp_path):
    root = _finalization_project(tmp_path)
    first = _run(root)
    second = _run(root)
    assert second["status"] == "reused_equivalent"
    assert second["test_parse_count"] == 0 and second["test_predict_count"] == 0
    assert second["metrics"] == first["metrics"]


@pytest.mark.parametrize("field", ["effective", "runtime"])
def test_complete_set_rejects_divergent_effective_parameters_or_runtime(tmp_path, field):
    from scripts.finalize_model import load_and_validate_inference_bundle
    root = _finalization_project(tmp_path)
    _run(root)
    path = root / FINAL_DIR / "final-model-manifest.json"
    manifest = json.loads(path.read_text())
    if field == "effective":
        manifest["selected_estimator_effective_parameters"]["alpha"] = 5.0
    else:
        manifest["runtime_versions"]["numpy"] = "0.0.0"
    path.write_text(json.dumps(manifest))
    with pytest.raises(FinalizationContractError):
        load_and_validate_inference_bundle(project_root=root, bundle_path=FINAL_DIR / "inference-bundle.json")


def test_selection_loader_rejects_invalid_evaluation_diagnostics(tmp_path):
    from scripts.select_models import ModelSelectionHandoffError, load_and_validate_model_selection_handoff
    preparation = build_synthetic_preparation(tmp_path)
    artifacts = synthetic_selection_artifacts(preparation)
    artifacts["model-selection-handoff.json"]["evaluation_diagnostics"] = {
        "group_overlap_columns": ["input_a", "input_b"], "used_for_selection": False}
    write_regression_model_selection_artifacts(output_directory=tmp_path / SELECTION_HANDOFF.parent, artifacts=artifacts)
    with pytest.raises(ModelSelectionHandoffError):
        load_and_validate_model_selection_handoff(project_root=tmp_path, handoff_path=SELECTION_HANDOFF)
