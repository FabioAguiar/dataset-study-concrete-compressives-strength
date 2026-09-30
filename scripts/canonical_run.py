"""Canonical reference-run manifest for the Concrete Compressive Strength study.

The manifest is a small, versioned, machine-readable projection of the
scientific reference execution. It is derived exclusively from persisted,
hash-validated runtime artifacts (exploration, preparation, model-selection,
and final-model artifacts) and never stores rows, predictions, residuals, or
model binaries. It adds no decision of its own: every value is read from an
artifact the notebooks already wrote.

The environment contract is not duplicated here: the exact interpreter is read
from ``.python-version`` and exact package versions from ``pylock.toml``. The
runtime recorded in the manifest is observed evidence of one execution.

Usage::

    python -m scripts.canonical_run build    # write evidence/canonical-run.json
    python -m scripts.canonical_run verify   # compare a fresh run with the manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = "canonical-run.v1"
MANIFEST_ARTIFACT_TYPE = "canonical_reference_run"
DEFAULT_MANIFEST_PATH = Path("evidence/canonical-run.json")
PYTHON_VERSION_FILE = Path(".python-version")
LOCK_FILE = Path("pylock.toml")
SOURCE_CONTRACT_FILE = Path("contracts/source.json")
DATASET_SLUG = "concrete-compressive-strength"
METRIC_TOLERANCE = 1e-9
REGRESSION_METRICS = ("mae", "rmse", "r2", "medae")
DIAGNOSTIC_METRICS = (
    "residual_mean",
    "residual_standard_deviation",
    "max_absolute_error",
    "absolute_error_p50",
    "absolute_error_p90",
    "absolute_error_p95",
)

_ROOT = f"artifacts/{{}}/{DATASET_SLUG}"
ARTIFACT_PATHS = {
    "exploration_handoff": Path(_ROOT.format("exploration")) / "exploration-handoff.json",
    "preparation_handoff": Path(_ROOT.format("preparation")) / "preparation-handoff.json",
    "feature_manifest": Path(_ROOT.format("preparation")) / "feature-manifest.json",
    "split_manifest": Path(_ROOT.format("preparation")) / "split-manifest.json",
    "model_selection_manifest": Path(_ROOT.format("model-selection")) / "model-selection-manifest.json",
    "model_selection_handoff": Path(_ROOT.format("model-selection")) / "model-selection-handoff.json",
    "candidate_results": Path(_ROOT.format("model-selection")) / "candidate-results.json",
    "selection_analysis": Path(_ROOT.format("model-selection")) / "selection-analysis.json",
    "final_model_manifest": Path(_ROOT.format("models")) / "final-model-manifest.json",
    "final_test_evidence": Path(_ROOT.format("models")) / "final-test-evidence.json",
    "final_model_handoff": Path(_ROOT.format("models")) / "final-model-handoff.json",
    "inference_bundle": Path(_ROOT.format("models")) / "inference-bundle.json",
}
RUNTIME_PACKAGES = ("numpy", "pandas", "scikit-learn", "joblib", "scipy", "matplotlib", "ucimlrepo")


class CanonicalRunError(RuntimeError):
    """Raised when artifacts cannot produce or satisfy the canonical manifest."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(root: Path, relative: Path) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise CanonicalRunError(f"Required artifact is missing: {relative.as_posix()}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifacts(project_root: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(project_root).resolve()
    return {name: _load_json(root, path) for name, path in ARTIFACT_PATHS.items()}


def runtime_environment() -> dict[str, Any]:
    packages = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "packages": packages,
    }


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def canonical_python_version(project_root: str | Path) -> str:
    """Return the exact canonical CPython version declared in .python-version."""
    path = Path(project_root) / PYTHON_VERSION_FILE
    if not path.is_file():
        raise CanonicalRunError(f"Missing {PYTHON_VERSION_FILE.as_posix()}.")
    version = path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise CanonicalRunError(f"{PYTHON_VERSION_FILE.as_posix()} must contain X.Y.Z, got {version!r}.")
    return version


def locked_package_versions(project_root: str | Path) -> dict[str, str]:
    """Return {normalized name: version} from the canonical PEP 751 lock."""
    import tomllib

    path = Path(project_root) / LOCK_FILE
    if not path.is_file():
        raise CanonicalRunError(f"Missing {LOCK_FILE.as_posix()}.")
    lock = tomllib.loads(path.read_text(encoding="utf-8"))
    if lock.get("lock-version") != "1.0":
        raise CanonicalRunError("pylock.toml must use lock-version 1.0 (PEP 751).")
    return {_normalize_name(item["name"]): item["version"] for item in lock["packages"]}


def environment_contract_report(project_root: str | Path, runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Compare an observed runtime with .python-version and pylock.toml."""
    expected_python = canonical_python_version(project_root)
    locked = locked_package_versions(project_root)
    package_mismatches = {
        name: {"locked": locked.get(_normalize_name(name)), "observed": version}
        for name, version in runtime["packages"].items()
        if locked.get(_normalize_name(name)) != version
    }
    return {
        "python_version_file": PYTHON_VERSION_FILE.as_posix(),
        "lock_file": LOCK_FILE.as_posix(),
        "expected_python": expected_python,
        "observed_python": runtime["python"],
        "python_matches": runtime["python"] == expected_python,
        "package_mismatches": package_mismatches,
        "matches_locked_environment": runtime["python"] == expected_python and not package_mismatches,
    }


def membership_sha256(tokens: Sequence[str]) -> str:
    """SHA-256 of a partition's technical membership tokens in source order."""
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()


def _float(value: Any) -> Any:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value


def _metrics(metrics: Mapping[str, Any], keys: Sequence[str] = REGRESSION_METRICS + DIAGNOSTIC_METRICS) -> dict[str, Any]:
    return {key: _float(metrics[key]) for key in keys if key in metrics}


def _subset(subset: Mapping[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {"status": subset["status"], "row_count": int(subset["row_count"])}
    if subset["status"] == "computed":
        projected["metrics"] = _metrics(subset["metrics"])
    return projected


def _group_overlap(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "group_columns": list(diagnostic["group_columns"]),
        "reference_partitions": list(diagnostic["reference_partitions"]),
        "evaluated_partition": diagnostic["evaluated_partition"],
        "evaluated_row_count": int(diagnostic["evaluated_row_count"]),
        "seen_group_row_count": int(diagnostic["seen_group_row_count"]),
        "unseen_group_row_count": int(diagnostic["unseen_group_row_count"]),
        "full": {"row_count": int(diagnostic["evaluated_row_count"]),
                 "metrics": _metrics(diagnostic["full_metrics"])},
        "seen_groups": _subset(diagnostic["seen_groups"]),
        "unseen_groups": _subset(diagnostic["unseen_groups"]),
        "diagnostic_only": bool(diagnostic["diagnostic_only"]),
        "used_for_selection": bool(diagnostic["used_for_selection"]),
    }


def build_canonical_manifest(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    source_sha256: str,
    source_contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the canonical manifest from already-validated artifacts."""
    exploration = artifacts["exploration_handoff"]
    feature = artifacts["feature_manifest"]
    split = artifacts["split_manifest"]
    selection_manifest = artifacts["model_selection_manifest"]
    selection = artifacts["model_selection_handoff"]
    candidates = artifacts["candidate_results"]
    analysis = artifacts["selection_analysis"]
    final_manifest = artifacts["final_model_manifest"]
    test_evidence = artifacts["final_test_evidence"]
    final_handoff = artifacts["final_model_handoff"]
    bundle = artifacts["inference_bundle"]

    pinned = source_contract["files"][0]
    if source_sha256 != pinned["sha256"]:
        raise CanonicalRunError("Source bytes differ from the pinned source contract.")
    if source_sha256 != exploration["source"]["sha256"]:
        raise CanonicalRunError("Source bytes differ from the exploration handoff.")
    if bundle["model_artifact_sha256"] != final_handoff["model_artifact_reference"]["sha256"]:
        raise CanonicalRunError("Bundle and final handoff model hashes differ.")
    if bundle["model_contract"]["selected_hyperparameters"] != selection["selected_hyperparameters"]:
        raise CanonicalRunError("Bundle hyperparameters differ from the model-selection freeze.")
    if list(bundle["feature_order"]) != list(selection["selected_feature_columns"]):
        raise CanonicalRunError("Bundle feature order differs from the model-selection freeze.")
    if test_evidence["test_partition_evaluation_count"] != 1 or test_evidence["test_prediction_call_count"] != 1:
        raise CanonicalRunError("Final test must be predicted and evaluated exactly once.")

    membership = {name: membership_sha256(tokens) for name, tokens in split["membership"].items()}
    families = sorted(
        (
            {
                "model_id": row["model_id"],
                "family": row["family"],
                "search_strategy": row["search_strategy"],
                "search_space": row["search_space"],
                "candidate_count": int(row["cv_aggregates"]["candidate_count"]),
                "best_parameters": row["selected_hyperparameters"],
                "cv_metrics": {f"cv_{m}_{s}": _float(row["cv_aggregates"][f"cv_{m}_{s}"])
                               for m in REGRESSION_METRICS for s in ("mean", "std")},
                "validation_metrics": _metrics(row["validation_aggregates"]),
            }
            for row in candidates["family_searches"]
        ),
        key=lambda row: row["model_id"],
    )
    ranking = [
        {
            "model_id": row["model_id"],
            "family": row["family"],
            "validation_mae": _float(row["validation_mae"]),
            "validation_rmse": _float(row["validation_rmse"]),
            "validation_medae": _float(row["validation_medae"]),
            "validation_r2": _float(row["validation_r2"]),
            "cv_mae_std": _float(row["cv_mae_std"]),
            "eligible": bool(row["eligible"]),
        }
        for row in candidates["selection"]["ranking"]
    ]
    baseline = candidates["baseline"]
    dependencies = exploration["leakage_and_dependencies"]["grouped_observation_dependencies"]
    if len(dependencies) != 1:
        raise CanonicalRunError("Exactly one grouped observation dependency (the mixture) is expected.")
    (grouped,) = dependencies
    fixed_constructor = selection["selected_estimator_fixed_constructor_parameters"]

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "derivation": {
            "method": "projection of persisted, hash-validated runtime artifacts (scripts/canonical_run.py)",
            "adds_no_decision": True,
            "row_level_data_persisted": False,
        },
        "dataset": {
            "slug": DATASET_SLUG,
            "name": source_contract["dataset_name"],
            "repository": source_contract["repository"],
            "uci_dataset_id": int(source_contract["dataset_id"]),
            "dataset_doi": source_contract["dataset_doi"],
            "source_file": pinned["filename"],
            "source_sha256": source_sha256,
            "source_size_bytes": int(pinned["size_bytes"]),
            "source_row_count": int(exploration["source"]["row_count"]),
            "source_column_count": int(exploration["source"]["column_count"]),
        },
        "target": {
            "column": feature["target_column"],
            "problem_type": feature["problem_type"],
            "semantics": feature["target_contract"]["semantics"],
            "unit": feature["target_contract"]["unit"],
            "classes": None,
            "positive_class": None,
            "decision_threshold": None,
        },
        "features": {
            "ordered_columns": list(feature["feature_columns"]),
            "count": len(feature["feature_columns"]),
            "identifiers": list(feature["identifier_columns"]),
        },
        "split": {
            "split_id": split.get("split_id"),
            "split_method": split["split_method"],
            "fractions": {"train": split["train_fraction"], "validation": split["validation_fraction"],
                          "test": split["test_fraction"]},
            "stratify_by": split["stratify_by"],
            "shuffle": split["shuffle"],
            "stage_seeds": dict(split["stage_seeds"]),
            "membership_kind": split["membership_kind"],
            "membership_digest": "sha256 of the partition's membership tokens joined by newlines in source order",
            "row_counts": {name: int(count) for name, count in split["row_counts"].items()},
            "partition_sha256": dict(split["partition_sha256"]),
            "membership_sha256": membership,
        },
        "cross_validation": {key: selection["cv_contract"][key]
                             for key in ("strategy", "n_splits", "shuffle", "random_state", "fit_partition")},
        "model_selection": {
            "primary_metric": selection["primary_metric"],
            "primary_metric_direction": selection["primary_metric_direction"],
            "secondary_metrics": list(selection["secondary_metrics"]),
            "practical_tie_tolerance": _float(selection_manifest["model_selection_contract"]["practical_tie_tolerance_mpa"]),
            "validation_in_search": selection_manifest["search_contract"]["validation_in_search"],
            "test_in_search": selection_manifest["search_contract"]["test_in_search"],
            "baseline": {
                "model_id": baseline["model_id"],
                "family": baseline["family"],
                "strategy": baseline["strategy"],
                "validation_metrics": _metrics(baseline["validation_aggregates"]),
            },
            "family_search": families,
            "validation_ranking": ranking,
            "practical_tie": bool(candidates["selection"]["practical_tie"]),
            "finalists": list(candidates["selection"]["finalists"]),
            "criteria_applied": list(candidates["selection"]["criteria_applied"]),
            "selected_model_id": selection["selected_model_id"],
            "selected_model_family": selection["selected_model_family"],
            "selected_feature_policy": selection["selected_feature_policy"],
            "selected_hyperparameters": dict(selection["selected_hyperparameters"]),
            "selected_estimator_fixed_constructor_parameters": dict(fixed_constructor),
            "selected_preprocessing": dict(selection["selected_preprocessing_contract"]),
        },
        "selected_cv_metrics": {f"cv_{m}_{s}": _float(selection["selected_cv_evidence"][f"cv_{m}_{s}"])
                                for m in REGRESSION_METRICS for s in ("mean", "std")},
        "validation_metrics": _metrics(selection["selected_validation_evidence"]),
        "mixture_overlap": {
            "group_columns": list(grouped["group_columns"]),
            "varying_columns": list(grouped["varying_columns"]),
            "validation": _group_overlap(analysis["mixture_overlap_sensitivity"]),
            "test": _group_overlap(test_evidence["group_overlap_diagnostic"]),
            "proven_duplicate_identity": False,
        },
        "final_model": {
            "training_partitions": list(final_manifest["training_partitions"]),
            "fit_row_count": int(final_manifest["training_row_count"]),
            "final_fit_count": int(final_manifest["final_fit_count"]),
            "selected_estimator_effective_parameters": final_manifest["selected_estimator_effective_parameters"],
            "artifact_path": bundle["model_artifact_path"],
            "artifact_sha256": bundle["model_artifact_sha256"],
            "model_state_fingerprint": bundle["model_state_fingerprint"],
            "frozen_finalization_contract_fingerprint": final_manifest["frozen_finalization_contract_fingerprint"],
        },
        "final_test": {
            "row_count": int(test_evidence["row_count"]),
            "evaluation_count": int(test_evidence["test_partition_evaluation_count"]),
            "prediction_call_count": int(test_evidence["test_prediction_call_count"]),
            "metrics": _metrics(test_evidence["metrics"]),
            "validation_to_test_deltas": {k: _float(v) for k, v in test_evidence["validation_to_test_deltas"].items()},
        },
        "protocol": {
            "test_opened_after_freeze_and_final_fit": bool(test_evidence["test_loaded_only_after_final_fit"]),
            "test_used_for_adjustment": not bool(test_evidence["no_post_test_adjustment"]),
            "diagnostics_used_for_selection": bool(selection["analysis_conclusions"]["sensitivities_used_for_selection"]),
            "row_level_predictions_persisted": False,
        },
        "runtime": {
            **dict(runtime),
            "recorded_by_final_bundle": dict(bundle["runtime_compatibility"]),
            "python_version_file": PYTHON_VERSION_FILE.as_posix(),
            "lock_file": LOCK_FILE.as_posix(),
        },
    }


def build_from_project(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    artifacts = load_artifacts(root)
    source_contract = json.loads((root / SOURCE_CONTRACT_FILE).read_text(encoding="utf-8"))
    source = root / artifacts["exploration_handoff"]["source"]["path"]
    if not source.is_file():
        raise CanonicalRunError("Source dataset file is missing.")
    manifest = build_canonical_manifest(artifacts, source_sha256=_sha256_file(source),
                                        source_contract=source_contract, runtime=runtime_environment())
    manifest["reference_verification"] = verify_against_committed_evidence(root, manifest)
    return manifest


def validate_manifest_structure(manifest: Mapping[str, Any]) -> None:
    """Check internal consistency of a canonical manifest without artifacts."""
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise CanonicalRunError("Unexpected canonical manifest schema version.")
    if manifest.get("artifact_type") != MANIFEST_ARTIFACT_TYPE:
        raise CanonicalRunError("Unexpected canonical manifest artifact type.")
    target = manifest["target"]
    if target["problem_type"] != "continuous_regression" or any(
            target[key] is not None for key in ("classes", "positive_class", "decision_threshold")):
        raise CanonicalRunError("A continuous-regression manifest carries no class concept.")
    split = manifest["split"]
    if split["stratify_by"] is not None:
        raise CanonicalRunError("The regression split is not stratified.")
    rows = split["row_counts"]
    if sum(rows.values()) != manifest["dataset"]["source_row_count"]:
        raise CanonicalRunError("Partition row counts do not cover the source.")
    selection = manifest["model_selection"]
    if selection["validation_in_search"] or selection["test_in_search"]:
        raise CanonicalRunError("Held-out partitions must not enter the search.")
    protocol = manifest["protocol"]
    if protocol["test_used_for_adjustment"] or protocol["diagnostics_used_for_selection"]:
        raise CanonicalRunError("Test and diagnostics must not feed any decision.")
    test = manifest["final_test"]
    if test["evaluation_count"] != 1 or test["prediction_call_count"] != 1:
        raise CanonicalRunError("Final test must be evaluated once.")
    if test["row_count"] != rows["test"]:
        raise CanonicalRunError("Final test rows differ from the test partition.")
    final = manifest["final_model"]
    if final["training_partitions"] != ["train", "validation"] or final["fit_row_count"] != rows["train"] + rows["validation"]:
        raise CanonicalRunError("Final fit must use exactly train + validation.")
    if final["final_fit_count"] != 1:
        raise CanonicalRunError("The final model must be fitted once.")
    families = {row["model_id"]: row for row in selection["family_search"]}
    selected = families[selection["selected_model_id"]]
    if selected["best_parameters"] != selection["selected_hyperparameters"]:
        raise CanonicalRunError("Selected hyperparameters differ from the family search winner.")
    overlap = manifest["mixture_overlap"]
    for name, reference in (("validation", ["train"]), ("test", ["train", "validation"])):
        diagnostic = overlap[name]
        if diagnostic["reference_partitions"] != reference or diagnostic["evaluated_row_count"] != rows[name]:
            raise CanonicalRunError(f"The {name} mixture diagnostic does not describe the {name} partition.")
        if diagnostic["seen_group_row_count"] + diagnostic["unseen_group_row_count"] != rows[name]:
            raise CanonicalRunError(f"The {name} mixture subsets do not cover the partition.")
        if diagnostic["used_for_selection"] or not diagnostic["diagnostic_only"]:
            raise CanonicalRunError("Mixture diagnostics are descriptive only.")
    features = manifest["features"]["ordered_columns"]
    if manifest["features"]["count"] != len(features) or target["column"] in features:
        raise CanonicalRunError("Feature contract is inconsistent.")
    if not set(overlap["group_columns"]) < set(features):
        raise CanonicalRunError("Mixture columns must be a strict subset of the features.")


SELECTION_NOTEBOOK = Path("notebooks/03_model_selection_and_evaluation.ipynb")
README_FILE = Path("README.md")


def _executed_selection_output(project_root: Path) -> dict[str, Any]:
    """The frozen SELECTION value rendered by the committed, executed Notebook 03."""
    import ast

    notebook = json.loads((project_root / SELECTION_NOTEBOOK).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and "".join(cell["source"]).strip() == "SELECTION":
            text = "".join(cell["outputs"][0]["data"]["text/plain"])
            return ast.literal_eval(text.replace("np.True_", "True").replace("np.False_", "False"))
    raise CanonicalRunError("The executed Notebook 03 has no SELECTION output.")


def verify_against_committed_evidence(project_root: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Check a manifest against the canonical evidence already versioned by the study.

    The executed notebooks and the README are the committed record of the
    canonical run. Every value they publish is compared with the manifest:
    validation candidates at full precision (Notebook 03), and the model
    SHA-256, final-test metrics, diagnostics and mixture subsets at the
    README's published precision. Every difference is listed, never hidden.
    """
    root = Path(project_root)
    readme = (root / README_FILE).read_text(encoding="utf-8")
    checks: list[dict[str, Any]] = []

    def check(quantity: str, expected: Any, observed: Any, *, published_decimals: int | None = None) -> None:
        if published_decimals is None:
            numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (expected, observed))
            delta = float(observed) - float(expected) if numeric else None
            equal = observed == expected
        else:
            delta = None
            equal = f"{float(observed):.{published_decimals}f}" in readme
        checks.append({"quantity": quantity, "reference": expected, "manifest": observed,
                       "delta": delta, "equal": bool(equal)})

    model_sha = manifest["final_model"]["artifact_sha256"]
    check("final_model.artifact_sha256", model_sha if model_sha in readme else "absent_from_README", model_sha)
    executed = _executed_selection_output(root)
    by_id = {row["model_id"]: row for row in manifest["model_selection"]["validation_ranking"]}
    for row in executed["ranking"]:
        for field in ("validation_mae", "validation_rmse", "validation_medae", "validation_r2", "cv_mae_std"):
            check(f"validation_ranking.{row['model_id']}.{field}", float(row[field]), by_id[row["model_id"]][field])
        check(f"validation_ranking.{row['model_id']}.eligible", bool(row["eligible"]), by_id[row["model_id"]]["eligible"])
    for key in ("selected_model_id", "practical_tie", "finalists", "criteria_applied"):
        check(f"model_selection.{key}", executed[key], manifest["model_selection"][key])
    for metric, value in manifest["final_test"]["metrics"].items():
        check(f"final_test.metrics.{metric}", "README (4 decimals)", value, published_decimals=4)
    for partition in ("validation", "test"):
        for subset in ("seen_groups", "unseen_groups"):
            diagnostic = manifest["mixture_overlap"][partition][subset]
            for metric in REGRESSION_METRICS:
                check(f"mixture_overlap.{partition}.{subset}.{metric}", "README (4 decimals)",
                      diagnostic["metrics"][metric], published_decimals=4)
    differing = [c for c in checks if not c["equal"]]
    return {
        "reference": "committed executed notebooks and README of the canonical run",
        "checked": len(checks),
        "equal": len(checks) - len(differing),
        "differences": differing,
        "max_absolute_numeric_difference": max((abs(c["delta"]) for c in differing if c["delta"] is not None),
                                               default=0.0),
    }


_EXACT_PATHS = (
    ("dataset",),
    ("target",),
    ("features",),
    ("split",),
    ("cross_validation",),
    ("model_selection", "selected_model_id"),
    ("model_selection", "selected_model_family"),
    ("model_selection", "selected_hyperparameters"),
    ("model_selection", "selected_estimator_fixed_constructor_parameters"),
    ("model_selection", "practical_tie"),
    ("model_selection", "finalists"),
    ("final_model", "fit_row_count"),
    ("final_test", "row_count"),
)
_NUMERIC_PATHS = (
    ("selected_cv_metrics",),
    ("validation_metrics",),
    ("final_test", "metrics"),
)
_RUNTIME_DEPENDENT_PATHS = (
    ("final_model", "model_state_fingerprint"),
    ("final_model", "artifact_sha256"),
)


def _get(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for key in path:
        value = value[key]
    return value


def _numeric_differences(expected: Mapping[str, Any], observed: Mapping[str, Any],
                         tolerance: float) -> dict[str, tuple[Any, Any]]:
    differences = {}
    for key in sorted(set(expected) | set(observed)):
        left, right = expected.get(key), observed.get(key)
        if left is None or right is None or not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance):
            differences[key] = (left, right)
    return differences


def compare_with_canonical(canonical: Mapping[str, Any], observed: Mapping[str, Any], *,
                           tolerance: float = METRIC_TOLERANCE) -> dict[str, Any]:
    """Compare a freshly built manifest with the canonical reference."""
    mismatches: dict[str, Any] = {}
    for path in _EXACT_PATHS:
        if _get(canonical, path) != _get(observed, path):
            mismatches[".".join(path)] = {"expected": _get(canonical, path), "observed": _get(observed, path)}
    for path in _NUMERIC_PATHS:
        for key, (left, right) in _numeric_differences(_get(canonical, path), _get(observed, path), tolerance).items():
            mismatches[".".join((*path, key))] = {"expected": left, "observed": right}
    same_runtime = (canonical["runtime"]["python"] == observed["runtime"]["python"]
                    and canonical["runtime"]["packages"] == observed["runtime"]["packages"])
    runtime_dependent = {}
    for path in _RUNTIME_DEPENDENT_PATHS:
        equal = _get(canonical, path) == _get(observed, path)
        runtime_dependent[".".join(path)] = equal
        if same_runtime and not equal:
            mismatches[".".join(path)] = {"expected": _get(canonical, path), "observed": _get(observed, path)}
    return {
        "matches_canonical": not mismatches,
        "same_runtime_as_reference": same_runtime,
        "metric_tolerance": tolerance,
        "mismatches": mismatches,
        "runtime_dependent_equalities": runtime_dependent,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH.as_posix())
    parser.add_argument("--tolerance", type=float, default=METRIC_TOLERANCE)
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve()
    manifest_path = root / args.manifest
    observed = build_from_project(root)
    validate_manifest_structure(observed)
    environment = environment_contract_report(root, observed["runtime"])
    if args.command == "build":
        if not environment["matches_locked_environment"]:
            print(json.dumps(environment, indent=2, sort_keys=True))
            raise CanonicalRunError("A canonical manifest can only be built in the environment declared "
                                    "by .python-version and pylock.toml.")
        _write_json(manifest_path, observed)
        print(f"Canonical manifest written: {args.manifest}")
        return 0

    canonical = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest_structure(canonical)
    report = compare_with_canonical(canonical, observed, tolerance=args.tolerance)
    report["environment_contract"] = environment
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["matches_canonical"] else 1


if __name__ == "__main__":
    sys.exit(main())
