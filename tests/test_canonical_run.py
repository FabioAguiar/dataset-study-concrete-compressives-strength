"""Contracts for the versioned canonical reference-run manifest (no runtime artifacts needed)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.canonical_run import (
    MANIFEST_SCHEMA_VERSION,
    CanonicalRunError,
    compare_with_canonical,
    environment_contract_report,
    validate_manifest_structure,
    verify_against_committed_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "evidence" / "canonical-run.json"
SOURCE_CONTRACT = json.loads((ROOT / "contracts" / "source.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _mutated(manifest: dict, path: tuple[str, ...], value) -> dict:
    observed = copy.deepcopy(manifest)
    target = observed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return observed


def test_committed_manifest_is_internally_consistent(manifest: dict) -> None:
    validate_manifest_structure(manifest)
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["derivation"]["adds_no_decision"] is True
    assert manifest["derivation"]["row_level_data_persisted"] is False


def test_manifest_source_is_the_pinned_source(manifest: dict) -> None:
    pinned = SOURCE_CONTRACT["files"][0]
    dataset = manifest["dataset"]
    assert dataset["source_sha256"] == pinned["sha256"]
    assert dataset["source_size_bytes"] == pinned["size_bytes"]
    assert (dataset["source_row_count"], dataset["source_column_count"]) == (pinned["rows"], pinned["columns"])
    assert dataset["uci_dataset_id"] == SOURCE_CONTRACT["dataset_id"]


def test_manifest_encodes_the_frozen_regression_protocol(manifest: dict) -> None:
    assert manifest["target"]["problem_type"] == "continuous_regression"
    assert manifest["split"]["stratify_by"] is None
    assert manifest["split"]["stage_seeds"] == {"train_vs_temporary": 42, "validation_vs_test": 43}
    assert manifest["split"]["row_counts"] == {"train": 721, "validation": 154, "test": 155}
    assert manifest["cross_validation"] == {"strategy": "KFold", "n_splits": 5, "shuffle": True,
                                            "random_state": 42, "fit_partition": "train_only"}
    selection = manifest["model_selection"]
    assert {row["model_id"]: row["candidate_count"] for row in selection["family_search"]} == {
        "decision_tree": 12, "hist_gradient_boosting": 24, "random_forest": 12, "ridge": 4}
    assert selection["baseline"]["strategy"] == "median"
    assert selection["practical_tie_tolerance"] == 0.10
    assert selection["selected_model_id"] == "hist_gradient_boosting"
    assert selection["practical_tie"] is False
    assert manifest["final_model"]["fit_row_count"] == 875
    assert manifest["final_test"]["evaluation_count"] == 1


def test_mixture_diagnostic_is_descriptive_and_excludes_age(manifest: dict) -> None:
    overlap = manifest["mixture_overlap"]
    assert "Age" not in overlap["group_columns"] and overlap["varying_columns"] == ["Age"]
    assert overlap["proven_duplicate_identity"] is False
    assert (overlap["validation"]["seen_group_row_count"], overlap["validation"]["unseen_group_row_count"]) == (111, 43)
    assert (overlap["test"]["seen_group_row_count"], overlap["test"]["unseen_group_row_count"]) == (116, 39)


def test_manifest_agrees_with_the_committed_canonical_evidence(manifest: dict) -> None:
    """Re-derive the verification against the executed notebooks and README."""
    verification = verify_against_committed_evidence(ROOT, manifest)
    assert verification == manifest["reference_verification"]
    decisions = [c for c in verification["differences"] if c["delta"] is None]
    assert decisions == [], "every decision, identity and published value must match exactly"
    assert verification["max_absolute_numeric_difference"] < 1e-12
    assert all(c["quantity"].startswith("validation_ranking.ridge.") for c in verification["differences"])


def test_comparison_accepts_identical_run(manifest: dict) -> None:
    report = compare_with_canonical(manifest, copy.deepcopy(manifest))
    assert report["matches_canonical"] is True


@pytest.mark.parametrize(
    "path, value",
    [
        (("final_test", "metrics", "mae"), 3.0),
        (("validation_metrics", "rmse"), 5.0),
        (("model_selection", "selected_model_id"), "random_forest"),
        (("split", "membership_sha256", "test"), "0" * 64),
    ],
)
def test_comparison_detects_scientific_divergence(manifest: dict, path, value) -> None:
    report = compare_with_canonical(manifest, _mutated(manifest, path, value))
    assert report["matches_canonical"] is False


def test_structure_validation_rejects_protocol_violations(manifest: dict) -> None:
    with pytest.raises(CanonicalRunError, match="evaluated once"):
        validate_manifest_structure(_mutated(manifest, ("final_test", "evaluation_count"), 2))
    with pytest.raises(CanonicalRunError, match="class concept"):
        validate_manifest_structure(_mutated(manifest, ("target", "positive_class"), "high"))
    with pytest.raises(CanonicalRunError, match="descriptive only"):
        observed = copy.deepcopy(manifest)
        observed["mixture_overlap"]["test"]["used_for_selection"] = True
        validate_manifest_structure(observed)


def test_manifest_runtime_was_observed_in_the_locked_environment(manifest: dict) -> None:
    runtime = manifest["runtime"]
    assert runtime["python_version_file"] == ".python-version"
    assert runtime["lock_file"] == "pylock.toml"
    report = environment_contract_report(ROOT, runtime)
    assert report["matches_locked_environment"] is True, report
