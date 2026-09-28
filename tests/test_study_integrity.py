"""Regression gates for the executed study: notebooks, runtime, artifacts, figures, README.

Notebook, figure, and environment-contract checks always run because notebooks,
``docs/images``, ``contracts/``, and the lock are versioned. Checks over runtime
artifacts (``artifacts/``, ``data/``) are skipped in a fresh clone where the
study has not been executed yet.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import nbformat
import pytest

from scripts.finalize_model import (
    load_and_validate_final_model_handoff,
    load_and_validate_inference_bundle,
)
from scripts.runtime_contract import (
    expected_runtime_versions,
    observed_runtime_mismatches,
    read_lock_pins,
    read_python_version,
)
from scripts.smoke_predict import (
    validate_bundle_handoff_alignment,
    validate_model_artifact_before_load,
)


ROOT = Path(__file__).resolve().parents[1]
SLUG = "concrete-compressive-strength"
NOTEBOOKS = (
    "notebooks/01_data_understanding_and_exploration.ipynb",
    "notebooks/02_data_preparation.ipynb",
    "notebooks/03_model_selection_and_evaluation.ipynb",
    "notebooks/04_final_model_and_bundle.ipynb",
    "notebooks/05_inference_demo.ipynb",
)
EXPLORATION = f"artifacts/exploration/{SLUG}/exploration-handoff.json"
PREPARATION = f"artifacts/preparation/{SLUG}"
MODEL_SELECTION = f"artifacts/model-selection/{SLUG}"
MODELS = f"artifacts/models/{SLUG}"
HANDOFF = f"{MODELS}/final-model-handoff.json"
BUNDLE = f"{MODELS}/inference-bundle.json"
RAW_SOURCE = f"data/raw/{SLUG}/dataset.csv"
PERSONAL_PATH = re.compile(r"(/home/[^/\s\"']+|/Users/[^/\s\"']+|[A-Za-z]:\\\\Users\\\\)")
README_EXAMPLES = (
    "illustrative_mix_early_age",
    "illustrative_mix_standard",
    "illustrative_mix_slag_fly_ash",
    "illustrative_mix_high_cement",
)

requires_artifacts = pytest.mark.skipif(
    not (ROOT / HANDOFF).is_file(),
    reason="Study artifacts are runtime outputs; execute notebooks 01-05 first.",
)


def _json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def _notebook(relative: str):
    return nbformat.read(ROOT / relative, as_version=4)


def _code_cells(relative: str):
    return [cell for cell in _notebook(relative).cells if cell.cell_type == "code"]


def _output_text(cell) -> str:
    parts = []
    for output in cell.get("outputs", []):
        if output.output_type == "stream":
            parts.append(output.get("text", ""))
        else:
            data = output.get("data", {})
            parts.append(str(data.get("text/plain", "")))
            # Styled tables keep their values only in the HTML representation.
            parts.append(str(data.get("text/html", "")))
    return "\n".join(parts)


def _code(relative: str) -> str:
    return "\n".join(cell.source for cell in _code_cells(relative))


def _outputs(relative: str) -> str:
    return "\n".join(_output_text(cell) for cell in _code_cells(relative))


# ---------------------------------------------------------------------------
# Notebooks: versioned executed, from one clean kernel each
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relative", NOTEBOOKS)
def test_notebook_was_executed_linearly_in_one_clean_kernel(relative: str) -> None:
    cells = _code_cells(relative)
    assert cells, relative
    assert all(cell.source.strip() for cell in cells), "empty code cell"
    counts = [cell.execution_count for cell in cells]
    assert counts == list(range(1, len(cells) + 1)), counts


@pytest.mark.parametrize("relative", NOTEBOOKS)
def test_notebook_outputs_have_no_errors_warnings_or_personal_paths(relative: str) -> None:
    for index, cell in enumerate(_code_cells(relative)):
        for output in cell.get("outputs", []):
            assert output.output_type != "error", (relative, index)
            if output.output_type == "stream":
                assert output.name != "stderr", (relative, index, output.text[:200])
        assert not PERSONAL_PATH.search(_output_text(cell)), (relative, index)


@pytest.mark.parametrize("relative", NOTEBOOKS)
def test_notebook_kernel_matches_the_canonical_python(relative: str) -> None:
    version = _notebook(relative).metadata.get("language_info", {}).get("version")
    assert version == read_python_version(ROOT)


def test_evidence_producing_notebooks_require_the_canonical_runtime() -> None:
    for relative in NOTEBOOKS[:4]:
        assert "require_canonical_runtime(" in _code(relative), relative


def test_source_acquisition_is_verified_against_the_versioned_contract() -> None:
    for relative in NOTEBOOKS[:2]:
        code = _code(relative)
        assert "load_source_contract(" in code, relative
        assert "verify_acquisition_against_source_contract(" in code, relative


def test_mixture_dependency_flows_from_exploration_to_final_evaluation() -> None:
    assert "analyze_grouped_observation_dependency(" in _code(NOTEBOOKS[0])
    assert "grouped_dependency_report=grouped_dependency_report" in _code(NOTEBOOKS[0])
    nb03 = _code(NOTEBOOKS[2])
    assert "analyze_regression_group_overlap_sensitivity(" in nb03
    assert "'evaluation_diagnostics'" in nb03 and "'used_for_selection':False" in nb03
    assert "group_overlap_diagnostic" in _code(NOTEBOOKS[3])


# ---------------------------------------------------------------------------
# Environment contract
# ---------------------------------------------------------------------------


def test_environment_contract_files_exist_at_the_root() -> None:
    for name in (".python-version", "pyproject.toml", "pylock.toml", "contracts/source.json"):
        assert (ROOT / name).is_file(), name
    for legacy in ("requirements.txt", "requirements", "environment.yml", "reproducibility"):
        assert not (ROOT / legacy).exists(), f"competing dependency source: {legacy}"


def test_pylock_is_a_generated_pep751_lock_covering_every_declared_dependency() -> None:
    import tomllib

    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet

    lock_text = (ROOT / "pylock.toml").read_text(encoding="utf-8")
    assert lock_text.startswith("# This file was autogenerated"), "pylock.toml must be tool-generated"
    lock = tomllib.loads(lock_text)
    assert lock["lock-version"] == "1.0"

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    python_version = read_python_version(ROOT)
    assert python_version in SpecifierSet(project["project"]["requires-python"])
    assert python_version in SpecifierSet(lock["requires-python"])

    groups = project["dependency-groups"]

    def expand(group: str) -> list[str]:
        items: list[str] = []
        for item in groups[group]:
            items.extend(expand(item["include-group"]) if isinstance(item, dict) else [item])
        return items

    declared = [*project["project"]["dependencies"], *expand("study")]
    pins = read_lock_pins(ROOT)
    for text in declared:
        requirement = Requirement(text)
        name = re.sub(r"[-_.]+", "-", requirement.name).lower()
        assert name in pins, f"{requirement.name} is declared but not locked"
        assert pins[name] in requirement.specifier, (text, pins[name])


def test_lock_pins_every_runtime_component() -> None:
    pins = read_lock_pins(ROOT)
    for distribution in ("pandas", "scikit-learn", "joblib", "numpy", "scipy", "ucimlrepo"):
        assert distribution in pins, distribution
    assert re.fullmatch(r"\d+\.\d+\.\d+", read_python_version(ROOT))


def test_no_script_or_test_hardcodes_the_canonical_python_version() -> None:
    version = read_python_version(ROOT)
    for path in (*sorted((ROOT / "scripts").glob("*.py")), *sorted((ROOT / "tests").glob("*.py"))):
        assert f'"{version}"' not in path.read_text(encoding="utf-8"), path.name


def test_readme_documents_the_locked_runtime() -> None:
    readme = _readme()
    for component, version in expected_runtime_versions(ROOT).items():
        assert f"| {version} |" in readme, (component, version)
    assert "python -m pip install -r pylock.toml" in readme
    assert "python -m scripts.runtime_contract" in readme


# ---------------------------------------------------------------------------
# Source, artifact chain, and recorded runtime
# ---------------------------------------------------------------------------


@requires_artifacts
def test_raw_source_and_exploration_handoff_match_the_source_contract() -> None:
    (pinned,) = _json("contracts/source.json")["files"]
    assert _sha256(ROOT / RAW_SOURCE) == pinned["sha256"]
    assert _json(EXPLORATION)["source"]["sha256"] == pinned["sha256"]
    assert pinned["sha256"] in _readme()


@requires_artifacts
def test_bundle_and_manifests_record_the_canonical_runtime() -> None:
    expected = expected_runtime_versions(ROOT)
    assert _json(BUNDLE)["runtime_compatibility"] == expected
    assert _json(f"{MODELS}/final-model-manifest.json")["runtime_versions"] == expected
    recorded = _json(f"{MODEL_SELECTION}/model-selection-manifest.json")["runtime_versions"]
    for component in ("python", "pandas", "scikit_learn", "numpy"):
        assert recorded[component] == expected[component], component


@requires_artifacts
def test_final_artifacts_form_one_consistent_chain() -> None:
    handoff = load_and_validate_final_model_handoff(project_root=ROOT, handoff_path=HANDOFF)
    bundle = load_and_validate_inference_bundle(project_root=ROOT, bundle_path=BUNDLE)
    manifest = _json(f"{MODELS}/final-model-manifest.json")
    validate_bundle_handoff_alignment(handoff, bundle, manifest=manifest)
    model_path = validate_model_artifact_before_load(
        project_root=ROOT, bundle=bundle, handoff=handoff, manifest=manifest
    )
    assert _sha256(model_path) == manifest["model_artifact"]["byte_sha256"]

    selection = _json(f"{MODEL_SELECTION}/model-selection-handoff.json")
    assert handoff["selected_model_id"] == selection["selected_model_id"]
    assert handoff["selected_hyperparameters"] == selection["selected_hyperparameters"]
    assert handoff["test_partition_evaluation_count"] == 1
    assert handoff["readiness"]["operational_modeling_ready"] is False
    assert manifest["model_selection_handoff_reference"]["sha256"] == _sha256(
        ROOT / MODEL_SELECTION / "model-selection-handoff.json"
    )


@requires_artifacts
def test_mixture_diagnostics_are_declared_descriptive_and_persisted() -> None:
    exploration = _json(EXPLORATION)
    (grouped,) = exploration["leakage_and_dependencies"]["grouped_observation_dependencies"]
    assert grouped["varying_columns"] == ["Age"]
    assert any(review["review_id"] == "REV-GRP-001" for review in exploration["open_reviews"])

    selection = _json(f"{MODEL_SELECTION}/model-selection-handoff.json")
    assert selection["evaluation_diagnostics"]["group_overlap_columns"] == grouped["group_columns"]
    validation = _json(f"{MODEL_SELECTION}/selection-analysis.json")["mixture_overlap_sensitivity"]
    test = _json(f"{MODELS}/final-test-evidence.json")["group_overlap_diagnostic"]
    for diagnostic in (validation, test):
        assert diagnostic["diagnostic_only"] is True and diagnostic["used_for_selection"] is False
        assert diagnostic["group_columns"] == grouped["group_columns"]
    assert test["reference_partitions"] == ["train", "validation"]


# ---------------------------------------------------------------------------
# README and saved notebook outputs against the artifacts
# ---------------------------------------------------------------------------


@requires_artifacts
def test_readme_reports_the_persisted_results() -> None:
    readme = _readme()
    manifest = _json(f"{MODELS}/final-model-manifest.json")
    evidence = _json(f"{MODELS}/final-test-evidence.json")
    selection = _json(f"{MODEL_SELECTION}/model-selection-handoff.json")
    assert manifest["model_artifact"]["byte_sha256"] in readme
    for metric in ("mae", "rmse", "r2", "medae"):
        assert f"{selection['selected_validation_evidence'][metric]:.4f}" in readme, metric
        assert f"{evidence['metrics'][metric]:.4f}" in readme, metric
    assert f"{selection['selected_cv_evidence']['cv_mae_mean']:.4f}" in readme
    diagnostic = evidence["group_overlap_diagnostic"]
    for subset in ("seen_groups", "unseen_groups"):
        assert f"{diagnostic[subset]['metrics']['mae']:.4f}" in readme, subset
        assert f"| {diagnostic[subset]['row_count']} |" in readme, subset


@requires_artifacts
def test_saved_notebook_outputs_match_the_current_artifacts() -> None:
    selection = _json(f"{MODEL_SELECTION}/model-selection-handoff.json")
    assert selection["selected_model_id"] in _outputs(NOTEBOOKS[2])

    nb04 = _outputs(NOTEBOOKS[3])
    assert "reused_equivalent" not in nb04, "Notebook 04 outputs must come from the evaluating run"
    evidence = _json(f"{MODELS}/final-test-evidence.json")
    assert f"{evidence['metrics']['mae']:.4f}" in nb04

    nb05 = _outputs(NOTEBOOKS[4])
    readme = _readme()
    for example in README_EXAMPLES:
        rows = [line for line in nb05.splitlines() if example in line and line.rstrip().endswith("MPa")]
        assert rows, example
        prediction = rows[0].split()[-2]
        assert f"| `{example}` | {prediction} |" in readme, (example, prediction)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_every_versioned_figure_is_exported_by_notebook_01_and_referenced() -> None:
    on_disk = {path.name for path in (ROOT / "docs/images").glob("*.png")}
    exported = set(re.findall(r'docs/images/([\w-]+\.png)', _code(NOTEBOOKS[0])))
    referenced = set(re.findall(r"\]\(docs/images/([\w-]+\.png)\)", _readme()))
    assert on_disk == exported == referenced


# ---------------------------------------------------------------------------
# Smoke inference over the generated model
# ---------------------------------------------------------------------------


@requires_artifacts
def test_smoke_inference_loads_the_generated_model_through_every_gate() -> None:
    import pandas as pd

    from scripts.smoke_predict import load_validated_inference_pipeline, predict_continuous_batch

    if observed_runtime_mismatches(ROOT):
        pytest.skip("Deserialization requires the canonical runtime (pylock.toml).")

    pipeline, _, bundle, runtime_report = load_validated_inference_pipeline(
        project_root=ROOT, handoff_path=HANDOFF, bundle_path=BUNDLE, trusted_source=True
    )
    synthetic = pd.DataFrame(
        [[350.0, 50.0, 0.0, 180.0, 7.0, 1000.0, 780.0, 28]], columns=bundle["feature_order"]
    )
    prediction = predict_continuous_batch(pipeline, synthetic, bundle=bundle, runtime_report=runtime_report)
    assert runtime_report.compatible
    assert len(prediction) == 1 and 0.0 < float(prediction.iloc[0]) < 120.0
