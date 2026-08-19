"""Tests for reusable classification target-contract validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.target_contract import (
    TargetContractError,
    define_classification_target_contract,
)


DRY_BEAN_CLASSES = (
    "SEKER",
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "DERMASON",
    "HOROZ",
    "SIRA",
)


def _multiclass_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Area": range(7),
            "Class": DRY_BEAN_CLASSES,
        }
    )


def test_multiclass_contract_validates_declared_labels() -> None:
    report = define_classification_target_contract(
        _multiclass_frame(),
        target="Class",
        expected_classes=DRY_BEAN_CLASSES,
        problem_type="multiclass_classification",
    )

    assert report.target == "Class"
    assert report.problem_type == "multiclass_classification"
    assert report.class_count == 7
    assert report.positive_class is None
    assert report.class_semantics == "Nominal / unordered"
    assert report.expected_classes == DRY_BEAN_CLASSES
    assert report.classes_frame()["Observed"].all()


def test_contract_rejects_unexpected_label() -> None:
    dataframe = _multiclass_frame()
    dataframe.loc[0, "Class"] = "UNKNOWN"

    with pytest.raises(TargetContractError, match="unexpected target labels"):
        define_classification_target_contract(
            dataframe,
            target="Class",
            expected_classes=DRY_BEAN_CLASSES,
            problem_type="multiclass_classification",
        )


def test_contract_rejects_declared_label_not_observed() -> None:
    dataframe = _multiclass_frame().loc[lambda frame: frame["Class"] != "SIRA"]

    with pytest.raises(
        TargetContractError,
        match="declared target labels not observed",
    ):
        define_classification_target_contract(
            dataframe,
            target="Class",
            expected_classes=DRY_BEAN_CLASSES,
            problem_type="multiclass_classification",
        )


def test_missing_targets_do_not_change_label_contract() -> None:
    dataframe = pd.concat(
        [
            _multiclass_frame(),
            pd.DataFrame({"Area": [10], "Class": [None]}),
        ],
        ignore_index=True,
    )

    report = define_classification_target_contract(
        dataframe,
        target="Class",
        expected_classes=DRY_BEAN_CLASSES,
        problem_type="multiclass_classification",
    )

    assert report.class_count == 7


def test_multiclass_requires_at_least_three_classes() -> None:
    dataframe = pd.DataFrame({"Class": ["A", "B"]})

    with pytest.raises(TargetContractError, match="at least three"):
        define_classification_target_contract(
            dataframe,
            target="Class",
            expected_classes=("A", "B"),
            problem_type="multiclass_classification",
        )


def test_binary_contract_requires_exactly_two_classes() -> None:
    dataframe = pd.DataFrame({"Class": ["A", "B", "C"]})

    with pytest.raises(TargetContractError, match="exactly two"):
        define_classification_target_contract(
            dataframe,
            target="Class",
            expected_classes=("A", "B", "C"),
            problem_type="binary_classification",
        )


def test_source_variables_must_declare_target_role(tmp_path: Path) -> None:
    variables_file = tmp_path / "variables.csv"
    pd.DataFrame(
        {
            "name": ["Area", "Class"],
            "role": ["Feature", "Target"],
        }
    ).to_csv(variables_file, index=False)

    report = define_classification_target_contract(
        _multiclass_frame(),
        target="Class",
        expected_classes=DRY_BEAN_CLASSES,
        problem_type="multiclass_classification",
        source_variables_file=variables_file,
    )

    assert report.source_role == "Target"


def test_source_variables_reject_wrong_target_role(tmp_path: Path) -> None:
    variables_file = tmp_path / "variables.csv"
    pd.DataFrame(
        {
            "name": ["Area", "Class"],
            "role": ["Feature", "Feature"],
        }
    ).to_csv(variables_file, index=False)

    with pytest.raises(TargetContractError, match="does not declare"):
        define_classification_target_contract(
            _multiclass_frame(),
            target="Class",
            expected_classes=DRY_BEAN_CLASSES,
            problem_type="multiclass_classification",
            source_variables_file=variables_file,
        )
