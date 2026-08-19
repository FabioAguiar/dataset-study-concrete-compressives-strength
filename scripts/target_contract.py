"""Reusable validation and presentation of classification target contracts.

The notebook remains responsible for declaring study-specific semantics such as
which column is the target and which labels are expected. This module performs
non-mutating structural validation without analyzing class prevalence or
encoding labels for modeling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import pandas as pd


ClassificationProblemType = Literal[
    "binary_classification",
    "multiclass_classification",
]

_SUMMARY_COLUMNS: Final[list[str]] = [
    "Contract item",
    "Value",
    "Interpretation",
]

_CLASS_COLUMNS: Final[list[str]] = [
    "Class label",
    "Declared",
    "Observed",
]


class TargetContractError(ValueError):
    """Raised when a classification target contract is inconsistent."""


@dataclass(frozen=True, slots=True)
class ClassificationTargetContract:
    """Validated, non-mutating classification target contract."""

    target: str
    problem_type: ClassificationProblemType
    expected_classes: tuple[object, ...]
    observed_classes: tuple[object, ...]
    source_role: str | None

    @property
    def class_count(self) -> int:
        """Return the number of classes declared by the contract."""
        return len(self.expected_classes)

    @property
    def positive_class(self) -> None:
        """Return no positive class for the neutral target contract layer."""
        return None

    @property
    def class_semantics(self) -> str:
        """Describe classification labels as nominal rather than ordinal."""
        return "Nominal / unordered"

    def summary_frame(self) -> pd.DataFrame:
        """Return the contract as a compact deterministic table."""
        rows = [
            {
                "Contract item": "Problem type",
                "Value": self.problem_type,
                "Interpretation": "Supervised classification task",
            },
            {
                "Contract item": "Target column",
                "Value": self.target,
                "Interpretation": "Outcome to be predicted",
            },
            {
                "Contract item": "Declared classes",
                "Value": self.class_count,
                "Interpretation": "Expected target cardinality",
            },
            {
                "Contract item": "Class semantics",
                "Value": self.class_semantics,
                "Interpretation": "Labels have no ordinal ranking",
            },
            {
                "Contract item": "Positive class",
                "Value": "Not applicable",
                "Interpretation": "No binary positive/negative semantics",
            },
            {
                "Contract item": "Source variable role",
                "Value": self.source_role or "Not checked",
                "Interpretation": "Role declared by source metadata",
            },
            {
                "Contract item": "Contract status",
                "Value": "Valid",
                "Interpretation": (
                    "Observed non-missing labels match the declared classes"
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def classes_frame(self) -> pd.DataFrame:
        """Return declared labels without leaking class-frequency analysis."""
        rows = [
            {
                "Class label": class_value,
                "Declared": True,
                "Observed": _contains_value(
                    self.observed_classes,
                    class_value,
                ),
            }
            for class_value in self.expected_classes
        ]
        return pd.DataFrame(rows, columns=_CLASS_COLUMNS)


def define_classification_target_contract(
    dataframe: pd.DataFrame,
    *,
    target: str,
    expected_classes: Sequence[object],
    problem_type: ClassificationProblemType,
    source_variables_file: str | Path | None = None,
) -> ClassificationTargetContract:
    """Validate and return a binary or multiclass target contract.

    Class counts and imbalance are intentionally out of scope. Missing target
    values are also left to the dedicated data-quality stages; they are ignored
    only while determining which non-missing labels are observed.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    target_name = _normalize_text(target, field="target")
    _require_unique_columns(dataframe)

    if target_name not in dataframe.columns:
        raise KeyError(f"Target column not found: {target_name!r}")

    normalized_classes = _normalize_expected_classes(expected_classes)
    _validate_problem_type(problem_type, normalized_classes)

    observed_series = dataframe[target_name].dropna()
    observed_classes = tuple(pd.unique(observed_series))

    unexpected_classes = tuple(
        value
        for value in observed_classes
        if not _contains_value(normalized_classes, value)
    )
    missing_classes = tuple(
        value
        for value in normalized_classes
        if not _contains_value(observed_classes, value)
    )

    failures: list[str] = []
    if unexpected_classes:
        failures.append(
            "unexpected target labels: "
            + ", ".join(repr(value) for value in unexpected_classes)
        )
    if missing_classes:
        failures.append(
            "declared target labels not observed: "
            + ", ".join(repr(value) for value in missing_classes)
        )

    source_role = None
    if source_variables_file is not None:
        source_role = _validate_source_target_role(
            source_variables_file,
            target=target_name,
        )

    if failures:
        raise TargetContractError(
            "Target contract validation failed: " + "; ".join(failures) + "."
        )

    return ClassificationTargetContract(
        target=target_name,
        problem_type=problem_type,
        expected_classes=normalized_classes,
        observed_classes=observed_classes,
        source_role=source_role,
    )


def _validate_problem_type(
    problem_type: ClassificationProblemType,
    classes: tuple[object, ...],
) -> None:
    if problem_type not in {
        "binary_classification",
        "multiclass_classification",
    }:
        raise TargetContractError(
            "problem_type must be 'binary_classification' or "
            "'multiclass_classification'."
        )

    expected_count = 2 if problem_type == "binary_classification" else 3
    if problem_type == "binary_classification" and len(classes) != 2:
        raise TargetContractError(
            "binary_classification requires exactly two declared classes."
        )
    if problem_type == "multiclass_classification" and len(classes) < expected_count:
        raise TargetContractError(
            "multiclass_classification requires at least three declared classes."
        )


def _validate_source_target_role(
    variables_file: str | Path,
    *,
    target: str,
) -> str:
    path = Path(variables_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"Source variables file not found: {path.name!r}"
        )

    try:
        variables = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise TargetContractError(
            "Could not read the source variables metadata table."
        ) from exc

    normalized_columns = {
        str(column).strip().lower(): str(column)
        for column in variables.columns
    }
    name_column = normalized_columns.get("name")
    role_column = normalized_columns.get("role")

    if name_column is None or role_column is None:
        raise TargetContractError(
            "Source variables metadata must contain 'name' and 'role' columns."
        )

    names = variables[name_column].astype("string").str.strip()
    matches = variables.loc[names.eq(target)]
    if len(matches) != 1:
        raise TargetContractError(
            "Source variables metadata must describe the target exactly once: "
            f"{target!r}."
        )

    role = str(matches.iloc[0][role_column]).strip()
    if role.lower() != "target":
        raise TargetContractError(
            f"Source metadata does not declare {target!r} as Target; "
            f"observed role={role!r}."
        )

    return role


def _normalize_expected_classes(
    values: Sequence[object],
) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(
            "expected_classes must be a sequence of class values, not a string."
        )

    normalized = tuple(values)
    if not normalized:
        raise TargetContractError("expected_classes must not be empty.")

    for value in normalized:
        if _is_missing_scalar(value):
            raise TargetContractError(
                "expected_classes must not contain missing values."
            )

    for index, value in enumerate(normalized):
        if _contains_value(normalized[:index], value):
            raise TargetContractError(
                f"expected_classes contains a duplicate value: {value!r}."
            )

    return normalized


def _normalize_text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise TargetContractError(f"{field} must not be empty.")
    return normalized


def _require_unique_columns(dataframe: pd.DataFrame) -> None:
    duplicated = dataframe.columns[
        dataframe.columns.duplicated(keep=False)
    ].tolist()
    if duplicated:
        raise TargetContractError(
            "DataFrame contains duplicated column labels: "
            + ", ".join(repr(value) for value in duplicated)
            + "."
        )


def _contains_value(values: Sequence[object], candidate: object) -> bool:
    return any(_values_equal(value, candidate) for value in values)


def _values_equal(left: object, right: object) -> bool:
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    if not pd.api.types.is_scalar(result):
        return False
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


def _is_missing_scalar(value: object) -> bool:
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if not pd.api.types.is_scalar(result):
        return False
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False
