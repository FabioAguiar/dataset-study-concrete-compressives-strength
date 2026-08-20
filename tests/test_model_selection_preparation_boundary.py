from pathlib import Path

import pandas as pd

from scripts.prepare_data import load_and_validate_preparation_for_model_selection
from scripts.select_models import load_and_validate_model_selection_handoff


ROOT = Path(__file__).resolve().parents[1]
PREPARATION = "artifacts/preparation/concrete-compressive-strength/preparation-handoff.json"
SELECTION = "artifacts/model-selection/concrete-compressive-strength/model-selection-handoff.json"


def _read_names(monkeypatch):
    original = pd.read_csv
    calls = []

    def spy(path, *args, **kwargs):
        calls.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    return calls


def test_safe_preparation_loader_hashes_but_never_parses_prepared_or_test(monkeypatch):
    calls = _read_names(monkeypatch)
    loaded = load_and_validate_preparation_for_model_selection(
        project_root=ROOT, preparation_handoff_path=PREPARATION
    )
    assert calls.count("train.csv") == 1
    assert calls.count("validation.csv") == 1
    assert calls.count("prepared.csv") == 0
    assert calls.count("test.csv") == 0
    assert loaded.sealed_test_integrity_reference["sha256"]
    assert loaded.prepared_integrity_reference["sha256"]
    assert not hasattr(loaded, "test") and not hasattr(loaded, "prepared")


def test_v3_selection_loader_never_parses_prepared_or_test(monkeypatch):
    calls = _read_names(monkeypatch)
    loaded = load_and_validate_model_selection_handoff(
        project_root=ROOT, handoff_path=SELECTION
    )
    assert loaded["schema_version"] == "model-selection-handoff.v3"
    assert calls.count("train.csv") == 1
    assert calls.count("validation.csv") == 1
    assert "prepared.csv" not in calls and "test.csv" not in calls


def test_safe_frames_are_defensive_and_isolated():
    loaded = load_and_validate_preparation_for_model_selection(
        project_root=ROOT, preparation_handoff_path=PREPARATION
    )
    first = loaded.train
    first.iloc[0, 0] = -999
    assert loaded.train.iloc[0, 0] != -999
