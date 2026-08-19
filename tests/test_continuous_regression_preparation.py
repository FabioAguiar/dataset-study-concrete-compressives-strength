from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_exploration_handoff import load_and_validate_exploration_handoff
from scripts.prepare_data import (
    ContinuousRegressionSplitPolicy,
    DatasetValidationError,
    HandoffValidationError,
    SplitPolicyError,
    analyze_repeated_profiles_across_partitions,
    build_feature_manifest,
    build_split_manifest,
    describe_continuous_target,
    fingerprint_dataframe,
    fingerprint_dataframe_csv,
    fingerprint_file,
    prepare_tabular_dataset,
    separate_dataset_roles,
    split_continuous_regression_dataset,
    validate_raw_dataset,
    validate_regression_partitions,
    validate_regression_split_policy,
    validate_source_against_exploration_handoff,
)

ROOT = Path(__file__).resolve().parents[1]
SLUG = "concrete-compressive-strength"
HANDOFF_PATH = ROOT / "artifacts/exploration" / SLUG / "exploration-handoff.json"
RAW_ROOT = ROOT / "data/raw" / SLUG
FEATURES = ("Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age")
TARGET = "Concrete compressive strength"


@pytest.fixture
def handoff():
    return load_and_validate_exploration_handoff(HANDOFF_PATH, expected_dataset_slug=SLUG, expected_source_dataset_id=165)


@pytest.fixture
def frame():
    return pd.read_csv(RAW_ROOT / "dataset.csv")


@pytest.fixture
def policy():
    return ContinuousRegressionSplitPolicy("shuffled_random_snapshot", "educational_benchmark", .70, .15, .15, 42, True)


def source_gate(frame, handoff):
    return validate_source_against_exploration_handoff(
        frame, handoff=handoff, source_file=RAW_ROOT / "dataset.csv",
        project_root=ROOT, dataset_slug=SLUG,
        source_repository="UCI Machine Learning Repository", source_dataset_id=165,
        metadata_file=RAW_ROOT / "metadata.json", variables_file=RAW_ROOT / "variables.csv",
    )


def test_real_continuous_source_contract_is_accepted(frame, handoff):
    report = source_gate(frame, handoff)
    assert report.problem_type == "continuous_regression"
    assert report.target_classes == ()
    assert report.feature_columns == FEATURES


@pytest.mark.parametrize("mutation", ["problem", "classes", "feature_order", "resolution"])
def test_handoff_contract_divergence_fails_closed(frame, handoff, mutation):
    bad = copy.deepcopy(handoff)
    if mutation == "problem":
        bad["prediction_contract"]["problem_type"] = "multiclass_classification"
    elif mutation == "classes":
        bad["prediction_contract"]["target_classes"] = ["fake"]
    elif mutation == "feature_order":
        bad["feature_contract"]["feature_columns"] = list(reversed(FEATURES))
    else:
        bad["preparation_contract"]["decisions"] = [d for d in bad["preparation_contract"]["decisions"] if d.get("Decision ID") != "PREP-001"]
    with pytest.raises(DatasetValidationError):
        source_gate(frame, bad)


def test_source_id_sha_column_target_and_roles_fail_closed(tmp_path, frame, handoff):
    with pytest.raises(DatasetValidationError):
        validate_source_against_exploration_handoff(frame, handoff=handoff, source_file=RAW_ROOT/"dataset.csv", project_root=ROOT, dataset_slug=SLUG, source_repository="UCI Machine Learning Repository", source_dataset_id=999)
    changed = frame.copy(); changed.iloc[0, 0] += 1
    path = tmp_path / "dataset.csv"; changed.to_csv(path, index=False)
    bad = copy.deepcopy(handoff); bad["source"]["path"] = path.relative_to(tmp_path).as_posix(); bad["source"]["sha256"] = fingerprint_file(path)
    with pytest.raises(DatasetValidationError):
        validate_source_against_exploration_handoff(changed[list(reversed(changed.columns))], handoff=bad, source_file=path, project_root=tmp_path, dataset_slug=SLUG, source_repository="UCI Machine Learning Repository", source_dataset_id=165)
    with pytest.raises(DatasetValidationError):
        source_gate(frame.drop(columns=TARGET), handoff)


def test_continuous_target_validation(frame):
    kwargs=dict(column_order=tuple(frame.columns),identifier_columns=(),feature_columns=FEATURES,target_column=TARGET,target_classes=(),categorical_expected_values={},expected_types={c:"numeric" for c in frame.columns},problem_type="continuous_regression")
    assert validate_raw_dataset(frame, **kwargs).is_valid
    for value in (np.nan, np.inf):
        bad=frame.copy(); bad.loc[0,TARGET]=value
        with pytest.raises(DatasetValidationError): validate_raw_dataset(bad, **kwargs)
    bad=frame.copy(); bad[TARGET]=bad[TARGET].astype(str)
    with pytest.raises(DatasetValidationError): validate_raw_dataset(bad, **kwargs)


def test_slag_decimals_and_identity_projection_are_preserved(frame):
    before=frame.copy(deep=True); fingerprint=fingerprint_dataframe(frame)
    result=prepare_tabular_dataset(frame); prepared=result.dataframe
    pd.testing.assert_frame_equal(frame,before); pd.testing.assert_frame_equal(prepared,before)
    assert fingerprint_dataframe(frame)==fingerprint
    assert int(((prepared["Blast Furnace Slag"]%1).abs()>0).sum())==298
    assert result.rules==() and result.materialized_counts==()


def test_role_separation_has_no_target_or_membership(frame):
    roles=separate_dataset_roles(frame,identifier_columns=(),feature_columns=FEATURES,target_column=TARGET)
    assert roles.lineage.shape==(1030,0)
    assert tuple(roles.features.columns)==FEATURES
    assert TARGET not in roles.features and "__membership_key__" not in roles.features
    pd.testing.assert_series_equal(roles.target,frame[TARGET])


@pytest.mark.parametrize("change", ["fractions", "seed", "shuffle", "stratify", "mode"])
def test_regression_policy_validation(change, policy):
    values=policy.as_dict()
    kwargs=dict(evaluation_mode=policy.evaluation_mode,purpose=policy.purpose,train_fraction=policy.train_fraction,validation_fraction=policy.validation_fraction,test_fraction=policy.test_fraction,random_seed=policy.random_seed,shuffle=policy.shuffle,stratify_by=None)
    if change=="fractions": kwargs["train_fraction"]=.6
    elif change=="seed": kwargs["random_seed"]=42.0
    elif change=="shuffle": kwargs["shuffle"]=False
    elif change=="stratify": kwargs["stratify_by"]=TARGET
    else: kwargs["evaluation_mode"]="stratified_random_snapshot"
    with pytest.raises(SplitPolicyError): validate_regression_split_policy(ContinuousRegressionSplitPolicy(**kwargs))
    assert values["operational_validity"]=="unconfirmed"


def test_split_is_deterministic_complete_and_non_stratified(frame, policy):
    first=split_continuous_regression_dataset(frame,policy=policy)
    second=split_continuous_regression_dataset(frame,policy=policy)
    assert first.membership_mapping()==second.membership_mapping()
    assert tuple(len(x) for x in (first.train,first.validation,first.test))==(721,154,155)
    report=validate_regression_partitions(frame,first,identifier_columns=(),target_column=TARGET)
    assert report.is_valid and sum(dict(report.row_counts).values())==1030
    assert "non_stratified" in first.split_method


def test_split_assignment_is_independent_of_target(frame, policy):
    original=split_continuous_regression_dataset(frame,policy=policy)
    changed=frame.copy(); changed[TARGET]=np.arange(len(changed),dtype=float)*1000+0.5
    resplit=split_continuous_regression_dataset(changed,policy=policy)
    # Membership hashes contain y by design as post-assignment evidence; source row positions do not.
    for name in ("train","validation","test"):
        assert original.as_mapping()[name].index.tolist()==resplit.as_mapping()[name].index.tolist()


def test_target_diagnostics_are_non_gating(frame, policy):
    parts=split_continuous_regression_dataset(frame,policy=policy)
    report=validate_regression_partitions(frame,parts,identifier_columns=(),target_column=TARGET)
    for diagnostic in dict(report.target_diagnostics).values():
        assert set(("count","minimum","maximum","mean","median","standard_deviation","quantiles")) <= set(diagnostic)
        assert tuple(diagnostic["quantiles"])==("1%","5%","25%","50%","75%","95%","99%")
        assert diagnostic["diagnostic_only"] and not diagnostic["used_for_assignment_or_seed_selection"]
    extreme=frame.copy(); extreme.loc[parts.test.index,TARGET] += 1_000_000
    validate_regression_partitions(extreme,split_continuous_regression_dataset(extreme,policy=policy),identifier_columns=(),target_column=TARGET)


def test_repeated_profiles_are_preserved_and_reported(frame, policy):
    parts=split_continuous_regression_dataset(frame,policy=policy)
    evidence=analyze_repeated_profiles_across_partitions(frame,parts,feature_columns=FEATURES,target_column=TARGET)
    assert evidence["source_exact_row_equality_group_count"]==11
    assert evidence["source_exact_row_equality_row_count"]==36
    assert evidence["source_repeated_feature_profile_group_count"]==19
    assert evidence["target_conflicting_feature_profile_group_count"]==9
    assert evidence["exact_row_multiplicity_preserved"] and evidence["feature_profile_multiplicity_preserved"]
    assert evidence["proven_duplicate_identity"] is False


def test_continuous_manifests_have_no_class_semantics(frame, policy):
    parts=split_continuous_regression_dataset(frame,policy=policy)
    report=validate_regression_partitions(frame,parts,identifier_columns=(),target_column=TARGET)
    feature=build_feature_manifest(dataset_slug=SLUG,identifier_columns=(),feature_columns=FEATURES,numerical_features=FEATURES,categorical_features=(),categorical_expected_values={},target_column=TARGET,target_classes=(),expected_dtypes={},preprocessing_contract={"learned_transformations_fitted_in_notebook_02":False},prohibited_predictors=(TARGET,),problem_type="continuous_regression",target_semantics="Continuous / quantitative",target_unit="MPa",prediction_output="Continuous numeric value on the original target scale")
    assert feature["schema_version"]=="feature-manifest.v3" and feature["target_classes"]==[]
    assert "target_encoding_contract" not in feature and feature["target_contract"]["unit"]=="MPa"
    split=build_split_manifest(dataset_slug=SLUG,policy=policy,partitions=parts,validation=report,partition_paths={n:f"{n}.csv" for n in ("train","validation","test")},partition_sha256={n:fingerprint_dataframe_csv(f) for n,f in parts.as_mapping().items()})
    assert split["schema_version"]=="split-manifest.v3" and split["stratification"] is None
    assert "class_counts" not in split and "class_prevalence" not in split
    assert split["target_bins_created"] is False and split["seed_shopping_performed"] is False

