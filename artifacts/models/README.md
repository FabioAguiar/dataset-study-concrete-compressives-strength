Model runtime artifacts are generated locally and ignored by Git.

Expected Concrete finalization outputs after running Notebook 04:

```text
artifacts/models/concrete-compressive-strength/final-pipeline.joblib
artifacts/models/concrete-compressive-strength/final-model-manifest.json
artifacts/models/concrete-compressive-strength/final-test-evidence.json
artifacts/models/concrete-compressive-strength/inference-bundle.json
artifacts/models/concrete-compressive-strength/final-model-handoff.json
artifacts/models/concrete-compressive-strength/.final-evaluation-state.json
```

These files are local runtime outputs, not source code, and remain ignored by
Git. Notebook 04 materializes them; Notebook 05 consumes them read-only. The
serialized joblib model must be loaded only after its SHA-256 has been
validated according to the bundle and handoff contract, and only in the
runtime recorded by the bundle (`runtime_compatibility`: Python, pandas,
scikit-learn, joblib, and numpy), which is the environment pinned by
`.python-version` and `pylock.toml`.

The manifest records both `selected_estimator_fixed_constructor_parameters`
(the base constructor of the searched family) and
`selected_estimator_effective_parameters` (after the selected hyperparameters
override it). `final-test-evidence.json` keeps the pre-evaluation
`frozen_partition_reference` next to the post-evaluation `partition_state`, and
carries the descriptive `group_overlap_diagnostic` (test error for mixtures
seen versus unseen in train + validation).

Recreate the outputs by running the notebooks in order after acquiring the UCI
data. The hidden `.final-evaluation-state.json` receipt prevents a second test
evaluation: while a complete equivalent set exists, Notebook 04 reuses it.
