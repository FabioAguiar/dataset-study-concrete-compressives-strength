Model runtime artifacts are generated locally and ignored by Git.

Expected Dry Bean finalization outputs after running Notebook 04:

```text
artifacts/models/dry-bean/final-pipeline.joblib
artifacts/models/dry-bean/final-model-manifest.json
artifacts/models/dry-bean/final-test-evidence.json
artifacts/models/dry-bean/inference-bundle.json
artifacts/models/dry-bean/final-model-handoff.json
```

These files are evidence and runtime outputs, not source code. Recreate them by
running the notebooks in order after acquiring the UCI data.
