Runtime data are intentionally not versioned.

Expected local layout:

```text
data/raw/concrete-compressive-strength/
data/processed/concrete-compressive-strength/
```

Acquire the UCI source snapshot with the locked environment activated:

```bash
python -m scripts.download_data \
  uci 165 \
  --destination data/raw/concrete-compressive-strength
```

`raw` is the local materialization of the UCI source. Its expected bytes are
pinned by the versioned [`contracts/source.json`](../contracts/source.json)
(SHA-256, size, rows, columns); Notebooks 01 and 02 verify every acquisition
against that contract before reading it. `dataset.csv` is written by pandas
from the `ucimlrepo` dataframe, so the pinned bytes also assume the locked
pandas version.

`processed` is produced by Notebook 02 and includes the prepared projection
and local split artifacts. These runtime files remain ignored by Git. The
notebooks authenticate their lineage and content fingerprints through
persisted manifests and handoffs.
