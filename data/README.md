Runtime data are intentionally not versioned.

Expected local layout after acquisition/preparation:

```text
data/raw/dry-bean/
data/processed/dry-bean/
```

Acquire the source snapshot with:

```bash
python -m scripts.download_data uci 602 --destination data/raw/dry-bean
```
