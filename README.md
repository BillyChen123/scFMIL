# CellFM + Hierarchical Pooling (.publish)

This workspace contains a two-stage pipeline:
1) (Optional) Fine-tune CellFM and export per-cell embeddings.
2) Train a hierarchical attention pooling model to predict sample phenotypes (default: age_z_score)
   and recover cell/sample weights.

## Layout
- cellfm/: CellFM fine-tuning and embedding export.
- pooling/: Hierarchical attention pooling training + weight extraction.
- csv/: Gene mapping files required by CellFM.
- outputs/: Example outputs.

## Quick start

0) Install dependencies
```bash
pip install -r requirements.txt
```

1) Optional: CellFM fine-tuning
```bash
python -m cellfm.finetune --adata data/train.h5ad --ms-ckpt path/to.ckpt --out-dir outputs/cellfm
```

2) Export per-cell embeddings
```bash
python -m cellfm.embed --adata data/train.h5ad --pt-ckpt outputs/cellfm/model.pt \
  --out-adata data/with_emb.h5ad --emb-key X_cellfm
```

3) Pooling training (predicts z-score labels)
```bash
python -m pooling.train --adata data/with_emb.h5ad --emb_key X_cellfm --out_dir outputs/pooling
```

4) Weight extraction from a trained checkpoint
```bash
python -m pooling.extract --adata data/with_emb.h5ad --emb_key X_cellfm \
  --ckpt_path outputs/pooling/model.pt --out_dir outputs/pooling
```

## Data expectations

CellFM input:
- h5ad with gene names in `var_names`; mapping uses `csv/expand_gene_info.csv` and
  `csv/updated_hgcn.tsv`.
- obs columns: celltype (string), batch_id (int), train (0/1). Missing columns are auto-filled.
- use `--val-frac` in `cellfm.finetune` to create a validation split.

Pooling input:
- obs has sample id column (default: donor_id; use `--sample_key` to override).
- label column (default: age_z_score; use `--label_key` to pick).
- embeddings in `obsm[emb_key]` (default is X; use `--emb_key` to point to X_cellfm).
- optional integer cell id column via `--cell_id_key`; otherwise obs index is used.
- Leiden clustering is run per sample (set `--resolution`).

## Outputs

CellFM:
- `outputs/cellfm/model.pt`

Embedding export:
- `.h5ad` with `obsm[emb_key]` containing per-cell embeddings.

Pooling training:
- `model.pt`, `label_scaler.csv`, `test_pred.csv`

Weight extraction:
- `weights.h5ad` with obs columns: agg_weight, sample_weight, cell_weight, sample_ids

## Notes
- MindSpore is only needed when loading a MindSpore checkpoint with `--ms-ckpt`.
