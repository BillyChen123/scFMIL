# CellFM MIL phenotype prediction

This repo contains a two-stage pipeline:
1) Fine-tune CellFM on your single-cell data and export per-cell embeddings.
2) Train a hierarchical attention pooling model to predict sample phenotypes and recover cell weights.

## Layout
- cellfm/: CellFM fine-tuning and embedding export
- pooling/: Hierarchical attention pooling for sample-level prediction
- csv/: Gene mapping files required by CellFM (not included)

## Quick start
python -m cellfm.finetune --adata data/train.h5ad --ms-ckpt path/to.ckpt --out-dir outputs/cellfm
python -m cellfm.embed --adata data/train.h5ad --pt-ckpt outputs/cellfm/model.pt --out-adata data/with_emb.h5ad
python -m pooling.train --adata data/with_emb.h5ad --emb-key X_cellfm --out-dir outputs/pooling
python -m pooling.train --adata data/with_emb.h5ad --emb-key X_cellfm --save_weights --out-dir outputs/pooling

## Data expectations
CellFM:
- h5ad with obs columns: celltype (string), batch_id (int), train (0/1/2). Missing fields are filled with defaults.
- genes should match the CellFM gene list; put mapping files in csv/.
Pooling:
- h5ad with obs columns: donor_id (sample id), age_value (phenotype).
- embeddings in obsm["X_cellfm"] or use --emb-key to point to another key.
Pooling output:
- labels are standardized using the training mean/std; predictions are saved back to raw scale in test_pred.csv.
- test_pred.csv includes both raw (label/pred) and standardized (label_z/pred_z) values.
- label_scaler.csv stores the mean/std.
- with --save_weights, exports per-cell attention weights to CSV and h5ad:
- train_cell_weights.csv / test_cell_weights.csv, plus train_weights.h5ad / test_weights.h5ad.
- h5ad obs columns: pooling_cell_id, pooling_sample_id, pooling_cluster_id, pooling_cell_weight, pooling_cluster_weight, pooling_weight.

## Notes
- MindSpore is only needed if you load a MindSpore checkpoint with --ms-ckpt.
- All paths are examples; adjust to your environment.
