from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from .model import WeightModel, detect_model_spec


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sc.settings.seed = seed


def read_expr(adata, emb_key: str):
    if emb_key and emb_key != "X":
        X = adata.obsm[emb_key]
    else:
        X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return X


def resolve_cell_ids(obs, cell_id_key: str | None):
    if cell_id_key and cell_id_key in obs.columns:
        vals = obs[cell_id_key].to_numpy()
        try:
            return vals.astype(int)
        except Exception:
            return np.arange(len(obs), dtype=int)
    try:
        return obs.index.astype(int).to_numpy()
    except Exception:
        return np.arange(len(obs), dtype=int)


def leiden_cluster(expr: np.ndarray, resolution: float = 0.5) -> np.ndarray:
    adata = sc.AnnData(X=expr)
    if len(adata) > 10:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata, use_rep="X_pca")
        sc.tl.leiden(adata, resolution=resolution, key_added="cluster")
    else:
        adata.obs["cluster"] = 0
    return adata.obs["cluster"].astype(int).values


def build_padded_samples(adata, expr, sample_key: str, resolution: float, cell_ids_all: np.ndarray):
    sample_ids = adata.obs[sample_key].unique()
    exprs, masks, cell_ids, clusters = [], [], [], []
    max_cells = 0

    for sid in sample_ids:
        mask = (adata.obs[sample_key] == sid).to_numpy()
        expr_i = expr[mask]
        cluster_i = leiden_cluster(expr_i, resolution)
        cell_ids_i = cell_ids_all[mask]

        exprs.append(expr_i)
        clusters.append(cluster_i)
        cell_ids.append(cell_ids_i)
        max_cells = max(max_cells, expr_i.shape[0])

    feat_dim = expr.shape[1]
    n_samples = len(exprs)
    expr_pad = np.zeros((n_samples, max_cells, feat_dim), dtype=expr.dtype)
    mask_pad = np.zeros((n_samples, max_cells), dtype=bool)
    cell_id_pad = np.full((n_samples, max_cells), -1, dtype=int)
    cluster_pad = np.full((n_samples, max_cells), -1, dtype=int)

    for i in range(n_samples):
        n = exprs[i].shape[0]
        expr_pad[i, :n] = exprs[i]
        mask_pad[i, :n] = True
        cell_id_pad[i, :n] = cell_ids[i]
        cluster_pad[i, :n] = clusters[i]

    return expr_pad, mask_pad, cell_id_pad, cluster_pad, sample_ids


class PaddedDataset(Dataset):
    def __init__(self, expr, mask, cell_ids, cluster, sample_ids):
        self.expr = torch.tensor(expr, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.cell_ids = torch.tensor(cell_ids, dtype=torch.long)
        self.cluster = torch.tensor(cluster, dtype=torch.long)
        self.sample_ids = np.asarray(sample_ids)

    def __len__(self):
        return self.expr.shape[0]

    def __getitem__(self, idx):
        return (
            self.expr[idx],
            self.mask[idx],
            self.cell_ids[idx],
            self.cluster[idx],
            self.sample_ids[idx],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", required=True)
    parser.add_argument("--emb_key", default="X")
    parser.add_argument("--sample_key", default="donor_id")
    parser.add_argument("--cell_id_key", default=None)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--attn_hidden", type=int, default=None)
    parser.add_argument("--out_dir", default="outputs/pooling2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref_adata", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.adata)
    expr = read_expr(adata, args.emb_key)
    cell_ids_all = resolve_cell_ids(adata.obs, args.cell_id_key)

    expr_pad, mask_pad, cell_id_pad, cluster_pad, sample_ids = build_padded_samples(
        adata, expr, args.sample_key, args.resolution, cell_ids_all
    )

    dataset = PaddedDataset(expr_pad, mask_pad, cell_id_pad, cluster_pad, sample_ids)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    spec = detect_model_spec(state_dict, input_dim=expr.shape[1], attn_hidden=args.attn_hidden)
    model = WeightModel(
        input_dim=spec.input_dim,
        attn_hidden=spec.attn_hidden,
        legacy_attn=spec.legacy_attn,
    ).to(args.device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    cell_id_to_pos = {cid: i for i, cid in enumerate(cell_ids_all)}
    n_cells = adata.n_obs
    agg_weight_all = np.full(n_cells, np.nan, dtype=float)
    sample_weight_all = np.full(n_cells, np.nan, dtype=float)
    sample_ids_all = np.full(n_cells, None, dtype=object)

    cell_to_sample = np.full(n_cells, None, dtype=object)
    for sid in sample_ids:
        mask = (adata.obs[args.sample_key] == sid).to_numpy()
        cell_to_sample[np.where(mask)[0]] = sid

    with torch.no_grad():
        for expr_b, mask_b, cell_ids_b, cluster_b, _ in loader:
            expr_b = expr_b.to(args.device)
            mask_b = mask_b.to(args.device)
            cell_ids_b = cell_ids_b.to(args.device)
            cluster_b = cluster_b.to(args.device)

            sample_weight, agg_weight, valid_idx = model(expr_b, cell_ids_b, mask_b, cluster_b)
            valid_idx = valid_idx.cpu().numpy()
            pos = np.array([cell_id_to_pos[int(cid)] for cid in valid_idx], dtype=int)

            agg_weight_all[pos] = agg_weight.detach().cpu().numpy()
            sample_weight_all[pos] = sample_weight.detach().cpu().numpy()
            sample_ids_all[pos] = cell_to_sample[pos]

    cell_weight_all = agg_weight_all * sample_weight_all

    adata_out = adata.copy()
    adata_out.obs["agg_weight"] = agg_weight_all
    adata_out.obs["sample_weight"] = sample_weight_all
    adata_out.obs["cell_weight"] = cell_weight_all
    adata_out.obs["sample_ids"] = sample_ids_all

    out_path = out_dir / "weights.h5ad"
    adata_out.write(out_path)

    if args.ref_adata:
        ref = sc.read_h5ad(args.ref_adata)
        if "sample_weight" in ref.obs:
            ref_w = ref.obs["sample_weight"].to_numpy(dtype=float)
            mask = ~np.isnan(sample_weight_all) & ~np.isnan(ref_w)
            if mask.any():
                corr = np.corrcoef(sample_weight_all[mask], ref_w[mask])[0, 1]
                print(f"ref sample_weight corr: {corr:.4f}")


if __name__ == "__main__":
    main()
