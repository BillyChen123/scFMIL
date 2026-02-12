from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from pooling.dataset import SampleDataset
from .model import LegacyHierarchicalPooling


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


def build_samples(adata, expr, label_key, sample_key, resolution, cell_ids_all):
    samples = sorted(adata.obs[sample_key].unique())
    exprs, labels, cell_ids, sample_ids, clusters = [], [], [], [], []

    for sid in samples:
        mask = (adata.obs[sample_key] == sid).to_numpy()
        expr_i = expr[mask]
        cluster_i = leiden_cluster(expr_i, resolution=resolution)
        cell_ids_i = cell_ids_all[mask]
        label_i = adata.obs.loc[mask, label_key].iloc[0]

        exprs.append(expr_i)
        labels.append(label_i)
        cell_ids.append(cell_ids_i)
        sample_ids.append(sid)
        clusters.append(cluster_i)

    return exprs, labels, cell_ids, sample_ids, clusters


def pearson_corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return 0.0
    x = x.float()
    y = y.float()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt((vx * vx).sum()) * torch.sqrt((vy * vy).sum())
    if denom.item() == 0.0:
        return 0.0
    return float((vx * vy).sum() / denom)


def ensure_out_dir(path_str: str) -> Path:
    out_dir = Path(path_str).expanduser()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot create out_dir at '{out_dir}'. "
            "Please use a writable path such as './outputs/pooling2' or "
            "'/data/chenyz/project/Age_classcify/.publish/outputs/pooling2'."
        ) from exc
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", required=True)
    parser.add_argument("--emb_key", default="X")
    parser.add_argument("--label_key", default="age_z_score")
    parser.add_argument("--sample_key", default="donor_id")
    parser.add_argument("--cell_id_key", default=None)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="outputs/pooling2")
    parser.add_argument("--attn_hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = ensure_out_dir(args.out_dir)

    adata = sc.read_h5ad(args.adata)
    expr = read_expr(adata, args.emb_key)
    cell_ids_all = resolve_cell_ids(adata.obs, args.cell_id_key)

    sample_ids = adata.obs[args.sample_key].unique()
    rng = np.random.RandomState(args.seed)
    rng.shuffle(sample_ids)
    n_test = int(len(sample_ids) * args.test_size)
    test_ids = set(sample_ids[:n_test])
    train_ids = set(sample_ids[n_test:])

    train_mask = adata.obs[args.sample_key].isin(train_ids)
    test_mask = ~train_mask

    train_adata = adata[train_mask].copy()
    test_adata = adata[test_mask].copy()
    train_expr = expr[train_mask.to_numpy()]
    test_expr = expr[test_mask.to_numpy()]
    train_cell_ids = cell_ids_all[train_mask.to_numpy()]
    test_cell_ids = cell_ids_all[test_mask.to_numpy()]

    train_lists = build_samples(
        train_adata, train_expr, args.label_key, args.sample_key, args.resolution, train_cell_ids
    )
    test_lists = build_samples(
        test_adata, test_expr, args.label_key, args.sample_key, args.resolution, test_cell_ids
    )

    train_exprs, train_labels_raw, train_cell_ids, train_sample_names, train_clusters = train_lists
    test_exprs, test_labels_raw, test_cell_ids, test_sample_names, test_clusters = test_lists

    train_labels = np.asarray(train_labels_raw, dtype=float)
    test_labels = np.asarray(test_labels_raw, dtype=float)

    train_sample_idx = list(range(len(train_sample_names)))
    test_sample_idx = list(range(len(test_sample_names)))
    train_dataset = SampleDataset(
        train_exprs, train_labels, train_cell_ids, train_sample_idx, train_clusters
    )
    test_dataset = SampleDataset(
        test_exprs, test_labels, test_cell_ids, test_sample_idx, test_clusters
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=test_dataset.collate_fn,
    )

    input_dim = train_exprs[0].shape[1]
    model = LegacyHierarchicalPooling(
        input_dim,
        attn_hidden=args.attn_hidden,
        dropout=args.dropout,
        legacy_attn=True,
    ).to(args.device)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.MSELoss()
    use_amp = str(args.device).startswith("cuda")
    scaler_amp = GradScaler(enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_pcc = 0.0
        train_pred_z = []
        train_label_z = []
        progress = tqdm(train_loader, desc=f"training [{epoch}/{args.epochs}]")
        for step, batch in enumerate(progress):
            expr, labels, mask, cell_ids, sample_ids, cluster = batch
            expr = expr.to(args.device)
            labels = labels.to(args.device)
            mask = mask.to(args.device)
            cell_ids = cell_ids.to(args.device)
            cluster = cluster.to(args.device)

            with autocast(enabled=use_amp):
                pred, _, _, _, _, _ = model(expr, cell_ids, mask, cluster)
                pred = pred.squeeze(1)
                loss = loss_fn(pred, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

            running_loss += loss.item()
            train_pred_z.append(pred.detach().cpu())
            train_label_z.append(labels.detach().cpu())

            if pred.numel() > 1:
                running_pcc += pearson_corrcoef(pred, labels)
            avg_loss = running_loss / (step + 1)
            avg_pcc = running_pcc / (step + 1)
            progress.set_postfix(loss=avg_loss, pcc=avg_pcc)

        train_pred_z = torch.cat(train_pred_z, dim=0)
        train_label_z = torch.cat(train_label_z, dim=0)
        train_pcc = pearson_corrcoef(train_pred_z, train_label_z)
        print(f"Epoch {epoch}, training data acc: {train_pcc:.4f}")

        model.eval()
        running_loss = 0.0
        running_pcc = 0.0
        test_pred_z = []
        test_label_z = []
        test_sample_idx = []
        progress = tqdm(test_loader, desc=f"testing [{epoch}/{args.epochs}]")
        with torch.no_grad():
            for step, batch in enumerate(progress):
                expr, labels, mask, cell_ids, sample_ids, cluster = batch
                expr = expr.to(args.device)
                labels = labels.to(args.device)
                mask = mask.to(args.device)
                cell_ids = cell_ids.to(args.device)
                cluster = cluster.to(args.device)

                with autocast(enabled=use_amp):
                    pred, _, _, _, _, _ = model(expr, cell_ids, mask, cluster)
                    pred = pred.squeeze(1)
                    loss = loss_fn(pred, labels)

                running_loss += loss.item()
                test_pred_z.append(pred.detach().cpu())
                test_label_z.append(labels.detach().cpu())
                test_sample_idx.append(sample_ids.detach().cpu())

                if pred.numel() > 1:
                    running_pcc += pearson_corrcoef(pred, labels)
                avg_loss = running_loss / (step + 1)
                avg_pcc = running_pcc / (step + 1)
                progress.set_postfix(loss=avg_loss, pcc=avg_pcc)

        test_pred_z = torch.cat(test_pred_z, dim=0)
        test_label_z = torch.cat(test_label_z, dim=0)
        test_pcc = pearson_corrcoef(test_pred_z, test_label_z)
        print(f"Epoch {epoch}, testing data acc: {test_pcc:.4f}")

    torch.save(model.state_dict(), out_dir / "model.pt")
    pd.DataFrame({"label_mean": [0.0], "label_scale": [1.0]}).to_csv(
        out_dir / "label_scaler.csv", index=False
    )

    test_pred_z = test_pred_z.numpy()
    test_label_z = test_label_z.numpy()
    test_sample_idx = torch.cat(test_sample_idx, dim=0).numpy()
    sample_names = [test_sample_names[i] for i in test_sample_idx]
    pd.DataFrame(
        {
            "sample_id": sample_names,
            "label_z": test_label_z,
            "pred_z": test_pred_z,
        }
    ).to_csv(out_dir / "test_pred.csv", index=False)


if __name__ == "__main__":
    main()
