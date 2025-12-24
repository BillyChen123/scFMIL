from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .dataset import SampleDataset
from .model import HierarchicalPooling


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def leiden_cluster(expr: np.ndarray, resolution: float = 5.0) -> np.ndarray:
    adata = sc.AnnData(X=expr)
    if len(adata) > 10:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata, use_rep="X_pca")
        sc.tl.leiden(adata, resolution=resolution, key_added="cluster")
    else:
        adata.obs["cluster"] = 0
    return adata.obs["cluster"].astype(int).values


def read_expr(adata, emb_key: str):
    if emb_key and emb_key != "X":
        X = adata.obsm[emb_key]
    else:
        X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return X


def build_samples(adata, expr, label_key, sample_key, resolution):
    samples = sorted(adata.obs[sample_key].unique())
    cell_ids_all = np.arange(adata.n_obs)
    exprs, labels, cell_ids, sample_ids, clusters = [], [], [], [], []

    for sid in samples:
        mask = adata.obs[sample_key] == sid
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


def pearson_corr(x, y) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def save_predictions(out_path: Path, sample_names, labels_raw, preds_raw, labels_z=None, preds_z=None) -> None:
    data = {
        "sample_id": sample_names,
        "label": labels_raw,
        "pred": preds_raw,
    }
    if labels_z is not None and preds_z is not None:
        data["label_z"] = labels_z
        data["pred_z"] = preds_z
    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False)


def label_stats(labels: np.ndarray):
    mean = float(np.mean(labels))
    std = float(np.std(labels))
    if std <= 0.0:
        raise ValueError("label_std is zero; cannot standardize labels.")
    return mean, std


def build_cell_cluster_map(cell_ids_list, clusters_list, n_cells: int) -> np.ndarray:
    cluster_map = np.full(n_cells, -1, dtype=int)
    for cell_ids, clusters in zip(cell_ids_list, clusters_list):
        cluster_map[cell_ids] = clusters
    return cluster_map


def collect_cell_weights(
    model,
    loader,
    cell_id_to_sample,
    cell_id_to_cluster,
    cell_id_to_name,
    device: str,
    desc: str | None = None,
):
    model.eval()
    frames = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            expr, labels, mask, cell_ids, sample_ids, cluster = batch
            expr = expr.to(device)
            mask = mask.to(device)
            cell_ids = cell_ids.to(device)
            cluster = cluster.to(device)

            pred, sample_emb, cluster_weights, cell_w, cluster_w, cell_ids_flat = model(
                expr, cell_ids, mask, cluster
            )
            if cell_ids_flat.numel() == 0:
                continue
            cell_w_np = cell_w.cpu().numpy()
            cluster_w_np = cluster_w.cpu().numpy()
            combined = cell_w_np * cluster_w_np
            cell_ids_np = cell_ids_flat.cpu().numpy()
            frame = pd.DataFrame(
                {
                    "cell_id": cell_ids_np.astype(int),
                    "cell_name": cell_id_to_name[cell_ids_np],
                    "sample_id": cell_id_to_sample[cell_ids_np],
                    "cluster_id": cell_id_to_cluster[cell_ids_np],
                    "cell_weight": cell_w_np.astype(float),
                    "cluster_weight": cluster_w_np.astype(float),
                    "weight": combined.astype(float),
                }
            )
            frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "cell_id",
                "cell_name",
                "sample_id",
                "cluster_id",
                "cell_weight",
                "cluster_weight",
                "weight",
            ]
        )

    return pd.concat(frames, ignore_index=True)


def build_weights_adata(adata, expr, weights_df, sample_key: str, emb_key: str):
    obs = adata.obs.copy()
    n_cells = adata.n_obs
    cell_weight = np.full(n_cells, np.nan, dtype=float)
    cluster_weight = np.full(n_cells, np.nan, dtype=float)
    weight = np.full(n_cells, np.nan, dtype=float)
    cluster_id = np.full(n_cells, -1, dtype=int)

    idx = weights_df["cell_id"].to_numpy(dtype=int)
    cell_weight[idx] = weights_df["cell_weight"].to_numpy(dtype=float)
    cluster_weight[idx] = weights_df["cluster_weight"].to_numpy(dtype=float)
    weight[idx] = weights_df["weight"].to_numpy(dtype=float)
    cluster_id[idx] = weights_df["cluster_id"].to_numpy(dtype=int)

    obs["pooling_cell_id"] = obs.index.astype(str)
    obs["pooling_sample_id"] = obs[sample_key].values
    obs["pooling_cluster_id"] = cluster_id
    obs["pooling_cell_weight"] = cell_weight
    obs["pooling_cluster_weight"] = cluster_weight
    obs["pooling_weight"] = weight

    if emb_key and emb_key != "X":
        var = pd.DataFrame(index=[f"emb_{i}" for i in range(expr.shape[1])])
        adata_out = sc.AnnData(X=expr, obs=obs, var=var)
    else:
        adata_out = sc.AnnData(X=expr, obs=obs, var=adata.var.copy())
    return adata_out


def run_diagnostics(
    model,
    loader,
    sample_names,
    device: str,
    out_path: Path,
    desc: str,
):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            expr, labels, mask, cell_ids, sample_ids, cluster = batch
            expr = expr.to(device)
            mask = mask.to(device)
            cell_ids = cell_ids.to(device)
            cluster = cluster.to(device)

            outputs = model(expr, cell_ids, mask, cluster, return_debug=True)
            pred, sample_emb, cluster_weights, cell_w, cluster_w, cell_ids_flat, debug = outputs

            cluster_scores = debug["cluster_scores"]
            cluster_mask = debug["cluster_mask"]
            cluster_emb = debug["cluster_emb"]

            for i in range(cluster_mask.shape[0]):
                valid = cluster_mask[i]
                n_clusters = int(valid.sum().item())
                sample_idx = int(sample_ids[i].item())
                sample_name = sample_names[sample_idx]

                if n_clusters <= 1:
                    rows.append(
                        {
                            "sample_id": sample_name,
                            "n_clusters": n_clusters,
                            "cluster_weight_var": 0.0,
                            "cluster_score_var": 0.0,
                            "cluster_emb_var": 0.0,
                        }
                    )
                    continue

                weights = cluster_weights[i][valid]
                scores = cluster_scores[i][valid]
                emb = cluster_emb[i][valid]

                rows.append(
                    {
                        "sample_id": sample_name,
                        "n_clusters": n_clusters,
                        "cluster_weight_var": float(weights.var(unbiased=False).item()),
                        "cluster_score_var": float(scores.var(unbiased=False).item()),
                        "cluster_emb_var": float(emb.var(dim=0, unbiased=False).mean().item()),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    if df.empty:
        print(f"{desc}: no diagnostics rows collected")
        return

    print(f"{desc}: cluster count stats")
    print(df["n_clusters"].describe())
    print(f"{desc}: cluster_weight_var stats")
    print(df["cluster_weight_var"].describe())
    print(f"{desc}: cluster_score_var stats")
    print(df["cluster_score_var"].describe())
    print(f"{desc}: cluster_emb_var stats")
    print(df["cluster_emb_var"].describe())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", required=True)
    parser.add_argument("--emb_key", default="X")
    parser.add_argument("--label_key", default="age_value")
    parser.add_argument("--sample_key", default="donor_id")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="outputs/pooling")
    parser.add_argument("--attn_hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_weights", action="store_true")
    parser.add_argument(
        "--cluster_pool",
        choices=["auto", "attn", "mean", "mean_attn"],
        default="auto",
    )
    parser.add_argument("--cluster_alpha", type=float, default=0.2)
    parser.add_argument("--use_mean_pool", action="store_true")
    parser.add_argument("--use_gated_attn", action="store_true")
    parser.add_argument("--cluster_entropy_weight", type=float, default=0.0)
    parser.add_argument("--cluster_var_weight", type=float, default=0.0)
    parser.add_argument("--cluster_var_target", type=float, default=0.1)
    parser.add_argument("--cluster_var_eps", type=float, default=1e-6)
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.adata)
    expr = read_expr(adata, args.emb_key)

    sample_ids = adata.obs[args.sample_key].unique()
    train_ids, test_ids = train_test_split(
        sample_ids, test_size=args.test_size, random_state=args.seed, shuffle=True
    )

    train_mask = adata.obs[args.sample_key].isin(train_ids)
    test_mask = ~train_mask

    train_adata = adata[train_mask].copy()
    test_adata = adata[test_mask].copy()

    train_expr = expr[train_mask.to_numpy()]
    test_expr = expr[test_mask.to_numpy()]

    train_lists = build_samples(
        train_adata, train_expr, args.label_key, args.sample_key, args.resolution
    )
    test_lists = build_samples(
        test_adata, test_expr, args.label_key, args.sample_key, args.resolution
    )

    train_exprs, train_labels_raw, train_cell_ids, train_sample_names, train_clusters = train_lists
    test_exprs, test_labels_raw, test_cell_ids, test_sample_names, test_clusters = test_lists

    train_labels_raw = np.array(train_labels_raw, dtype=float)
    test_labels_raw = np.array(test_labels_raw, dtype=float)

    label_mean, label_std = label_stats(train_labels_raw)
    train_labels = (train_labels_raw - label_mean) / label_std
    test_labels = (test_labels_raw - label_mean) / label_std

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

    if args.cluster_pool == "auto":
        cluster_pool = "mean_attn" if args.use_mean_pool else "attn"
    else:
        cluster_pool = args.cluster_pool
        if args.use_mean_pool:
            print("注意: 已指定 --cluster_pool，忽略 --use_mean_pool。")

    input_dim = train_exprs[0].shape[1]
    model = HierarchicalPooling(
        input_dim,
        attn_hidden=args.attn_hidden,
        dropout=args.dropout,
        cluster_pool=cluster_pool,
        use_gated_attn=args.use_gated_attn,
        mean_attn_alpha=args.cluster_alpha,
    )
    model = model.to(args.device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"train {epoch}/{args.epochs}"):
            expr, labels, mask, cell_ids, sample_ids, cluster = batch
            expr = expr.to(args.device)
            labels = labels.to(args.device)
            mask = mask.to(args.device)
            cell_ids = cell_ids.to(args.device)
            cluster = cluster.to(args.device)

            if args.cluster_var_weight > 0.0:
                (
                    pred,
                    sample_emb,
                    cluster_weights,
                    cell_w,
                    cluster_w,
                    cell_ids_flat,
                    debug,
                ) = model(expr, cell_ids, mask, cluster, return_debug=True)
            else:
                pred, sample_emb, cluster_weights, cell_w, cluster_w, cell_ids_flat = model(
                    expr, cell_ids, mask, cluster
                )

            pred = pred.squeeze(1)
            loss = loss_fn(pred, labels)
            if args.cluster_entropy_weight > 0.0:
                eps = 1e-9
                ent = -(cluster_weights * (cluster_weights + eps).log()).sum(dim=1)
                loss = loss + args.cluster_entropy_weight * ent.mean()
            if args.cluster_var_weight > 0.0:
                emb = debug["cluster_emb"]
                mask_b = debug["cluster_mask"]
                counts = mask_b.sum(dim=1, keepdim=True)
                valid = counts.squeeze(1) > 1
                if valid.any():
                    mask_f = mask_b.unsqueeze(-1).float()
                    mean = (emb * mask_f).sum(dim=1) / counts.clamp(min=1.0)
                    diff = emb - mean.unsqueeze(1)
                    var = (diff * diff * mask_f).sum(dim=1) / counts.clamp(min=1.0)
                    std = torch.sqrt(var + args.cluster_var_eps)
                    var_loss = torch.relu(args.cluster_var_target - std)
                    loss = loss + args.cluster_var_weight * var_loss[valid].mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item()

        avg_loss = running / max(len(train_loader), 1)
        scheduler.step(avg_loss)

        model.eval()
        preds = []
        labels = []
        sample_idx = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"test {epoch}/{args.epochs}"):
                expr, label, mask, cell_ids, sidx, cluster = batch
                expr = expr.to(args.device)
                label = label.to(args.device)
                mask = mask.to(args.device)
                cell_ids = cell_ids.to(args.device)
                cluster = cluster.to(args.device)

                pred, _, _, _, _, _ = model(expr, cell_ids, mask, cluster)
                preds.append(pred.squeeze(1).cpu().numpy())
                labels.append(label.cpu().numpy())
                sample_idx.append(sidx.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        sample_idx = np.concatenate(sample_idx, axis=0)

        labels_raw = labels * label_std + label_mean
        preds_raw = preds * label_std + label_mean

        mse = mean_squared_error(labels_raw, preds_raw)
        mae = mean_absolute_error(labels_raw, preds_raw)
        r2 = r2_score(labels_raw, preds_raw)
        pcc = pearson_corr(labels_raw, preds_raw)

        print(
            f"epoch {epoch}: train_loss={avg_loss:.6f} test_mse={mse:.6f} "
            f"mae={mae:.6f} r2={r2:.6f} pcc={pcc:.6f}"
        )

    sample_names = [test_sample_names[i] for i in sample_idx]
    save_predictions(
        out_dir / "test_pred.csv",
        sample_names,
        labels_raw,
        preds_raw,
        labels_z=labels,
        preds_z=preds,
    )
    torch.save(model.state_dict(), out_dir / "model.pt")
    pd.DataFrame(
        {
            "label_mean": [label_mean],
            "label_std": [label_std],
        }
    ).to_csv(out_dir / "label_scaler.csv", index=False)

    if args.save_weights:
        train_weight_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=train_dataset.collate_fn,
        )
        test_weight_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=test_dataset.collate_fn,
        )

        train_cell_id_to_sample = np.asarray(train_adata.obs[args.sample_key].values)
        test_cell_id_to_sample = np.asarray(test_adata.obs[args.sample_key].values)
        train_cell_id_to_name = np.asarray(train_adata.obs_names)
        test_cell_id_to_name = np.asarray(test_adata.obs_names)
        train_cell_id_to_cluster = build_cell_cluster_map(
            train_cell_ids, train_clusters, train_adata.n_obs
        )
        test_cell_id_to_cluster = build_cell_cluster_map(
            test_cell_ids, test_clusters, test_adata.n_obs
        )

        train_weights_df = collect_cell_weights(
            model,
            train_weight_loader,
            train_cell_id_to_sample,
            train_cell_id_to_cluster,
            train_cell_id_to_name,
            args.device,
            desc="weights train",
        )
        test_weights_df = collect_cell_weights(
            model,
            test_weight_loader,
            test_cell_id_to_sample,
            test_cell_id_to_cluster,
            test_cell_id_to_name,
            args.device,
            desc="weights test",
        )

        train_weights_df.to_csv(out_dir / "train_cell_weights.csv", index=False)
        test_weights_df.to_csv(out_dir / "test_cell_weights.csv", index=False)
        test_weights_df.to_csv(out_dir / "cell_weights.csv", index=False)

        train_weights_adata = build_weights_adata(
            train_adata, train_expr, train_weights_df, args.sample_key, args.emb_key
        )
        test_weights_adata = build_weights_adata(
            test_adata, test_expr, test_weights_df, args.sample_key, args.emb_key
        )
        train_weights_adata.write(out_dir / "train_weights.h5ad")
        test_weights_adata.write(out_dir / "test_weights.h5ad")

    if args.diagnose:
        diagnose_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=test_dataset.collate_fn,
        )
        run_diagnostics(
            model,
            diagnose_loader,
            test_sample_names,
            args.device,
            out_dir / "diagnostics_test.csv",
            desc="diagnostics test",
        )


if __name__ == "__main__":
    main()
