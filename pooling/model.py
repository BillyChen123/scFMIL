from __future__ import annotations

import torch
from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, gated: bool = False):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.gated = gated
        if gated:
            self.gate = nn.Linear(input_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_scores: bool = False):
        # x: [batch, n_items, input_dim], mask: [batch, n_items]
        h = torch.tanh(self.fc(x))
        if self.gated:
            h = h * torch.sigmoid(self.gate(x))
        scores = self.context(h).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        # pooled: [batch, input_dim], weights: [batch, n_items]
        if return_scores:
            return pooled, weights, scores
        return pooled, weights


class MeanPooling(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        mask = mask.unsqueeze(-1)
        mask_f = mask.float()
        x = x * mask_f
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = x.sum(dim=1) / denom
        weights = mask_f.squeeze(-1) / denom
        return pooled, weights


class ClusterAggregator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        pool_mode: str = "attn",
        use_gated_attn: bool = False,
        mean_attn_alpha: float = 1.0,
    ):
        super().__init__()
        if pool_mode not in {"attn", "mean", "mean_attn"}:
            raise ValueError(f"Unknown pool_mode: {pool_mode}")
        self.pool_mode = pool_mode
        if pool_mode in {"attn", "mean_attn"}:
            self.pool = AttentionPooling(input_dim, hidden_dim, gated=use_gated_attn)
        if pool_mode in {"mean", "mean_attn"}:
            self.mean_pool = MeanPooling()
        self.mean_attn_alpha = mean_attn_alpha

    def forward(self, expr, cell_ids, mask, cluster):
        # expr: [max_cells, input_dim], cell_ids/mask/cluster: [max_cells]
        valid = mask
        expr = expr[valid]
        cell_ids = cell_ids[valid]
        cluster = cluster[valid]
        # expr: [n_cells, input_dim]

        unique = torch.unique(cluster[cluster >= 0])
        n_clusters = unique.numel()
        if n_clusters == 0:
            empty = expr.new_zeros((1, expr.shape[1]))
            return empty, expr.new_zeros((1, 1)), cell_ids.new_full((1, 1), -1)

        max_cells = 0
        for cid in unique:
            max_cells = max(max_cells, int((cluster == cid).sum().item()))

        feat_dim = expr.shape[1]
        cluster_expr = expr.new_zeros((n_clusters, max_cells, feat_dim))
        cluster_ids = cell_ids.new_full((n_clusters, max_cells), -1)
        cluster_mask = torch.zeros((n_clusters, max_cells), dtype=torch.bool, device=expr.device)

        for i, cid in enumerate(unique):
            c_mask = cluster == cid
            c_expr = expr[c_mask]
            c_ids = cell_ids[c_mask]
            n = c_expr.shape[0]
            cluster_expr[i, :n] = c_expr
            cluster_ids[i, :n] = c_ids
            cluster_mask[i, :n] = True

        if self.pool_mode == "mean":
            cluster_emb, cell_weights = self.mean_pool(cluster_expr, cluster_mask)
        else:
            attn_emb, cell_weights = self.pool(cluster_expr, cluster_mask)
            if self.pool_mode == "mean_attn":
                mean_emb, _ = self.mean_pool(cluster_expr, cluster_mask)
                cluster_emb = mean_emb + self.mean_attn_alpha * attn_emb
            else:
                cluster_emb = attn_emb
        # cluster_emb: [n_clusters, input_dim], cell_weights: [n_clusters, max_cells]
        return cluster_emb, cell_weights, cluster_ids


class HierarchicalPooling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        attn_hidden: int = 64,
        dropout: float = 0.2,
        cluster_pool: str = "attn",
        use_gated_attn: bool = False,
        mean_attn_alpha: float = 1.0,
    ):
        super().__init__()
        self.cluster = ClusterAggregator(
            input_dim,
            attn_hidden,
            pool_mode=cluster_pool,
            use_gated_attn=use_gated_attn,
            mean_attn_alpha=mean_attn_alpha,
        )
        self.sample_pool = AttentionPooling(input_dim, attn_hidden, gated=use_gated_attn)
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _pad_clusters(cluster_embs, cell_weights, cell_ids):
        batch_size = len(cluster_embs)
        max_clusters = max(x.shape[0] for x in cluster_embs)
        feat_dim = cluster_embs[0].shape[1]
        max_cells = max(w.shape[1] for w in cell_weights)

        emb_tensor = cluster_embs[0].new_zeros((batch_size, max_clusters, feat_dim))
        weight_tensor = cell_weights[0].new_zeros((batch_size, max_clusters, max_cells))
        id_tensor = cell_ids[0].new_full((batch_size, max_clusters, max_cells), -1)
        cluster_mask = torch.zeros((batch_size, max_clusters), dtype=torch.bool, device=emb_tensor.device)

        for i in range(batch_size):
            n = cluster_embs[i].shape[0]
            m = cell_weights[i].shape[1]
            emb_tensor[i, :n] = cluster_embs[i]
            weight_tensor[i, :n, :m] = cell_weights[i]
            id_tensor[i, :n, :m] = cell_ids[i]
            cluster_mask[i, :n] = True

        return emb_tensor, weight_tensor, id_tensor, cluster_mask

    @staticmethod
    def _flatten_weights(cluster_weights, cell_weights, cell_ids):
        all_cell_ids = []
        all_cell_weights = []
        all_cluster_weights = []
        for b in range(cluster_weights.shape[0]):
            ids = cell_ids[b]
            mask = ids != -1
            if not mask.any():
                continue
            cell_w = cell_weights[b][mask]
            cluster_idx = mask.nonzero(as_tuple=False)[:, 0]
            cluster_w = cluster_weights[b][cluster_idx]
            all_cell_ids.append(ids[mask])
            all_cell_weights.append(cell_w)
            all_cluster_weights.append(cluster_w)

        if not all_cell_ids:
            return (
                cell_weights.new_zeros((0,)),
                cell_weights.new_zeros((0,)),
                cell_ids.new_full((0,), -1),
            )

        cell_ids_flat = torch.cat(all_cell_ids, dim=0)
        cell_weights_flat = torch.cat(all_cell_weights, dim=0)
        cluster_weights_flat = torch.cat(all_cluster_weights, dim=0)
        return cell_weights_flat, cluster_weights_flat, cell_ids_flat

    def forward(self, expr, cell_ids, mask, cluster, return_debug: bool = False):
        # expr: [batch, max_cells, input_dim]
        cluster_embs = []
        cell_weights = []
        cluster_ids = []

        for i in range(expr.shape[0]):
            emb, weights, ids = self.cluster(expr[i], cell_ids[i], mask[i], cluster[i])
            cluster_embs.append(emb)
            cell_weights.append(weights)
            cluster_ids.append(ids)

        emb_tensor, weight_tensor, id_tensor, cluster_mask = self._pad_clusters(
            cluster_embs, cell_weights, cluster_ids
        )
        # emb_tensor: [batch, max_clusters, input_dim]
        # weight_tensor/id_tensor: [batch, max_clusters, max_cells]

        if return_debug:
            sample_emb, cluster_weights, cluster_scores = self.sample_pool(
                emb_tensor, cluster_mask, return_scores=True
            )
        else:
            sample_emb, cluster_weights = self.sample_pool(emb_tensor, cluster_mask)
            cluster_scores = None
        # sample_emb: [batch, input_dim], cluster_weights: [batch, max_clusters]
        pred = self.head(sample_emb)
        # pred: [batch, 1]

        cell_w, cluster_w, cell_ids_flat = self._flatten_weights(
            cluster_weights, weight_tensor, id_tensor
        )
        # cell_w/cluster_w/cell_ids_flat: [n_cells_total]

        if return_debug:
            debug = {
                "cluster_scores": cluster_scores,
                "cluster_mask": cluster_mask,
                "cluster_emb": emb_tensor,
            }
            return pred, sample_emb, cluster_weights, cell_w, cluster_w, cell_ids_flat, debug

        return pred, sample_emb, cluster_weights, cell_w, cluster_w, cell_ids_flat
