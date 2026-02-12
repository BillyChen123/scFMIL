from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class AttentionPoolingLegacy(nn.Module):
    """Legacy attention: a single linear layer over input features."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(input_dim, 1, bias=False))

    def forward(self, x: torch.Tensor, idx: torch.Tensor, return_scores: bool = False):
        # x: [B, N, D], idx: [B, N] with -1 as padding
        mask = idx != -1
        attn_scores = self.attention(x).squeeze(-1)  # [B, N]
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)
        if return_scores:
            return pooled, attn_weights, attn_scores
        return pooled, attn_weights


class AttentionPooling(nn.Module):
    """Current attention: tanh(hidden) + context vector."""

    def __init__(self, input_dim: int, hidden_dim_attn: int = 64):
        super().__init__()
        self.attention_fc = nn.Linear(input_dim, hidden_dim_attn)
        self.context_vector_fc = nn.Linear(hidden_dim_attn, 1)

    def forward(self, x: torch.Tensor, idx: torch.Tensor, return_scores: bool = False):
        # x: [B, N, D], idx: [B, N] with -1 as padding
        mask = idx != -1
        h = torch.tanh(self.attention_fc(x))
        attn_scores = self.context_vector_fc(h).squeeze(-1)  # [B, N]
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)
        if return_scores:
            return pooled, attn_weights, attn_scores
        return pooled, attn_weights


class MeanPooling(nn.Module):
    def forward(self, x: torch.Tensor, idx: torch.Tensor):
        mask = idx != -1
        mask = mask.unsqueeze(-1)
        x_masked = x * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = x_masked.sum(dim=1) / denom
        weights = mask.squeeze(-1).float() / denom.squeeze(-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return pooled, weights


class AggCluster(nn.Module):
    def __init__(self, input_dim: int, attn_hidden: int, legacy_attn: bool = False):
        super().__init__()
        self.fc_att2 = (
            AttentionPoolingLegacy(input_dim)
            if legacy_attn
            else AttentionPooling(input_dim, attn_hidden)
        )

    def agg_cluster(self, expr_i, idx_i, mask_i, cluster_i):
        # expr_i: [max_cell, D], idx_i/mask_i/cluster_i: [max_cell]
        unmask = mask_i == 1
        expr_i = expr_i[unmask]
        idx_i = idx_i[unmask]
        cluster_i = cluster_i[unmask]

        unique_clusters = cluster_i[cluster_i >= 0].unique()
        n_clusters = unique_clusters.numel()
        unique_clusters = torch.arange(n_clusters, device=expr_i.device)

        feat_dim = expr_i.shape[1]
        max_cells = 0
        for cluster_id in unique_clusters:
            max_cells = max(max_cells, int((cluster_i == cluster_id).sum().item()))

        agg_expr = torch.full(
            (n_clusters, max_cells, feat_dim),
            fill_value=0.0,
            device=expr_i.device,
            dtype=expr_i.dtype,
        )
        agg_idx = torch.full(
            (n_clusters, max_cells),
            fill_value=-1,
            device=idx_i.device,
            dtype=idx_i.dtype,
        )
        agg_cluster = torch.full(
            (n_clusters, max_cells),
            fill_value=-1,
            device=idx_i.device,
            dtype=idx_i.dtype,
        )

        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = cluster_i == cluster_id
            expr_in_cluster = expr_i[cluster_mask]
            idx_in_cluster = idx_i[cluster_mask]
            n = expr_in_cluster.shape[0]
            agg_expr[i, :n] = expr_in_cluster
            agg_idx[i, :n] = idx_in_cluster
            agg_cluster[i, :n] = cluster_id

        return agg_expr, agg_idx, agg_cluster

    def forward(self, expr, idx, mask, cluster):
        # expr: [B, max_cell, D]
        agg_res, agg_weight, agg_idx, agg_cluster = [], [], [], []
        for i in range(expr.shape[0]):
            agg_expr_i, agg_idx_i, agg_cluster_i = self.agg_cluster(
                expr[i], idx[i], mask[i], cluster[i]
            )
            agg_expr_i, agg_weight_i = self.fc_att2(agg_expr_i, agg_idx_i)
            agg_res.append(agg_expr_i)
            agg_weight.append(agg_weight_i)
            agg_idx.append(agg_idx_i)
            agg_cluster.append(agg_cluster_i)
        return agg_res, agg_weight, agg_idx, agg_cluster


class WeightModel(nn.Module):
    def __init__(self, input_dim: int, attn_hidden: int = 64, legacy_attn: bool = False):
        super().__init__()
        self.agg = AggCluster(input_dim, attn_hidden, legacy_attn=legacy_attn)
        self.fc_att = (
            AttentionPoolingLegacy(input_dim)
            if legacy_attn
            else AttentionPooling(input_dim, attn_hidden)
        )
        self.fc_out = nn.Identity()

    @staticmethod
    def ls_to_tensor(agg_res, agg_weight, agg_idx, agg_cluster):
        bsz = len(agg_res)
        n_clusters = max(x.shape[0] for x in agg_res)
        feat_dim = agg_res[0].shape[1]
        max_cells = max(w.shape[1] for w in agg_weight)

        agg_res_ts = torch.full(
            (bsz, n_clusters, feat_dim),
            fill_value=0.0,
            device=agg_res[0].device,
            dtype=agg_res[0].dtype,
        )
        agg_weight_ts = torch.full(
            (bsz, n_clusters, max_cells),
            fill_value=0.0,
            device=agg_weight[0].device,
            dtype=agg_weight[0].dtype,
        )
        agg_idx_ts = torch.full(
            (bsz, n_clusters, max_cells),
            fill_value=-1,
            device=agg_idx[0].device,
            dtype=agg_idx[0].dtype,
        )
        agg_cluster_ts = torch.full(
            (bsz, n_clusters, max_cells),
            fill_value=-1,
            device=agg_idx[0].device,
            dtype=agg_idx[0].dtype,
        )
        agg_mask_ts = torch.full(
            (bsz, n_clusters),
            fill_value=0,
            device=agg_idx[0].device,
            dtype=agg_idx[0].dtype,
        )

        for i in range(bsz):
            nn, mm = agg_weight[i].shape
            agg_res_ts[i, :nn] = agg_res[i]
            agg_weight_ts[i, :nn, :mm] = agg_weight[i]
            agg_idx_ts[i, :nn, :mm] = agg_idx[i]
            agg_cluster_ts[i, :nn, :mm] = agg_cluster[i]
            agg_mask_ts[i, :nn] = torch.where((agg_idx[i] == -1).all(dim=1), -1, 0)

        return agg_res_ts, agg_weight_ts, agg_idx_ts, agg_cluster_ts, agg_mask_ts

    @staticmethod
    def process_weights_to_n_cell(sample_weight, agg_weight, agg_idx, agg_cluster):
        all_cell_weight = []
        all_cluster_weight = []
        all_cell_idx = []
        for b in range(sample_weight.shape[0]):
            batch_agg_weight = agg_weight[b]
            batch_agg_idx = agg_idx[b]
            batch_cluster = agg_cluster[b]
            batch_sample_weight = sample_weight[b]

            valid_mask = batch_agg_idx != -1
            valid_agg_weight = batch_agg_weight[valid_mask]
            valid_agg_idx = batch_agg_idx[valid_mask]
            valid_batch_cluster = batch_cluster[valid_mask]

            cell_cluster_weights = torch.gather(batch_sample_weight, 0, valid_batch_cluster)
            all_cell_weight.append(valid_agg_weight)
            all_cluster_weight.append(cell_cluster_weights)
            all_cell_idx.append(valid_agg_idx)

        cell_weight = torch.cat(all_cell_weight, dim=0)
        cluster_weight = torch.cat(all_cluster_weight, dim=0)
        idx = torch.cat(all_cell_idx, dim=0)
        return cell_weight, cluster_weight, idx

    def forward(self, expr, idx, mask, cluster):
        agg_expr, agg_weight, agg_idx, agg_cluster = self.agg(expr, idx, mask, cluster)
        agg_expr, agg_weight, agg_idx, agg_cluster, agg_mask = self.ls_to_tensor(
            agg_expr, agg_weight, agg_idx, agg_cluster
        )
        _, sample_weight = self.fc_att(agg_expr, agg_mask)
        agg_weight, sample_weight, valid_idx = self.process_weights_to_n_cell(
            sample_weight, agg_weight, agg_idx, agg_cluster
        )
        return sample_weight, agg_weight, valid_idx


class LegacyHierarchicalPooling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        attn_hidden: int = 64,
        dropout: float = 0.2,
        legacy_attn: bool = True,
    ):
        super().__init__()
        self.agg = AggCluster(input_dim, attn_hidden, legacy_attn=legacy_attn)
        self.fc_att = (
            AttentionPoolingLegacy(input_dim)
            if legacy_attn
            else AttentionPooling(input_dim, attn_hidden)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    @staticmethod
    def ls_to_tensor(agg_res, agg_weight, agg_idx, agg_cluster):
        return WeightModel.ls_to_tensor(agg_res, agg_weight, agg_idx, agg_cluster)

    @staticmethod
    def process_weights_to_n_cell(sample_weight, agg_weight, agg_idx, agg_cluster):
        return WeightModel.process_weights_to_n_cell(sample_weight, agg_weight, agg_idx, agg_cluster)

    def forward(self, expr, idx, mask, cluster, return_debug: bool = False):
        agg_expr, agg_weight, agg_idx, agg_cluster = self.agg(expr, idx, mask, cluster)
        agg_expr, agg_weight, agg_idx, agg_cluster, agg_mask = self.ls_to_tensor(
            agg_expr, agg_weight, agg_idx, agg_cluster
        )

        if return_debug:
            sample_emb, cluster_weights, cluster_scores = self.fc_att(
                agg_expr, agg_mask, return_scores=True
            )
        else:
            sample_emb, cluster_weights = self.fc_att(agg_expr, agg_mask)
            cluster_scores = None

        pred = self.fc_out(sample_emb)
        cell_w, cluster_w, cell_ids_flat = self.process_weights_to_n_cell(
            cluster_weights, agg_weight, agg_idx, agg_cluster
        )

        if return_debug:
            debug = {
                "cluster_scores": cluster_scores,
                "cluster_mask": agg_mask != -1,
                "cluster_emb": agg_expr,
            }
            return pred, sample_emb, cluster_weights, cell_w, cluster_w, cell_ids_flat, debug

        return pred, sample_emb, cluster_weights, cell_w, cluster_w, cell_ids_flat


@dataclass
class ModelSpec:
    input_dim: int
    attn_hidden: int
    legacy_attn: bool


def detect_model_spec(state_dict: dict, input_dim: int, attn_hidden: int | None = None) -> ModelSpec:
    if any("attention_fc" in k for k in state_dict):
        legacy_attn = False
        if attn_hidden is None:
            attn_hidden = state_dict["fc_att.attention_fc.weight"].shape[0]
    else:
        legacy_attn = True
        if attn_hidden is None:
            attn_hidden = 64
    return ModelSpec(input_dim=input_dim, attn_hidden=attn_hidden, legacy_attn=legacy_attn)
