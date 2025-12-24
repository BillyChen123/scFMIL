from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self, exprs, labels, cell_ids, sample_ids, clusters):
        self.exprs = exprs
        self.labels = labels
        self.cell_ids = cell_ids
        self.sample_ids = sample_ids
        self.clusters = clusters

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        expr = torch.tensor(self.exprs[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        cell_id = torch.tensor(self.cell_ids[idx], dtype=torch.long)
        sample_id = torch.tensor(self.sample_ids[idx], dtype=torch.long)
        cluster = torch.tensor(self.clusters[idx], dtype=torch.long)
        return expr, label, cell_id, sample_id, cluster

    @staticmethod
    def _pad_1d(arr_list, pad_value=-1):
        max_len = max(x.shape[0] for x in arr_list)
        padded = []
        for arr in arr_list:
            pad = torch.full((max_len,), pad_value, dtype=arr.dtype, device=arr.device)
            pad[: arr.shape[0]] = arr
            padded.append(pad)
        return torch.stack(padded, dim=0)

    @staticmethod
    def _pad_cells(expr_list, id_list):
        max_cells = max(x.shape[0] for x in expr_list)
        feat_dim = expr_list[0].shape[1]
        padded_expr = []
        padded_ids = []
        masks = []
        for expr, cid in zip(expr_list, id_list):
            n = expr.shape[0]
            expr_pad = torch.zeros((max_cells, feat_dim), dtype=expr.dtype, device=expr.device)
            expr_pad[:n] = expr
            padded_expr.append(expr_pad)

            id_pad = torch.full((max_cells,), -1, dtype=cid.dtype, device=cid.device)
            id_pad[:n] = cid
            padded_ids.append(id_pad)

            mask = torch.zeros((max_cells,), dtype=torch.bool, device=expr.device)
            mask[:n] = True
            masks.append(mask)

        return torch.stack(padded_expr, dim=0), torch.stack(padded_ids, dim=0), torch.stack(masks, dim=0)

    def collate_fn(self, batch):
        expr_list = [item[0] for item in batch]
        label = torch.stack([item[1] for item in batch], dim=0)
        cell_id_list = [item[2] for item in batch]
        sample_id = torch.stack([item[3] for item in batch], dim=0)
        cluster_list = [item[4] for item in batch]

        expr, cell_ids, mask = self._pad_cells(expr_list, cell_id_list)
        cluster = self._pad_1d(cluster_list, pad_value=-1)
        # expr: [batch, max_cells, feat_dim], mask/cell_ids/cluster: [batch, max_cells]

        return expr, label, mask, cell_ids, sample_id, cluster
