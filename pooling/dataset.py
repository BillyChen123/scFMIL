from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    """Per-sample dataset with variable-length cell dimension."""

    def __init__(
        self,
        exprs: Sequence[np.ndarray],
        labels: Sequence[float],
        cell_ids: Sequence[np.ndarray],
        sample_ids: Sequence[int],
        clusters: Sequence[np.ndarray],
    ) -> None:
        n = len(exprs)
        if not (len(labels) == len(cell_ids) == len(sample_ids) == len(clusters) == n):
            raise ValueError("exprs/labels/cell_ids/sample_ids/clusters must have the same length")

        self.exprs = list(exprs)
        self.labels = list(labels)
        self.cell_ids = list(cell_ids)
        self.sample_ids = list(sample_ids)
        self.clusters = list(clusters)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        expr_i = np.asarray(self.exprs[idx])
        cell_ids_i = np.asarray(self.cell_ids[idx])
        cluster_i = np.asarray(self.clusters[idx])

        n_cells = expr_i.shape[0]
        if n_cells == 0:
            raise ValueError(f"sample at index {idx} has zero cells")
        if cell_ids_i.shape[0] != n_cells or cluster_i.shape[0] != n_cells:
            raise ValueError(f"sample at index {idx} has inconsistent cell dimensions")

        return (
            torch.as_tensor(expr_i, dtype=torch.float32),
            torch.as_tensor(float(self.labels[idx]), dtype=torch.float32),
            torch.as_tensor(cell_ids_i, dtype=torch.long),
            torch.as_tensor(int(self.sample_ids[idx]), dtype=torch.long),
            torch.as_tensor(cluster_i, dtype=torch.long),
        )

    @staticmethod
    def _pad_1d(arrays: Sequence[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(x.shape[0] for x in arrays)
        out = []
        for arr in arrays:
            padded = torch.full((max_len,), pad_value, dtype=arr.dtype, device=arr.device)
            padded[: arr.shape[0]] = arr
            out.append(padded)
        return torch.stack(out, dim=0)

    @staticmethod
    def _pad_expr(exprs: Sequence[torch.Tensor]):
        max_cells = max(x.shape[0] for x in exprs)
        feat_dim = exprs[0].shape[1]

        expr_out = []
        mask_out = []
        for expr in exprs:
            n_cells = expr.shape[0]
            padded = torch.zeros((max_cells, feat_dim), dtype=expr.dtype, device=expr.device)
            padded[:n_cells] = expr
            expr_out.append(padded)

            mask = torch.zeros((max_cells,), dtype=torch.bool, device=expr.device)
            mask[:n_cells] = True
            mask_out.append(mask)

        return torch.stack(expr_out, dim=0), torch.stack(mask_out, dim=0)

    def collate_fn(self, batch):
        exprs = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch], dim=0)
        cell_ids = [item[2] for item in batch]
        sample_ids = torch.stack([item[3] for item in batch], dim=0)
        clusters = [item[4] for item in batch]

        expr_pad, mask_pad = self._pad_expr(exprs)
        cell_id_pad = self._pad_1d(cell_ids, pad_value=-1)
        cluster_pad = self._pad_1d(clusters, pad_value=-1)

        return expr_pad, labels, mask_pad, cell_id_pad, sample_ids, cluster_pad
