from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from .layers.torch_finetune import FinetuneModel


class CellFM(nn.Module):
    def __init__(self, n_genes: int, cfg):
        super().__init__()
        self.cfg = cfg
        self.net = FinetuneModel(n_genes, cfg)

    def forward(self, raw_nzdata, dw_nzdata, st_feat, nonz_gene, mask_gene, zero_idx):
        loss = self.net(raw_nzdata, dw_nzdata, st_feat, nonz_gene, mask_gene, zero_idx)
        emb, _ = self.net.encode(dw_nzdata, nonz_gene, st_feat, zero_idx)
        cls_token = emb[:, 0]
        return loss, cls_token


def _map_ms_key(ms_key: str) -> str:
    name = ms_key
    name = name.replace("layer_norm.gamma", "weight")
    name = name.replace("layer_norm.beta", "bias")
    name = name.replace("post_norm1.gamma", "post_norm1.weight")
    name = name.replace("post_norm1.beta", "post_norm1.bias")
    name = name.replace("post_norm2.gamma", "post_norm2.weight")
    name = name.replace("post_norm2.beta", "post_norm2.bias")
    return name


def load_mindspore_checkpoint(model: CellFM, ckpt_path: str) -> Tuple[list, list]:
    try:
        from mindspore.train.serialization import load_checkpoint
    except ImportError as exc:
        raise ImportError("mindspore is required to load a MindSpore checkpoint") from exc

    ms_ckpt = load_checkpoint(ckpt_path)
    torch_state_dict: Dict[str, torch.Tensor] = {}

    for ms_key, ms_param in ms_ckpt.items():
        pt_key = _map_ms_key(ms_key)
        if pt_key.startswith("moment1.") or pt_key.startswith("moment2."):
            continue
        if pt_key in {
            "global_step",
            "learning_rate",
            "beta1_power",
            "beta2_power",
            "current_iterator_step",
            "last_overflow_iterator_step",
        }:
            continue
        torch_state_dict[pt_key] = torch.tensor(ms_param.asnumpy())

    missing, unexpected = model.net.load_state_dict(torch_state_dict, strict=False)
    return missing, unexpected
