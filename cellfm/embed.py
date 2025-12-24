from __future__ import annotations

import argparse
from pathlib import Path

import scanpy as sc
import torch
from tqdm import tqdm

from .model import CellFM, load_mindspore_checkpoint
from .layers.utils import Config_80M, SCrna, Prepare, build_dataset


def ensure_obs(adata) -> None:
    if "celltype" not in adata.obs:
        adata.obs["celltype"] = "unknown"
    if "batch_id" not in adata.obs:
        adata.obs["batch_id"] = 0
    if "train" not in adata.obs:
        adata.obs["train"] = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", required=True)
    parser.add_argument("--out-adata", required=True)
    parser.add_argument("--pt-ckpt", default=None)
    parser.add_argument("--ms-ckpt", default=None)
    parser.add_argument("--emb-key", default="X_cellfm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    adata = sc.read_h5ad(args.adata)
    ensure_obs(adata)

    cfg = Config_80M()
    cfg.pad_zero = True
    cfg.add_zero = True
    cfg.ecs = False

    model = CellFM(n_genes=27855, cfg=cfg).to(args.device)
    if args.ms_ckpt:
        load_mindspore_checkpoint(model, args.ms_ckpt)
    if args.pt_ckpt:
        state = torch.load(args.pt_ckpt, map_location=args.device)
        model.load_state_dict(state, strict=False)

    prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=0.0)
    dataset = SCrna(adata, mode="train")
    loader = build_dataset(
        dataset,
        prep=prep,
        batch_size=args.batch_size,
        pad_zero=cfg.pad_zero,
        drop=False,
        shuffle=False,
    )

    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="embed", leave=False):
            raw_nzdata = batch["raw_nzdata"].to(args.device)
            dw_nzdata = batch["dw_nzdata"].to(args.device)
            st_feat = batch["ST_feat"].to(args.device)
            nonz_gene = batch["nonz_gene"].to(args.device)
            mask_gene = batch["mask_gene"].to(args.device)
            zero_idx = batch["zero_idx"].to(args.device)

            _, cls_token = model(raw_nzdata, dw_nzdata, st_feat, nonz_gene, mask_gene, zero_idx)
            embeddings.append(cls_token.cpu())

    emb = torch.cat(embeddings, dim=0).numpy()
    out_adata = dataset.adata
    out_adata.obsm[args.emb_key] = emb

    out_path = Path(args.out_adata)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sc.write(out_path, out_adata)
    print(f"Saved embeddings to {out_path}")


if __name__ == "__main__":
    main()
