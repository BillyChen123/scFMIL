from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm

from .model import CellFM, load_mindspore_checkpoint
from .layers.utils import Config_80M, SCrna, Prepare, build_dataset


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_obs(adata) -> None:
    if "celltype" not in adata.obs:
        adata.obs["celltype"] = "unknown"
    if "batch_id" not in adata.obs:
        adata.obs["batch_id"] = 0
    if "train" not in adata.obs:
        adata.obs["train"] = 0


def split_val(adata, val_frac: float, seed: int) -> None:
    if val_frac <= 0:
        return
    rng = np.random.default_rng(seed)
    n_val = int(val_frac * adata.n_obs)
    idx = rng.choice(adata.n_obs, size=n_val, replace=False)
    adata.obs["train"] = 0
    adata.obs.iloc[idx, adata.obs.columns.get_loc("train")] = 1


def build_cfg(args) -> Config_80M:
    cfg = Config_80M()
    cfg.enc_nlayers = args.enc_layers
    cfg.pad_zero = True
    cfg.add_zero = True
    cfg.ecs = args.ecs
    cfg.ecs_threshold = args.ecs_threshold
    cfg.nonz_len = args.nonz_len
    cfg.mask_len = args.nonz_len
    return cfg


def train_one_epoch(model, loader, optimizer, scaler, device: str) -> float:
    model.train()
    running = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        raw_nzdata = batch["raw_nzdata"].to(device)
        dw_nzdata = batch["dw_nzdata"].to(device)
        st_feat = batch["ST_feat"].to(device)
        nonz_gene = batch["nonz_gene"].to(device)
        mask_gene = batch["mask_gene"].to(device)
        zero_idx = batch["zero_idx"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            loss, _ = model(raw_nzdata, dw_nzdata, st_feat, nonz_gene, mask_gene, zero_idx)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()
    return running / max(len(loader), 1)


def eval_one_epoch(model, loader, device: str) -> float:
    model.eval()
    running = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            raw_nzdata = batch["raw_nzdata"].to(device)
            dw_nzdata = batch["dw_nzdata"].to(device)
            st_feat = batch["ST_feat"].to(device)
            nonz_gene = batch["nonz_gene"].to(device)
            mask_gene = batch["mask_gene"].to(device)
            zero_idx = batch["zero_idx"].to(device)

            loss, _ = model(raw_nzdata, dw_nzdata, st_feat, nonz_gene, mask_gene, zero_idx)
            running += loss.item()
    return running / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", required=True)
    parser.add_argument("--out-dir", default="outputs/cellfm")
    parser.add_argument("--ms-ckpt", default=None)
    parser.add_argument("--pt-ckpt", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--val-frac", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enc-layers", type=int, default=2)
    parser.add_argument("--ecs", action="store_true")
    parser.add_argument("--ecs-threshold", type=float, default=0.8)
    parser.add_argument("--nonz-len", type=int, default=2048)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.adata)
    ensure_obs(adata)
    split_val(adata, args.val_frac, args.seed)

    cfg = build_cfg(args)
    model = CellFM(n_genes=27855, cfg=cfg).to(args.device)

    if args.ms_ckpt:
        missing, unexpected = load_mindspore_checkpoint(model, args.ms_ckpt)
        print(f"Loaded MS checkpoint. Missing: {len(missing)} Unexpected: {len(unexpected)}")

    if args.pt_ckpt:
        state = torch.load(args.pt_ckpt, map_location=args.device)
        model.load_state_dict(state, strict=False)

    prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=args.mask_ratio)
    train_dataset = SCrna(adata, mode="train")
    train_loader = build_dataset(
        train_dataset,
        prep=prep,
        batch_size=args.batch_size,
        pad_zero=cfg.pad_zero,
        drop=True,
        shuffle=True,
    )

    val_loader = None
    if args.val_frac > 0:
        val_dataset = SCrna(adata, mode="val")
        val_loader = build_dataset(
            val_dataset,
            prep=prep,
            batch_size=args.batch_size,
            pad_zero=cfg.pad_zero,
            drop=False,
            shuffle=False,
        )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, args.device)
        if val_loader is not None:
            val_loss = eval_one_epoch(model, val_loader, args.device)
            print(f"epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        else:
            print(f"epoch {epoch}: train_loss={train_loss:.6f}")

    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"Saved: {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
