#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script for UniD Visual QA
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from utils import seed_everything, iou_xywh_pixel, giou_loss, zip_submission
from preprocess import find_jsons, UniDSet, collate_fn, Vocab
from model import CrossAttnVLM

# Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warning] wandb not installed. Run: pip install wandb")


def make_loader(json_dir: str, jpg_dir: str, vocab: Vocab = None, build_vocab: bool = False,
                batch_size: int = CFG.BATCH_SIZE, img_size: int = CFG.IMG_SIZE,
                num_workers: int = CFG.NUM_WORKERS, shuffle: bool = False):
    """Create data loader from JSON/JPG directories"""
    json_files = find_jsons(json_dir)
    ds = UniDSet(json_files, jpg_dir=jpg_dir, vocab=vocab, build_vocab=build_vocab,
                 resize_to=(img_size, img_size))
    if build_vocab:
        # use only supervised samples (have bbox)
        sup_idx = [i for i in range(len(ds)) if ds.items[i]["bbox"] is not None]
        if len(sup_idx) == 0:
            raise RuntimeError("No supervised samples (no bboxes) in given json_dir.")
        ds = torch.utils.data.Subset(ds, sup_idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                    num_workers=num_workers, collate_fn=collate_fn)
    return ds, dl


def predict_test(model, test_json_dir, test_jpg_dir, vocab, device, batch_size, img_size, num_workers, epoch, run_ckpt_dir):
    """Test 세트 예측 → CSV + ZIP 생성"""
    model.eval()

    # Load test data
    json_files = find_jsons(test_json_dir)
    test_ds = UniDSet(json_files, jpg_dir=test_jpg_dir, vocab=vocab, build_vocab=False,
                      resize_to=(img_size, img_size))
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_fn)

    rows = []
    with torch.no_grad():
        pbar = tqdm(test_dl, desc=f"Test Prediction (Epoch {epoch})", ncols=100)
        for imgs, ids, lens, targets, meta in pbar:
            imgs = imgs.to(device)
            ids = ids.to(device)
            lens = lens.to(device)
            pred = model(imgs, ids, lens)  # (B,4) normalized

            for i in range(imgs.size(0)):
                W, H = meta[i]["orig_size"]
                cx, cy, nw, nh = [float(v) for v in pred[i].cpu().numpy().tolist()]
                x = (cx - nw / 2.0) * W
                y = (cy - nh / 2.0) * H
                w = nw * W
                h = nh * H

                rows.append({
                    "query_id": meta[i]["query_id"],
                    "query_text": meta[i]["query_text"],
                    "pred_x": x, "pred_y": y, "pred_w": w, "pred_h": h
                })
        pbar.close()

    # Save CSV
    csv_path = os.path.join(run_ckpt_dir, f"epoch{epoch}_test_pred.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(rows, columns=["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Create ZIP
    zip_path = os.path.join(run_ckpt_dir, f"epoch{epoch}_submission.zip")
    zip_submission(csv_path, zip_path)

    print(f"  📦 [Test] Saved: {os.path.basename(csv_path)} & {os.path.basename(zip_path)}")
    return csv_path, zip_path


def validate(model, val_dl, device):
    """Run validation and compute mIoU and loss"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    ious = []

    with torch.no_grad():
        pbar = tqdm(val_dl, desc="Validation", ncols=100)
        for imgs, ids, lens, targets, meta in pbar:
            imgs = imgs.to(device)
            ids = ids.to(device)
            lens = lens.to(device)

            pred = model(imgs, ids, lens)  # (B,4) normalized

            # ---- 1) 유효한 타겟 인덱스만 골라서 loss 계산 ----
            valid_idx = [i for i, t in enumerate(targets) if t is not None]
            if valid_idx:
                t = torch.stack([targets[i] for i in valid_idx], dim=0).to(device)
                pred_valid = pred[valid_idx]

                if CFG.LOSS_TYPE == "smooth_l1":
                    loss = F.smooth_l1_loss(pred_valid, t, reduction="mean")
                elif CFG.LOSS_TYPE == "giou":
                    loss = giou_loss(pred_valid, t)
                else:
                    raise ValueError(f"Unknown loss type: {CFG.LOSS_TYPE}")

                total_loss += float(loss.item()) * len(valid_idx)
                total_samples += len(valid_idx)

            # ---- 2) mIoU 계산 (픽셀 좌표로 변환) ----
            B = imgs.size(0)
            for i in range(B):
                if targets[i] is None:
                    continue

                W, H = meta[i]["orig_size"]

                # pred bbox (normalized [cx,cy,w,h] -> pixel [x,y,w,h])
                cx, cy, nw, nh = [float(v) for v in pred[i].cpu().numpy().tolist()]
                px = (cx - nw / 2.0) * W
                py = (cy - nh / 2.0) * H
                pw = nw * W
                ph = nh * H

                # gt bbox (normalized [cx,cy,w,h] -> pixel [x,y,w,h])
                gt = [float(v) for v in targets[i].numpy().tolist()]
                gx = (gt[0] - gt[2] / 2.0) * W
                gy = (gt[1] - gt[3] / 2.0) * H
                gw = gt[2] * W
                gh = gt[3] * H

                iou = iou_xywh_pixel([px, py, pw, ph], [gx, gy, gw, gh])
                ious.append(iou)

            current_miou = float(np.mean(ious)) if ious else 0.0
            pbar.set_postfix({'mIoU': f'{current_miou:.4f}'})

        pbar.close()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_iou = float(np.mean(ious)) if ious else 0.0

    return avg_loss, avg_iou



def train_loop(args):
    """Main training loop"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    
    # ✅ run_name 기반 checkpoint 디렉토리
    base_ckpt_dir = os.path.dirname(args.save_ckpt)            # 예: ./checkpoints
    run_name = args.wandb_run_name or "default_run"
    run_ckpt_dir = os.path.join(base_ckpt_dir, run_name)       # 예: ./checkpoints/test-epoch1
    os.makedirs(run_ckpt_dir, exist_ok=True)

    # Initialize WandB
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "dim": args.dim,
                "no_pretrain": args.no_pretrain,
            }
        )
        print(f"[WandB] Initialized: {args.wandb_entity}/{args.wandb_project}/{args.wandb_run_name}")

    # Build dataset + vocab on train jsons
    print(f"[Loading] Train data from {args.json_dir}")
    train_ds, train_dl = make_loader(args.json_dir, args.jpg_dir, vocab=None, build_vocab=True,
                                     batch_size=args.batch_size, img_size=args.img_size,
                                     num_workers=args.num_workers, shuffle=True)

    # Resolve vocab (Subset wrapper case)
    vocab = train_ds.dataset.vocab if isinstance(train_ds, torch.utils.data.Subset) else train_ds.vocab
    vocab_size = len(vocab.itos)
    print(f"[Vocab] Built vocabulary with {vocab_size} tokens")

    # Load validation data if provided
    val_dl = None
    if args.val_json_dir and args.val_jpg_dir:
        print(f"[Loading] Validation data from {args.val_json_dir}")
        val_ds, val_dl = make_loader(args.val_json_dir, args.val_jpg_dir, vocab=vocab, build_vocab=False,
                                     batch_size=args.batch_size, img_size=args.img_size,
                                     num_workers=args.num_workers, shuffle=False)
        print(f"[Validation] Loaded {len(val_ds)} validation samples")

    # Create model
    model = CrossAttnVLM(vocab_size=vocab_size,
                         dim=args.dim,
                         pretrained_backbone=not args.no_pretrain,
                         img_size=args.img_size,
                         backbone=args.backbone,).to(device)
    print(f"[Model] CrossAttnVLM with dim={args.dim}, pretrained={not args.no_pretrain}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # Resume from checkpoint if specified
    start_epoch = 1
    global_step = 0
    best_miou = 0.0

    if args.resume:
        print(f"[Resume] Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt["epoch"] + 1
        # Optionally restore optimizer/scheduler state (not implemented for simplicity)
        print(f"[Resume] Resuming from epoch {start_epoch}")

    total_samples = len(train_ds)
    print(f"[Training] Starting training for {args.epochs} epochs on {total_samples} samples")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0

        # Training loop with tqdm progress bar
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for batch_idx, (imgs, ids, lens, targets, meta) in enumerate(pbar):
            global_step += 1
            imgs = imgs.to(device)
            ids = ids.to(device)
            lens = lens.to(device)
            t = torch.stack([tar for tar in targets if tar is not None], dim=0).to(device)  # (B,4)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                pred = model(imgs, ids, lens)   # (B,4) normalized

                if CFG.LOSS_TYPE == "smooth_l1":
                    loss = F.smooth_l1_loss(pred, t, reduction="mean")
                elif CFG.LOSS_TYPE == "giou":
                    loss = giou_loss(pred, t)
                else:
                    raise ValueError(f"Unknown loss type: {CFG.LOSS_TYPE}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item()) * imgs.size(0)
            current_loss = float(loss.item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })

            # Log to WandB every iteration
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "train/loss": current_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                }, step=global_step)

        pbar.close()

        scheduler.step()
        avg_train_loss = running / total_samples

        # Test 예측 제거 (시간 절약 - test.py로 나중에 실행)
        # if args.test_json_dir and args.test_jpg_dir:
        #     csv_path, zip_path = predict_test(...)

        # Validation (optional - 있으면 실행)
        if val_dl is not None:
            val_loss, val_miou = validate(model, val_dl, device)

            # Log validation metrics to WandB
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "val/loss": val_loss,
                    "val/mIoU": val_miou,
                    "train/epoch_loss": avg_train_loss,
                }, step=global_step)

            print(f"[Epoch {epoch}/{args.epochs}] "
                  f"train_loss={avg_train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_mIoU={val_miou:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

            # Save best checkpoint based on mIoU
            if val_miou > best_miou:
                best_miou = val_miou
                # Create checkpoint name with epoch and mIoU
                ckpt_name = f"epoch{epoch}_miou{val_miou:.4f}.pth"
                best_ckpt_path = os.path.join(run_ckpt_dir, ckpt_name)

                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "vocab_itos": vocab.itos,
                    "dim": args.dim,
                    "no_pretrain": args.no_pretrain,
                    "img_size": args.img_size,
                    "backbone": args.backbone,
                    "val_miou": val_miou,
                    "val_loss": val_loss,
                }, best_ckpt_path)
                print(f"  ✅ [Best] Saved checkpoint: {best_ckpt_path}")
        else:
            # No validation - save checkpoint every epoch
            # Log epoch metrics to WandB
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "train/epoch_loss": avg_train_loss,
                }, step=global_step)

            print(f"[Epoch {epoch}/{args.epochs}] "
                  f"train_loss={avg_train_loss:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

            # Save checkpoint for every epoch (no validation, so save all)
            ckpt_name = f"{args.backbone}_epoch{epoch}.pth"
            ckpt_path = os.path.join(run_ckpt_dir, ckpt_name)

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "vocab_itos": vocab.itos,
                "dim": args.dim,
                "no_pretrain": args.no_pretrain,
                "img_size": args.img_size,
                "backbone": args.backbone,
                "train_loss": avg_train_loss,
            }, ckpt_path)
            print(f"  ✅ [Saved] Checkpoint: {ckpt_path}")

        print("-" * 80)

    # Save final checkpoint
    final_path = os.path.join(run_ckpt_dir, "final.pth")
    torch.save({
        "epoch": args.epochs,
        "model_state": model.state_dict(),
        "vocab_itos": vocab.itos,
        "dim": args.dim,
        "no_pretrain": args.no_pretrain,
        "img_size": args.img_size,
        }, final_path)
    print("=" * 80)
    print(f"[Saved] Final checkpoint saved to {final_path}")
    if val_dl is not None:
        print(f"[Best] Best validation mIoU: {best_miou:.4f}")

    # Finish WandB run
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("[WandB] Run finished")


def get_args():
    """Parse command line arguments"""
    ap = argparse.ArgumentParser(description="Train UniD Visual QA Model")
    ap.add_argument("--json_dir", type=str, default=CFG.TRAIN_JSON_DIR, help="Directory with training JSON files")
    ap.add_argument("--jpg_dir", type=str, default=CFG.TRAIN_JPG_DIR, help="Directory with training JPG images")
    ap.add_argument("--val_json_dir", type=str, default=CFG.VAL_JSON_DIR, help="Directory with validation JSON files")
    ap.add_argument("--val_jpg_dir", type=str, default=CFG.VAL_JPG_DIR, help="Directory with validation JPG images")
    ap.add_argument("--test_json_dir", type=str, default=CFG.TEST_JSON_DIR, help="Directory with test JSON files (for submission)")
    ap.add_argument("--test_jpg_dir", type=str, default=CFG.TEST_JPG_DIR, help="Directory with test JPG images")
    ap.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE, help="Batch size")
    ap.add_argument("--img_size", type=int, default=CFG.IMG_SIZE, help="Image size")
    ap.add_argument("--dim", type=int, default=CFG.DIM, help="Hidden dimension")
    ap.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS, help="DataLoader workers")
    ap.add_argument("--epochs", type=int, default=CFG.EPOCHS, help="Number of epochs")
    ap.add_argument("--lr", type=str, default=CFG.LEARNING_RATE, help="Learning rate")
    ap.add_argument("--no_pretrain", action="store_true", help="Disable ImageNet pretrained weights")
    ap.add_argument("--save_ckpt", type=str, default=CFG.CKPT_PATH, help="Path to save checkpoint")
    ap.add_argument("--backbone", type=str, default=CFG.BACKBONE,
                    choices=["resnet18", "resnet34", "resnet50", "efficientnet_b3", "efficientnet_v2_s", "clip_vit_b16"],
                    help="CNN backbone for image encoder")

    # WandB arguments
    ap.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", type=str, default="unid", help="WandB project name")
    ap.add_argument("--wandb_entity", type=str, default="slpr", help="WandB entity/team name")
    ap.add_argument("--wandb_run_name", type=str, default="baseline-cross-attn", help="WandB run name")

    # Resume training
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    return ap.parse_args()


def main():
    seed_everything(CFG.SEED)
    args = get_args()
    train_loop(args)


if __name__ == "__main__":
    main()
