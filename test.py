#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation and Prediction Script for UniD Visual QA
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import CFG
from utils import seed_everything, iou_xywh_pixel, zip_submission
from preprocess import find_jsons, UniDSet, collate_fn, Vocab
from model import CrossAttnVLM


def load_model_from_ckpt(ckpt_path: str, device: torch.device, backbone_override: str = None):
    """Load model from checkpoint"""
    print(f"[Loading] Checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Reconstruct vocab
    vocab = Vocab()
    vocab.itos = ckpt["vocab_itos"]
    vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}

    # Reconstruct model
    backbone = backbone_override if backbone_override else ckpt.get("backbone", "resnet34")
    model = CrossAttnVLM(
        vocab_size=len(vocab.itos),
        dim=ckpt["dim"],
        pretrained_backbone=not ckpt.get("no_pretrain", False),
        img_size=ckpt.get("img_size", CFG.IMG_SIZE),
        backbone=backbone
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img_size = ckpt.get("img_size", CFG.IMG_SIZE)
    print(f"[Model] Loaded with vocab_size={len(vocab.itos)}, dim={ckpt['dim']}, img_size={img_size}, backbone={backbone}")
    return model, vocab, img_size


def evaluate_loop(args):
    """Evaluate model on validation set with ground truth"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    model, vocab, img_size = load_model_from_ckpt(args.ckpt, device, args.backbone)

    print(f"[Loading] Evaluation data from {args.json_dir}")
    json_files = find_jsons(args.json_dir)
    ds = UniDSet(json_files, jpg_dir=args.jpg_dir, vocab=vocab, build_vocab=False,
                 resize_to=(img_size, img_size))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_fn)

    rows = []
    ious = []
    print(f"[Evaluating] {len(ds)} samples")
    print("=" * 80)

    with torch.no_grad():
        for batch_idx, (imgs, ids, lens, targets, meta) in enumerate(dl):
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

                if targets[i] is not None:
                    gt = [float(v) for v in targets[i].numpy().tolist()]
                    gx = (gt[0] - gt[2] / 2.0) * W
                    gy = (gt[1] - gt[3] / 2.0) * H
                    gw = gt[2] * W
                    gh = gt[3] * H
                    iou = iou_xywh_pixel([x, y, w, h], [gx, gy, gw, gh])
                    ious.append(iou)

            if (batch_idx + 1) % max(1, len(dl) // 10) == 0:
                current_miou = float(np.mean(ious)) if ious else 0.0
                print(f"  Batch [{batch_idx + 1}/{len(dl)}] mIoU: {current_miou:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = pd.DataFrame(rows, columns=["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"])
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"[Saved] Predictions saved to {args.out_csv}")
    if ious:
        final_miou = float(np.mean(ious))
        print(f"[Eval] mIoU = {final_miou:.4f}")
    else:
        print("[Eval] No GT found; mIoU not computed.")


def predict_loop(args):
    """Predict on test set without ground truth"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    model, vocab, img_size = load_model_from_ckpt(args.ckpt, device, args.backbone)

    print(f"[Loading] Test data from {args.json_dir}")
    json_files = find_jsons(args.json_dir)
    ds = UniDSet(json_files, jpg_dir=args.jpg_dir, vocab=vocab, build_vocab=False,
                 resize_to=(img_size, img_size))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_fn)

    rows = []
    print(f"[Predicting] {len(ds)} samples")
    print("=" * 80)

    with torch.no_grad():
        for batch_idx, (imgs, ids, lens, targets, meta) in enumerate(dl):
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

            if (batch_idx + 1) % max(1, len(dl) // 10) == 0:
                print(f"  Batch [{batch_idx + 1}/{len(dl)}] processed", flush=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = pd.DataFrame(rows, columns=["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"])
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"[Saved] Predictions saved to {args.out_csv}")


def get_args():
    """Parse command line arguments"""
    ap = argparse.ArgumentParser(description="Evaluate/Predict UniD Visual QA Model")
    sub = ap.add_subparsers(dest="mode", required=True)

    # Shared arguments function
    def add_common(p):
        p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
        p.add_argument("--json_dir", type=str, required=True, help="Directory with JSON files")
        p.add_argument("--jpg_dir", type=str, required=True, help="Directory with JPG images")
        p.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE, help="Batch size")
        p.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS, help="DataLoader workers")
        p.add_argument("--out_csv", type=str, required=True, help="Output CSV path")
        p.add_argument("--backbone", type=str, default=None, help="Override backbone (resnet18/resnet34/resnet50/efficientnet_b0/clip_vit_b16)")

    # eval subcommand
    p_eval = sub.add_parser("eval", help="Evaluate on validation set with ground truth")
    add_common(p_eval)

    # predict subcommand
    p_pred = sub.add_parser("predict", help="Predict on test set without ground truth")
    add_common(p_pred)

    return ap.parse_args()


def main():
    seed_everything(CFG.SEED)
    args = get_args()

    if args.mode == "eval":
        evaluate_loop(args)
    elif args.mode == "predict":
        predict_loop(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
