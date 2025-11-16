#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Functions
"""

import random
import zipfile
import os
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Fix random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def iou_xywh_pixel(pred_xywh, gt_xywh):
    """
    Calculate IoU between two bounding boxes in pixel coordinates [x, y, w, h]
    """
    px, py, pw, ph = pred_xywh
    gx, gy, gw, gh = gt_xywh
    px2, py2 = px + pw, py + ph
    gx2, gy2 = gx + gw, gy + gh
    ix1, iy1 = max(px, gx), max(py, gy)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = pw * ph + gw * gh - inter if (pw * ph + gw * gh - inter) > 0 else 1e-6
    return inter / union

def xywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    """
    [cx, cy, w, h] (정규화든 픽셀이든 상관 없음) -> [x1, y1, x2, y2]
    """
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=-1)


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target: (B, 4) [cx, cy, w, h]
    """
    eps = 1e-6

    pred_xyxy = xywh_to_xyxy(pred)
    target_xyxy = xywh_to_xyxy(target)

    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    gx1, gy1, gx2, gy2 = target_xyxy.unbind(-1)

    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)

    inter_w = (ix2 - ix1).clamp(min=0)
    inter_h = (iy2 - iy1).clamp(min=0)
    inter = inter_w * inter_h

    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)

    union = (area_p + area_t - inter).clamp(min=eps)
    iou = inter / union

    cx1 = torch.min(px1, gx1)
    cy1 = torch.min(py1, gy1)
    cx2 = torch.max(px2, gx2)
    cy2 = torch.max(py2, gy2)
    c_w = (cx2 - cx1).clamp(min=0)
    c_h = (cy2 - cy1).clamp(min=0)
    c_area = (c_w * c_h).clamp(min=eps)

    giou = iou - (c_area - union) / c_area
    loss = 1.0 - giou
    return loss.mean()


def zip_submission(csv_path: str, zip_path: str):
    """Create submission zip file"""
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        arcname = os.path.basename(csv_path)
        zf.write(csv_path, arcname=arcname)
    print(f"[Submission] Zipped {csv_path} → {zip_path}")
