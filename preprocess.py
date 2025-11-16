#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing and Dataset for UniD Visual QA
"""

import os
import json
from glob import glob
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

from config import CFG

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# torchvision transforms (optional)
try:
    from torchvision import transforms as T
    _TF_OK = True
except Exception:
    _TF_OK = False


def find_jsons(json_dir: str) -> List[str]:
    """Find all JSON files in directory (including subdirectories)"""
    if os.path.isdir(json_dir):
        # Search recursively for all .json files
        json_files = sorted(glob(os.path.join(json_dir, "**", "*.json"), recursive=True))
        if not json_files:
            # Fallback: try direct search without subdirs
            json_files = sorted(glob(os.path.join(json_dir, "*.json")))
        return json_files
    raise FileNotFoundError(f"json_dir not found: {json_dir}")


def read_json(path: str) -> Dict[str, Any]:
    """Read JSON file safely"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return {}  # Return empty dict for empty files
        return json.loads(content)


def get_image_path(json_path: str, data: Dict[str, Any], jpg_dir: str = None) -> str:
    """Resolve JPG path from JSON metadata"""
    # Prefer explicit mapping via source_data_name_jpg
    src = data.get("source_data_info", {})
    jpg_name = src.get("source_data_name_jpg", None)

    if jpg_dir and jpg_name:
        # Try direct path
        path = os.path.join(jpg_dir, jpg_name)
        if os.path.exists(path):
            return path

        # Try searching in subdirectories (e.g., press_jpg, report_jpg)
        for subdir in os.listdir(jpg_dir):
            subdir_path = os.path.join(jpg_dir, subdir)
            if os.path.isdir(subdir_path):
                path = os.path.join(subdir_path, jpg_name)
                if os.path.exists(path):
                    return path

    # Fallback: replace "json" with "jpg" in path (e.g., press_json -> press_jpg)
    if jpg_name:
        # Replace *_json with *_jpg in directory path
        json_dir = os.path.dirname(json_path)
        jpg_dir_candidate = json_dir.replace("_json", "_jpg")
        maybe = os.path.join(jpg_dir_candidate, jpg_name)
        if os.path.exists(maybe):
            return maybe

    # Last resort: same dir, MI3 -> MI2.jpg
    base = os.path.splitext(os.path.basename(json_path))[0]
    sibling = os.path.join(os.path.dirname(json_path), base.replace("MI3", "MI2") + ".jpg")
    if os.path.exists(sibling):
        return sibling

    raise FileNotFoundError(f"Could not resolve JPG for {json_path} (jpg_dir={jpg_dir}, jpg_name={jpg_name})")


def simple_tokenize(s: str) -> List[str]:
    """Simple Korean tokenization by whitespace and punctuation"""
    s = (s or "")
    s = s.replace("##", " ").replace(",", " ").replace("(", " ").replace(")", " ")
    s = s.replace(":", " ").replace("?", " ").replace("!", " ").replace("·", " ")
    return [t for t in s.strip().split() if t]


def is_visual_ann(a: dict) -> bool:
    """Filter visual elements (V* class_id or table/chart-related) with non-empty query"""
    cid = str(a.get("class_id", "") or "")
    cname = str(a.get("class_name", "") or "")
    has_q = bool(str(a.get("visual_instruction", "") or "").strip())
    looks_visual = cid.startswith("V") or any(k in cname for k in ["표", "차트", "그래프", "chart", "table"])
    return has_q and looks_visual


class Vocab:
    """Simple vocabulary for Korean text queries"""
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.freq: Dict[str, int] = {}
        self.itos: List[str] = ["<pad>", "<unk>"]
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    def build(self, texts: List[str]):
        """Build vocabulary from list of texts"""
        for s in texts:
            for tok in simple_tokenize(s):
                self.freq[tok] = self.freq.get(tok, 0) + 1
        for tok, f in sorted(self.freq.items(), key=lambda x: (-x[1], x[0])):
            if f >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, s: str, max_len: int = 40) -> List[int]:
        """Encode text to token ids"""
        toks = simple_tokenize(s)[:max_len]
        if not toks:
            return [1]  # ensure length>=1 with <unk>
        return [self.stoi.get(t, 1) for t in toks]


class UniDSet(Dataset):
    """UniD Visual QA Dataset for Korean document understanding"""
    def __init__(self, json_files: List[str], jpg_dir: str = None, vocab: Vocab = None,
                 build_vocab: bool = False, resize_to: Tuple[int, int] = (CFG.IMG_SIZE, CFG.IMG_SIZE)):
        self.items = []
        skipped = 0
        skipped_empty = 0
        skipped_no_img = 0
        total = len(json_files)

        print(f"[Loading] Processing {total} JSON files...")

        for idx, jf in enumerate(json_files):
            data = read_json(jf)
            if not data:  # Skip empty JSON files
                skipped += 1
                skipped_empty += 1
                continue
            ann = data.get("learning_data_info", {}).get("annotation", [])
            try:
                img_path = get_image_path(jf, data, jpg_dir=jpg_dir)
            except FileNotFoundError:
                skipped += 1
                skipped_no_img += 1
                continue
            for a in ann:
                if not is_visual_ann(a):
                    continue
                qid = a.get("instance_id", "")

                # Combine visual_instruction and visual_answer as query text
                instruction = str(a.get("visual_instruction", "")).strip()
                answer = str(a.get("visual_answer", "")).strip()

                # Concatenate instruction and answer with space
                if instruction and answer:
                    qtxt = f"{instruction} {answer}"
                elif instruction:
                    qtxt = instruction
                elif answer:
                    qtxt = answer
                else:
                    qtxt = ""

                bbox = a.get("bounding_box", None)  # train/val has bbox; test may be None
                cname = a.get("class_name", "")
                self.items.append({
                    "json": jf, "img": img_path,
                    "query_id": qid, "query": qtxt,
                    "bbox": bbox, "class_name": cname,
                })

        print(f"✅ JSON 로딩 완료!")
        print(f"   Total JSON files: {total}")
        print(f"   Skipped (empty): {skipped_empty}")
        print(f"   Skipped (no image): {skipped_no_img}")
        print(f"   Valid files: {total - skipped}")
        print(f"   Total samples: {len(self.items)}")

        if len(self.items) == 0:
            print(f"\n⚠️  ERROR: No samples loaded!")
            print(f"   JSON dir: {json_files[0] if json_files else 'N/A'}")
            print(f"   JPG dir: {jpg_dir}")
            # Debug: show first file's structure
            if json_files:
                sample_data = read_json(json_files[0])
                if sample_data:
                    print(f"   Sample JSON keys: {list(sample_data.keys())}")
                    src = sample_data.get("source_data_info", {})
                    print(f"   source_data_info: {src}")
                    print(f"   Expected JPG name: {src.get('source_data_name_jpg', 'NOT FOUND')}")

        self.vocab = vocab if vocab is not None else Vocab(min_freq=1)
        if build_vocab:
            self.vocab.build([it["query"] for it in self.items])

        self.resize_to = resize_to
        if _TF_OK:
            self.tf = T.Compose([T.Resize(resize_to), T.ToTensor()])
        else:
            self.tf = None  # will manually convert

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        """Manually convert PIL Image to tensor"""
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        img = Image.open(it["img"]).convert("RGB")
        W, H = img.size
        if self.tf is not None:
            img_t = self.tf(img)
        else:
            img = img.resize(self.resize_to, Image.BILINEAR)
            img_t = self._pil_to_tensor(img)

        ids = self.vocab.encode(it["query"], max_len=40)
        length = max(1, len(ids))  # safety: ensure >=1

        sample: Dict[str, Any] = {
            "image": img_t,
            "query_ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "query_text": it["query"],
            "query_id": it["query_id"],
            "orig_size": (W, H),
            "class_name": it["class_name"],
        }
        if it["bbox"] is not None and isinstance(it["bbox"], (list, tuple)) and len(it["bbox"]) == 4:
            x, y, w, h = it["bbox"]
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            nw = w / W
            nh = h / H
            target = torch.tensor([cx, cy, nw, nh], dtype=torch.float32)
        else:
            target = None
        sample["target"] = target
        return sample


def collate_fn(batch: List[Dict[str, Any]]):
    """Collate function for padding variable-length queries"""
    # pad variable-length queries
    max_len = max(max(1, int(b["length"])) for b in batch)
    B = len(batch)
    ids = torch.zeros(B, max_len, dtype=torch.long)
    lens = torch.zeros(B, dtype=torch.long)
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    targets = []
    meta = []
    for i, b in enumerate(batch):
        l = max(1, int(b["length"]))
        ids[i, :l] = b["query_ids"][:l]
        lens[i] = l
        targets.append(b["target"])
        meta.append({
            "query_id": b["query_id"],
            "query_text": b["query_text"],
            "orig_size": b["orig_size"],
            "class_name": b["class_name"],
        })
    return imgs, ids, lens, targets, meta
