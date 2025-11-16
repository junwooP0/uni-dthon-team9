#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Attention VLM Model for UniD Visual QA
ResNet18 (vision) + BiGRU (text) + Cross-Attention fusion
"""

import math
import torch
import torch.nn as nn

from config import CFG

# torchvision ResNet backbone
try:
    from torchvision.models import (
    resnet18, resnet34,
    ResNet18_Weights, ResNet34_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    )
    _BACKBONE_OK = True
except Exception:
    _BACKBONE_OK = False
    
# 🔥 CLIP (optional)
try:
    import clip  # pip install git+https://github.com/openai/CLIP.git
    _CLIP_OK = True
except Exception:
    _CLIP_OK = False

class TextEncoder(nn.Module):
    """BiGRU-based text encoder for Korean queries"""
    def __init__(self, vocab_size: int, emb_dim: int = CFG.DIM, hidden: int = CFG.DIM):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, emb_dim)
        # self.norm = nn.LayerNorm(emb_dim)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, L) token ids
            lengths: (B,) sequence lengths
        Returns:
            q: (B, D) query embedding
        """
        x = self.emb(tokens)  # (B, L, E)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.gru(packed)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)  # (B, 2*hidden)
        q = self.proj(h_cat)  # (B, D)
        # q = self.norm(q)
        # q = self.dropout(q)
        return q


class TinyCNN(nn.Module):
    """Fallback tiny CNN if ResNet18 not available"""
    def __init__(self, out_dim: int = CFG.DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)  # (B, D, H', W')


class ImageEncoder(nn.Module):
    """ResNet18 / ResNet34 / ResNet50 / EfficientNet-B0 / CLIP ViT-B/16 기반 Image Encoder"""
    def __init__(self, out_dim: int = CFG.DIM, pretrained: bool = True,
                 img_size: int = CFG.IMG_SIZE, backbone: str = "resnet34"):
        super().__init__()
        self.resize = None
        self.use_clip = False  # 기본은 False

        # 🔥 0) CLIP backbone 특수 처리
        if backbone == "clip_vit_b16":
            if not _CLIP_OK:
                raise RuntimeError("CLIP is not installed. Run: pip install git+https://github.com/openai/CLIP.git")

            clip_model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
            self.clip_visual = clip_model.visual
            self.use_clip = True

            # in_dim = self.clip_visual.width  # ViT-B/16 hidden dim = 768
            in_dim = self.clip_visual.conv1.out_channels  # 보통 768 (ViT-B/16 width)

            # proj: CLIP dim -> 모델 dim(예: 256)
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

            # CLIP 정규화 mean/std를 buffer로 등록
            import torch
            self.register_buffer(
                "clip_mean",
                torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "clip_std",
                torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
            )

            # 일단 CLIP 비전은 freeze해서 쓰는 걸 기본으로
            if pretrained:
                for p in self.clip_visual.parameters():
                    p.requires_grad = False

            # CLIP은 torchvision backbone 안 쓰므로 여기서 바로 return
            return

        # 🔥 1) 기존 torchvision backbone들 (ResNet / EfficientNet)
        if _BACKBONE_OK:
            try:
                # ---- 1) Select backbone ----
                if backbone == "resnet18":
                    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                    m = resnet18(weights=weights)
                    in_dim = 512  # resnet18 output channels

                elif backbone == "resnet34":
                    weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
                    m = resnet34(weights=weights)
                    in_dim = 512  # resnet34 output channels

                elif backbone == "resnet50":
                    from torchvision.models import resnet50, ResNet50_Weights
                    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
                    m = resnet50(weights=weights)
                    in_dim = 2048  # resnet50 output channels

                elif backbone == "efficientnet_b3":
                    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
                    m = efficientnet_b3(weights=weights)
                    in_dim = 1536  # EfficientNet-B3 output channels
                    self.backbone = m.features  # EfficientNet은 features가 backbone

                elif backbone == "efficientnet_v2_s":
                    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
                    m = efficientnet_v2_s(weights=weights)
                    in_dim = 1280  # EfficientNetV2-S output channels
                    self.backbone = m.features

                else:
                    raise ValueError(f"Unknown backbone: {backbone}")

                # ---- 2) Remove classifier layer ----
                if backbone.startswith("resnet"):
                    layers = list(m.children())[:-2]      # conv ~ layer4까지
                    self.backbone = nn.Sequential(*layers)
                elif backbone.startswith("efficientnet"):
                    self.backbone = m.features
                else:
                    raise ValueError(f"Unknown backbone when building backbone: {backbone}")

                # ---- 3) Project to model dim ----
                self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

                # ---- 4) Resize transforms ----
                from torchvision import transforms as T
                self.resize = T.Compose([
                    T.Resize((img_size, img_size)),
                    T.ToTensor()
                ])

            except Exception:
                # In case of torchvision issues
                self.backbone = TinyCNN(out_dim)
                self.proj = nn.Identity()

        else:
            # Fallback when torchvision not available
            self.backbone = TinyCNN(out_dim)
            self.proj = nn.Identity()


    def forward(self, x):
        # x: (B, 3, H, W)

        # 🔥 CLIP ViT-B/16 backbone일 때
        if getattr(self, "use_clip", False):
            visual = self.clip_visual
            B = x.size(0)

            # 1) CLIP 정규화
            x = (x - self.clip_mean) / self.clip_std
            x = x.to(visual.conv1.weight.dtype)

            # 2) patch embedding
            x = visual.conv1(x)  # (B, width, gh, gw)
            x = x.reshape(B, visual.conv1.out_channels, -1)  # (B, width, N)
            x = x.permute(0, 2, 1)  # (B, N, width)

            # 3) cls token 붙이기
            cls_token = visual.class_embedding.to(x.dtype)
            cls_token = cls_token + torch.zeros(B, 1, cls_token.shape[-1],
                                                dtype=x.dtype, device=x.device)
            x = torch.cat([cls_token, x], dim=1)  # (B, 1+N, width)

            # 4) positional embedding + ln_pre
            pos = visual.positional_embedding.to(x.dtype)

            # pos: (L, D) 또는 (1, L, D) 일 수 있음
            if pos.ndim == 2:
                # (L, D) -> (1, L, D) 로 만들어서 (B, L, D)에 broadcast
                pos = pos.unsqueeze(0)
            elif pos.ndim == 3:
                # 이미 (1, L, D) 형태면 그대로 사용
                pass
            else:
                raise RuntimeError(f"Unexpected positional_embedding shape: {pos.shape}")

            x = x + pos  # (B, L, D) + (1, L, D)
            x = visual.ln_pre(x)

            # 5) transformer 통과
            x = x.permute(1, 0, 2)  # (1+N, B, width)
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)  # (B, 1+N, width)

            # 6) cls 토큰 제외한 patch들만 사용
            patch_tokens = x[:, 1:, :]  # (B, N, width)
            N = patch_tokens.size(1)
            D = patch_tokens.size(2)

            # N = H' * W' 가정하고 2D로 reshape
            H = W = int(N ** 0.5)
            patch_tokens = patch_tokens[:, :H * W, :]  # 안전하게 자르기
            fmap = patch_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)

            fmap = self.proj(fmap)  # (B, out_dim, H, W)
            return fmap

        # 🔁 기존 ResNet / EfficientNet backbone일 때
        f = self.backbone(x)
        f = self.proj(f)
        return f




class CrossAttentionBBox(nn.Module):
    """Cross-Attention fusion + BBox prediction head"""
    def __init__(self, dim: int = CFG.DIM):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        
        # self.norm_ctx = nn.LayerNorm(dim)
        
        self.bbox_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            # nn.Linear(dim * 2, dim),
            # nn.ReLU(inplace=True),
            nn.Linear(dim, 4)  # (cx, cy, w, h) normalized via sigmoid
        )

    def forward(self, q_vec: torch.Tensor, fmap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_vec: (B, D) query embedding from text encoder
            fmap: (B, D, H, W) image feature map
        Returns:
            pred: (B, 4) normalized bbox [cx, cy, w, h] in [0,1]
        """
        B, D, H, W = fmap.shape
        q = self.q_proj(q_vec)             # (B, D)
        K = self.k_proj(fmap)              # (B, D, H, W)
        V = self.v_proj(fmap)              # (B, D, H, W)

        Kf = K.flatten(2).transpose(1, 2)  # (B, HW, D)
        Vf = V.flatten(2).transpose(1, 2)  # (B, HW, D)
        q = q.unsqueeze(1)                 # (B, 1, D)

        attn = torch.matmul(q, Kf.transpose(1, 2)) / math.sqrt(D)  # (B, 1, HW)
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, Vf).squeeze(1)  # (B, D)
        # ctx = self.norm_ctx(ctx)

        pred = self.bbox_head(ctx)         # (B, 4)
        pred = torch.sigmoid(pred)         # normalize to [0,1]
        return pred


class CrossAttnVLM(nn.Module):
    """Complete Cross-Attention Vision-Language Model for BBox prediction"""
    def __init__(self, vocab_size: int, dim: int = CFG.DIM, pretrained_backbone: bool = True, 
                 img_size: int = CFG.IMG_SIZE, backbone: str = "resnet34"):
        super().__init__()
        self.txt = TextEncoder(vocab_size=vocab_size, emb_dim=dim, hidden=dim)
        self.img = ImageEncoder(out_dim=dim,
                        pretrained=pretrained_backbone,
                        img_size=img_size,
                        backbone=backbone)   # 🔥 전달
        self.head = CrossAttentionBBox(dim=dim)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) image tensor
            tokens: (B, L) query token ids
            lengths: (B,) query lengths
        Returns:
            pred_norm: (B, 4) normalized bbox [cx, cy, w, h] in [0,1]
        """
        q = self.txt(tokens, lengths)             # (B, D)
        fmap = self.img(images)                   # (B, D, H', W')
        pred_norm = self.head(q, fmap)            # (B, 4) in [0,1]
        return pred_norm
