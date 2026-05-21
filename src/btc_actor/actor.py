"""
actor.py  v4

New
---
* in_channels default = 31  (full feature set)
* Regime head: auxiliary 3-class output (trend / range / volatile)
  trained with a small auxiliary loss -> forces encoder to learn regime
* Independent value network: takes detached encoder output so
  critic gradients don't interfere with policy learning
* save() / load() store/restore in_channels and patch_sizes so
  checkpoints are self-describing
* MC-Dropout inference: get_signal(mc_samples=20) returns uncertainty
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PatchEncoder, RMSNorm

ACTION_NAMES = ["LONG", "FLAT", "SHORT"]
REGIME_NAMES = ["TREND", "RANGE", "VOLATILE"]


class BTCActor(nn.Module):

    def __init__(
        self,
        in_channels: int       = 31,
        patch_sizes: List[int] = None,
        d_model: int           = 128,
        n_heads: int           = 4,
        n_layers: int          = 3,
        dropout: float         = 0.1,
    ):
        super().__init__()
        if patch_sizes is None:
            patch_sizes = [4, 12, 24]
        self.in_channels = in_channels
        self.patch_sizes = patch_sizes
        self.d_model     = d_model

        self.encoder = PatchEncoder(
            in_channels=in_channels,
            patch_sizes=patch_sizes,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            RMSNorm(d_model // 2),
            nn.Linear(d_model // 2, 3),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Independent value network (uses detached ctx)
        self.value_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            RMSNorm(d_model // 2),
            nn.Linear(d_model // 2, 1),
        )

        # Regime auxiliary head
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 3),
        )

        # Masked-patch reconstruction head (for Stage 0 self-supervised)
        self.recon_head = nn.Linear(d_model, in_channels * patch_sizes[0])

        self._init_weights()

    def _init_weights(self):
        for mod in [self.policy_head, self.confidence_head,
                    self.value_net, self.regime_head, self.recon_head]:
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        last = [m for m in self.policy_head.modules()
                if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last.weight, gain=0.01)

    def forward(
        self,
        x: torch.Tensor,
        return_regime: bool = False,
    ):
        ctx    = self.encoder(x)
        logits = self.policy_head(ctx)
        conf   = self.confidence_head(ctx)
        value  = self.value_net(ctx.detach())   # independent critic

        if return_regime:
            regime_logits = self.regime_head(ctx)
            return logits, conf, value, regime_logits
        return logits, conf, value

    def masked_reconstruct(self, x: torch.Tensor, mask_ratio: float = 0.15):
        """
        Stage-0 self-supervised: randomly mask patches, reconstruct.
        Returns (recon, mask) where mask is bool (True = masked).
        """
        B, T, C = x.shape
        ps   = self.encoder.patch_sizes[0]
        pad  = (ps - T % ps) % ps
        if pad:
            x_pad = F.pad(x, (0, 0, 0, pad))
        else:
            x_pad = x
        n_p  = x_pad.shape[1] // ps
        mask = torch.rand(B, n_p, device=x.device) < mask_ratio  # (B, n_p)

        # zero out masked patches in input
        x_norm = self.encoder.norm(x_pad.clone())
        x_vsn  = self.encoder.vsn(x_norm)
        patches = x_vsn.reshape(B, n_p, ps * C)
        patches[mask] = 0.0

        # target: original patches
        target = x_norm.reshape(B, n_p, ps * C)

        # run through branch[0] only for reconstruction
        x_m   = patches.reshape(B, n_p * ps, C)
        tok   = self.encoder.branches[0](x_m)          # (B, n_p, branch_dim0)
        # pad to d_model with zeros
        pad_d = self.d_model - tok.shape[-1]
        if pad_d > 0:
            tok = F.pad(tok, (0, pad_d))
        tok = self.encoder.transformer(tok)
        recon = self.recon_head(tok)                    # (B, n_p, ps*C)
        return recon, target, mask

    @torch.no_grad()
    def get_signal(
        self,
        ohlcv: torch.Tensor,
        mc_samples: int = 0,
    ) -> Dict:
        """
        mc_samples > 0  -> MC-Dropout uncertainty estimate.
        Returns action, confidence, probs, uncertainty (std of probs).
        """
        if ohlcv.dim() == 2:
            ohlcv = ohlcv.unsqueeze(0)

        if mc_samples > 0:
            self.train()   # enable dropout
            all_probs = []
            for _ in range(mc_samples):
                logits, _, _ = self(ohlcv)
                all_probs.append(F.softmax(logits, dim=-1))
            probs_stack = torch.stack(all_probs)         # (mc, B, 3)
            mean_probs  = probs_stack.mean(0)
            std_probs   = probs_stack.std(0)
            self.eval()
        else:
            self.eval()
            logits, _, _ = self(ohlcv)
            mean_probs   = F.softmax(logits, dim=-1)
            std_probs    = torch.zeros_like(mean_probs)

        action = mean_probs.argmax(dim=-1).item()
        return {
            "action":      action,
            "action_str":  ACTION_NAMES[action],
            "confidence":  mean_probs[0, action].item(),
            "probs":       {n: mean_probs[0, i].item() for i, n in enumerate(ACTION_NAMES)},
            "uncertainty": std_probs[0].mean().item(),
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "in_channels": self.in_channels,
                "patch_sizes": self.patch_sizes,
                "d_model":     self.d_model,
                "n_heads":     self.encoder.transformer.layers[0].self_attn.num_heads,
                "n_layers":    len(self.encoder.transformer.layers),
            },
        }, path)
        print(f"[BTCActor] saved -> {path}")

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt  = torch.load(path, map_location=map_location, weights_only=False)
        cfg   = ckpt.get("config", {})
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[BTCActor] loaded <- {path}")
        return model
