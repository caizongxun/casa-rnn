"""
actor.py -- BTCActor  (v3)

Default d_model reduced to 128 for faster CPU iteration.
Can be overridden via --d-model 256 when on GPU.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PatchEncoder

ACTION_NAMES = ["LONG", "FLAT", "SHORT"]


class BTCActor(nn.Module):

    def __init__(
        self,
        in_channels: int = 5,
        patch_size: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = PatchEncoder(
            in_channels=in_channels,
            patch_size=patch_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for mod in [self.policy_head, self.confidence_head, self.value_head]:
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        last_policy = [m for m in self.policy_head.modules()
                       if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_policy.weight, gain=0.1)

    def forward(self, x: torch.Tensor):
        ctx    = self.encoder(x)
        logits = self.policy_head(ctx)
        conf   = self.confidence_head(ctx)
        value  = self.value_head(ctx)
        return logits, conf, value

    @torch.no_grad()
    def get_signal(self, ohlcv: torch.Tensor) -> Dict:
        self.eval()
        if ohlcv.dim() == 2:
            ohlcv = ohlcv.unsqueeze(0)
        logits, conf, _ = self(ohlcv)
        probs  = F.softmax(logits, dim=-1)
        action = probs.argmax(dim=-1).item()
        return {
            "action":     action,
            "action_str": ACTION_NAMES[action],
            "confidence": conf.item(),
            "probs":      {n: probs[0, i].item() for i, n in enumerate(ACTION_NAMES)},
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "in_channels": self.encoder.stem.in_features // self.encoder.patch_size,
                "patch_size":  self.encoder.patch_size,
                "d_model":     self.encoder.d_model,
                "n_heads":     self.encoder.transformer.layers[0].self_attn.num_heads,
                "n_layers":    len(self.encoder.transformer.layers),
            },
        }, path)
        print(f"[BTCActor] saved -> {path}")

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> "BTCActor":
        ckpt  = torch.load(path, map_location=map_location, weights_only=False)
        cfg   = ckpt.get("config", {})
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[BTCActor] loaded <- {path}")
        return model
