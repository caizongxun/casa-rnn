"""
actor.py -- BTCActor

The full model used for live entry signal generation.

Outputs
-------
* direction_logits : (B, 3)   LONG / FLAT / SHORT
* confidence       : (B, 1)   [0, 1]  how certain the model is
* value            : (B, 1)   critic estimate of expected return
                              (used during RL fine-tuning, ignored at inference)

The actor and critic share the encoder.  Only the heads are separate.
This is standard Actor-Critic (used in PPO).

Inference usage (live trading)
------------------------------
    actor = BTCActor.load("checkpoint.pt")
    signal = actor.get_signal(ohlcv_window)   # returns {action, confidence, probs}
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
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
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

        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),   # LONG / FLAT / SHORT
        )

        # Confidence head -- separate from policy to avoid entanglement
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Value head (critic, used only during PPO fine-tuning)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        x : (B, T, C)  raw OHLCV window
        returns logits (B,3), confidence (B,1), value (B,1)
        """
        ctx = self.encoder(x)                          # (B, d_model)
        logits = self.policy_head(ctx)                 # (B, 3)
        conf   = self.confidence_head(ctx)             # (B, 1)
        value  = self.value_head(ctx)                  # (B, 1)
        return logits, conf, value

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_signal(self, ohlcv: torch.Tensor) -> Dict:
        """
        Live inference.  ohlcv can be (T, C) or (1, T, C).
        Returns a dict ready for downstream trading logic.
        """
        self.eval()
        if ohlcv.dim() == 2:
            ohlcv = ohlcv.unsqueeze(0)          # add batch dim
        logits, conf, _ = self(ohlcv)
        probs  = F.softmax(logits, dim=-1)      # (1, 3)
        action = probs.argmax(dim=-1).item()    # 0=LONG 1=FLAT 2=SHORT
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
            "state_dict":  self.state_dict(),
            "config": {
                "in_channels": self.encoder.stem.in_features // self.encoder.patch_size,
                "patch_size":  self.encoder.patch_size,
                "d_model":     self.encoder.d_model,
            },
        }, path)
        print(f"[BTCActor] saved -> {path}")

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> "BTCActor":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg  = ckpt.get("config", {})
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[BTCActor] loaded <- {path}")
        return model
