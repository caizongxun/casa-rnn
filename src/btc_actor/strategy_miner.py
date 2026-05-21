"""
strategy_miner.py

Two complementary strategy discovery methods:

  PatternMiner  (Type C — recurrent market patterns)
  ─────────────────────────────────────────────────
  Extracts the actor's internal hidden-state vectors from the validation
  set, clusters them with KMeans, and for each cluster:
    * computes mean feature values (human-readable profile)
    * measures forward return distribution  (next `horizon` candles)
    * assigns a direction label  (LONG / FLAT / SHORT)
    * measures win-rate and mean reward
    * names the pattern automatically from its dominant features

  RuleMiner  (Type A — decision rules)
  ─────────────────────────────────────
  Trains a shallow decision tree (max_depth=4) to mimic the actor's
  output probabilities on the validation set, then converts every
  leaf path into a plain-English IF/THEN rule with:
    * feature conditions (e.g. "RSI14 < 30")
    * predicted action  (LONG / FLAT / SHORT)
    * estimated win-rate and mean return from that leaf
    * sample count and confidence

Usage
-----
    from btc_actor.strategy_miner import PatternMiner, RuleMiner

    pm = PatternMiner(model, device=device, feature_names=feat_cols)
    patterns = pm.mine(X_val, ohlcv_val, horizon=4, n_clusters=12)

    rm = RuleMiner(feature_names=feat_cols)
    rules = rm.mine(X_val, y_val, model, device=device)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

# Labels
_LABEL = {0: "LONG", 1: "FLAT", 2: "SHORT"}
_LABEL_IDX = {"LONG": 0, "FLAT": 1, "SHORT": 2}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MarketPattern:
    """A single discovered market pattern (Type C)."""
    pattern_id:     int
    name:           str               # auto-generated human-readable name
    direction:      str               # LONG / FLAT / SHORT
    n_samples:      int
    win_rate:       float             # fraction where signal matched outcome
    mean_return:    float             # mean forward return (log-return × horizon)
    dominant_features: List[str]      # top-3 features driving this cluster
    feature_profile: dict             # feat_name -> mean z-score in cluster
    confidence:     float             # probability mass on the predicted class


@dataclass
class TradingRule:
    """A single extracted decision rule (Type A)."""
    rule_id:     int
    conditions:  List[str]            # e.g. ["RSI14 < 30.5", "ATR14 > 0.02"]
    action:      str                  # LONG / FLAT / SHORT
    n_samples:   int
    win_rate:    float
    mean_return: float
    confidence:  float                # leaf value / leaf total
    readable:    str                  # full IF ... THEN ... string


# ---------------------------------------------------------------------------
# PatternMiner
# ---------------------------------------------------------------------------

class PatternMiner:
    """
    Cluster the model's hidden representations to find recurring market
    patterns.  Works with any model that has a `encode(x)` method returning
    a (batch, d_model) tensor, OR falls back to using the raw feature
    statistics of each window.
    """

    def __init__(
        self,
        model,
        device: str = "cpu",
        feature_names: Optional[List[str]] = None,
    ):
        self.model        = model
        self.device       = device
        self.feature_names = feature_names or [f"feat_{i}" for i in range(64)]

    # ------------------------------------------------------------------
    def _extract_representations(self, X: torch.Tensor, batch_size: int = 512) -> np.ndarray:
        """Extract hidden-state vectors (or fallback stats) for each window."""
        self.model.eval()
        reps = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = X[i:i+batch_size].to(self.device)
                if hasattr(self.model, "encode"):
                    h = self.model.encode(xb)          # (B, d_model)
                else:
                    # fallback: use last-timestep features
                    h = xb[:, -1, :]                   # (B, C)
                reps.append(h.cpu().numpy())
        return np.concatenate(reps, axis=0)

    # ------------------------------------------------------------------
    def _compute_forward_returns(self, ohlcv: np.ndarray, horizon: int) -> np.ndarray:
        """Compute log forward return from close[t] to close[t+horizon]."""
        close = ohlcv[:, 3]
        fwd = np.full(len(close), np.nan)
        for i in range(len(close) - horizon):
            fwd[i] = np.log(close[i + horizon] / (close[i] + 1e-8))
        return fwd

    # ------------------------------------------------------------------
    def _name_pattern(
        self,
        top_feats: List[str],
        direction: str,
        mean_return: float,
    ) -> str:
        """Generate a descriptive name for a pattern from its top features."""
        parts = []
        feat_keywords = {
            "rsi":      "RSI-extreme",
            "vol":      "vol-surge",
            "atr":      "high-volatility",
            "funding":  "funding-pressure",
            "ls_ratio": "LS-imbalance",
            "open_int": "OI-spike",
            "corr_eth": "ETH-corr",
            "corr_sol": "SOL-corr",
            "corr_bnb": "BNB-corr",
            "corr_xrp": "XRP-corr",
            "momentum": "momentum",
            "taker":    "taker-pressure",
            "close_str": "breakout",
            "log_ret":  "sharp-move",
        }
        for f in top_feats[:2]:
            fl = f.lower()
            for kw, label in feat_keywords.items():
                if kw in fl:
                    if label not in parts:
                        parts.append(label)
                    break
        tag = "-".join(parts) if parts else "mixed"
        strength = "strong" if abs(mean_return) > 0.003 else "mild"
        return f"{direction.lower()}-{tag}-{strength}"

    # ------------------------------------------------------------------
    def mine(
        self,
        X: torch.Tensor,
        ohlcv: np.ndarray,
        horizon: int = 4,
        n_clusters: int = 12,
        min_samples: int = 30,
    ) -> List[MarketPattern]:
        """
        Main entry point.  Returns a list of MarketPattern objects sorted
        by |mean_return| descending (most actionable first).
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        print(f"[PatternMiner] Extracting representations for {len(X):,} windows ...")
        reps = self._extract_representations(X)

        # Normalise before clustering
        scaler = StandardScaler()
        reps_scaled = scaler.fit_transform(reps)

        print(f"[PatternMiner] KMeans clustering (k={n_clusters}) ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = km.fit_predict(reps_scaled)

        # Get model predictions to assign direction per cluster
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                xb = X[i:i+512].to(self.device)
                logits = self.model(xb)
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
        all_probs = np.concatenate(all_probs, axis=0)  # (N, 3)

        # Last-timestep feature values for profiling
        feat_vals = X[:, -1, :].numpy()   # (N, C)
        fwd_ret   = self._compute_forward_returns(ohlcv, horizon)

        patterns = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if mask.sum() < min_samples:
                continue

            cluster_probs   = all_probs[mask]             # (K, 3)
            mean_probs      = cluster_probs.mean(axis=0)  # (3,)
            direction_idx   = int(mean_probs.argmax())
            direction       = _LABEL[direction_idx]
            confidence      = float(mean_probs[direction_idx])

            # Forward return of cluster members
            fwd_sub = fwd_ret[mask]
            valid   = fwd_sub[~np.isnan(fwd_sub)]
            mean_ret = float(valid.mean()) if len(valid) > 0 else 0.0

            # Win-rate: fraction where realized direction matches prediction
            if direction == "LONG":
                wins = (valid > 0).sum()
            elif direction == "SHORT":
                wins = (valid < 0).sum()
            else:
                wins = (np.abs(valid) < 0.002).sum()
            win_rate = float(wins / len(valid)) if len(valid) > 0 else 0.0

            # Feature profile (z-scores relative to full set)
            feat_mean_cluster = feat_vals[mask].mean(axis=0)
            feat_mean_all     = feat_vals.mean(axis=0)
            feat_std_all      = feat_vals.std(axis=0) + 1e-8
            z_scores          = (feat_mean_cluster - feat_mean_all) / feat_std_all

            top_idx    = np.argsort(np.abs(z_scores))[::-1][:5]
            top_feats  = [self.feature_names[i] if i < len(self.feature_names) else f"feat_{i}"
                          for i in top_idx]
            feat_profile = {top_feats[j]: float(z_scores[top_idx[j]]) for j in range(len(top_feats))}

            name = self._name_pattern(top_feats, direction, mean_ret)

            patterns.append(MarketPattern(
                pattern_id       = cluster_id,
                name             = name,
                direction        = direction,
                n_samples        = int(mask.sum()),
                win_rate         = win_rate,
                mean_return      = mean_ret,
                dominant_features= top_feats[:3],
                feature_profile  = feat_profile,
                confidence       = confidence,
            ))

        patterns.sort(key=lambda p: abs(p.mean_return), reverse=True)
        print(f"[PatternMiner] Found {len(patterns)} patterns (>={min_samples} samples each)")
        return patterns


# ---------------------------------------------------------------------------
# RuleMiner
# ---------------------------------------------------------------------------

class RuleMiner:
    """
    Distill the trained actor into a shallow decision tree, then convert
    every leaf into a plain-English IF/THEN rule.

    Uses the last-timestep feature vector as input to the tree so that
    rules are expressed in the original feature space (RSI, ATR, etc.).
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        max_depth: int = 4,
    ):
        self.feature_names = feature_names or []
        self.max_depth = max_depth
        self.tree_ = None

    # ------------------------------------------------------------------
    def _get_feature_name(self, idx: int) -> str:
        if idx < len(self.feature_names):
            return self.feature_names[idx]
        return f"feat_{idx}"

    # ------------------------------------------------------------------
    def _extract_rules(
        self,
        tree,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        fwd_ret: Optional[np.ndarray] = None,
    ) -> List[TradingRule]:
        """Walk the decision tree and produce one TradingRule per leaf."""
        from sklearn.tree import _tree

        t         = tree.tree_
        feat_name = [
            self._get_feature_name(i) if i != _tree.TREE_UNDEFINED else "undefined"
            for i in t.feature
        ]

        rules = []
        rule_id = 0

        def recurse(node, conditions):
            nonlocal rule_id
            if t.feature[node] == _tree.TREE_UNDEFINED:
                # Leaf node
                values   = t.value[node][0]
                total    = values.sum()
                pred_cls = int(values.argmax())
                action   = _LABEL.get(pred_cls, "FLAT")
                conf     = float(values[pred_cls] / (total + 1e-8))
                n_samples = int(total)

                # Find which training samples reach this leaf
                node_indicator = tree.decision_path(X_raw)
                leaf_ids       = tree.apply(X_raw)
                in_leaf        = leaf_ids == node
                n_in_leaf      = int(in_leaf.sum())

                if n_in_leaf > 0:
                    y_leaf = y_raw[in_leaf]
                    # Win-rate: fraction with matching direction
                    if fwd_ret is not None:
                        fr = fwd_ret[in_leaf]
                        valid = fr[~np.isnan(fr)]
                        mean_ret = float(valid.mean()) if len(valid) > 0 else 0.0
                        if action == "LONG":
                            win_rate = float((valid > 0).sum() / (len(valid) + 1e-8))
                        elif action == "SHORT":
                            win_rate = float((valid < 0).sum() / (len(valid) + 1e-8))
                        else:
                            win_rate = float((np.abs(valid) < 0.002).sum() / (len(valid) + 1e-8))
                    else:
                        # fallback: use tree label accuracy
                        win_rate = float((y_leaf == pred_cls).sum() / (n_in_leaf + 1e-8))
                        mean_ret = 0.0
                else:
                    win_rate = 0.0
                    mean_ret = 0.0

                readable_conds = "  AND  ".join(conditions) if conditions else "(always)"
                readable = f"IF {readable_conds}  THEN {action}  (win={win_rate:.1%}, ret={mean_ret:+.4f}, n={n_in_leaf})"

                rules.append(TradingRule(
                    rule_id     = rule_id,
                    conditions  = list(conditions),
                    action      = action,
                    n_samples   = n_in_leaf,
                    win_rate    = win_rate,
                    mean_return = mean_ret,
                    confidence  = conf,
                    readable    = readable,
                ))
                rule_id += 1
            else:
                threshold = t.threshold[node]
                fname     = feat_name[node]
                recurse(t.children_left[node],  conditions + [f"{fname} <= {threshold:.4f}"])
                recurse(t.children_right[node], conditions + [f"{fname} > {threshold:.4f}"])

        recurse(0, [])
        return rules

    # ------------------------------------------------------------------
    def mine(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        model,
        device: str = "cpu",
        ohlcv: Optional[np.ndarray] = None,
        horizon: int = 4,
        min_samples_leaf: int = 50,
    ) -> List[TradingRule]:
        """
        Distill the model into a decision tree and extract rules.
        Returns list of TradingRule sorted by |mean_return| descending.
        """
        from sklearn.tree import DecisionTreeClassifier

        # Get model soft labels (probability vectors) as supervision
        model.eval()
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                xb = X[i:i+512].to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
        all_probs = np.concatenate(all_probs, axis=0)
        pseudo_labels = all_probs.argmax(axis=1)

        # Use last-timestep feature vector (interpretable features)
        X_feat = X[:, -1, :].numpy()   # (N, C)
        y_raw  = y.numpy()

        print(f"[RuleMiner] Training decision tree (max_depth={self.max_depth}, "
              f"min_samples_leaf={min_samples_leaf}) ...")
        dt = DecisionTreeClassifier(
            max_depth         = self.max_depth,
            min_samples_leaf  = min_samples_leaf,
            class_weight      = "balanced",
            random_state      = 42,
        )
        dt.fit(X_feat, pseudo_labels)
        self.tree_ = dt

        acc = (dt.predict(X_feat) == pseudo_labels).mean()
        print(f"[RuleMiner] Tree fidelity (matches model): {acc:.1%}")

        # Compute forward returns if OHLCV provided
        if ohlcv is not None:
            close  = ohlcv[:, 3]
            n      = len(close)
            fwd    = np.full(n, np.nan)
            for i in range(n - horizon):
                fwd[i] = np.log(close[i + horizon] / (close[i] + 1e-8))
        else:
            fwd = None

        rules = self._extract_rules(dt, X_feat, y_raw, fwd)

        # Filter low-sample leaves and sort by actionability
        rules = [r for r in rules if r.n_samples >= min_samples_leaf]
        rules = [r for r in rules if r.action != "FLAT"]  # optional: keep only directional
        rules.sort(key=lambda r: abs(r.mean_return) * r.win_rate, reverse=True)

        print(f"[RuleMiner] Extracted {len(rules)} actionable rules")
        return rules
