"""
report_writer.py

Converts the outputs of PatternMiner + RuleMiner into:
  1. strategies_report.json   — machine-readable, for downstream tooling
  2. strategies_report.md     — human-readable Markdown report

Usage
-----
    from btc_actor.report_writer import write_strategy_report
    write_strategy_report(patterns, rules, output_dir="checkpoints")
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from btc_actor.strategy_miner import MarketPattern, TradingRule


def _direction_emoji(direction: str) -> str:
    return {"LONG": "↑", "SHORT": "↓", "FLAT": "→"}.get(direction, "?")


def _bar(value: float, width: int = 20) -> str:
    """Simple ASCII progress bar for win-rate / confidence."""
    filled = max(0, min(width, int(value * width)))
    return "█" * filled + "░" * (width - filled)


def _format_pattern_md(p: MarketPattern, rank: int) -> str:
    lines = [
        f"### Pattern #{rank}: `{p.name}`",
        "",
        f"| | |",
        f"|---|---|",
        f"| **Direction** | {_direction_emoji(p.direction)} {p.direction} |",
        f"| **Win rate** | {p.win_rate:.1%}  `{_bar(p.win_rate)}` |",
        f"| **Mean forward return** | {p.mean_return:+.4f} |",
        f"| **Confidence** | {p.confidence:.1%} |",
        f"| **Samples** | {p.n_samples:,} |",
        "",
        "**Dominant features (z-score vs. market average):**",
        "",
    ]
    for feat, z in p.feature_profile.items():
        direction_sign = "+" if z > 0 else ""
        lines.append(f"- `{feat}`: {direction_sign}{z:.2f}σ")
    lines.append("")
    return "\n".join(lines)


def _format_rule_md(r: TradingRule, rank: int) -> str:
    cond_str = "  \n  AND  ".join(r.conditions) if r.conditions else "(always)"
    lines = [
        f"### Rule #{rank}: {_direction_emoji(r.action)} {r.action}",
        "",
        "**Conditions:**",
        "",
        f"```",
        f"IF  {cond_str}",
        f"THEN  → {r.action}",
        f"```",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Win rate | {r.win_rate:.1%}  `{_bar(r.win_rate)}` |",
        f"| Mean return | {r.mean_return:+.4f} |",
        f"| Confidence | {r.confidence:.1%} |",
        f"| Samples | {r.n_samples:,} |",
        "",
    ]
    return "\n".join(lines)


def write_strategy_report(
    patterns: List[MarketPattern],
    rules: List[TradingRule],
    output_dir: str = "checkpoints",
    top_k_patterns: int = 10,
    top_k_rules: int = 15,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
) -> dict:
    """
    Write strategies_report.json + strategies_report.md.
    Returns a dict summary.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    top_patterns = patterns[:top_k_patterns]
    top_rules    = rules[:top_k_rules]

    # ----------------------------------------------------------------
    # 1. JSON report
    # ----------------------------------------------------------------
    report = {
        "generated_at":  ts,
        "symbol":        symbol,
        "interval":      interval,
        "n_patterns":    len(top_patterns),
        "n_rules":       len(top_rules),
        "patterns":      [asdict(p) for p in top_patterns],
        "rules":         [asdict(r) for r in top_rules],
    }
    json_path = out_path / "strategies_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[ReportWriter] JSON -> {json_path}")

    # ----------------------------------------------------------------
    # 2. Markdown report
    # ----------------------------------------------------------------
    md_lines = [
        f"# Strategy Discovery Report",
        f"",
        f"**Symbol:** {symbol}  |  **Interval:** {interval}  |  **Generated:** {ts}",
        f"",
        f"---",
        f"",
        f"## Executive Summary",
        f"",
    ]

    long_patterns  = [p for p in top_patterns if p.direction == "LONG"]
    short_patterns = [p for p in top_patterns if p.direction == "SHORT"]
    long_rules     = [r for r in top_rules if r.action == "LONG"]
    short_rules    = [r for r in top_rules if r.action == "SHORT"]

    if top_patterns:
        best_p = top_patterns[0]
        md_lines += [
            f"- Best pattern: **{best_p.name}** ({best_p.direction},  "
            f"win-rate {best_p.win_rate:.1%},  mean return {best_p.mean_return:+.4f})",
        ]
    if top_rules:
        best_r = top_rules[0]
        md_lines += [
            f"- Best rule: **{best_r.action}** when `{'  AND  '.join(best_r.conditions[:2])}`  "
            f"(win-rate {best_r.win_rate:.1%},  mean return {best_r.mean_return:+.4f})",
        ]
    md_lines += [
        f"- Discovered **{len(long_patterns)} LONG** + **{len(short_patterns)} SHORT** patterns",
        f"- Extracted **{len(long_rules)} LONG** + **{len(short_rules)} SHORT** rules",
        f"",
        f"---",
        f"",
        f"## Part 1 — Market Patterns (Type C)",
        f"",
        f"Patterns are recurring hidden-state clusters discovered by the model.",
        f"Each represents a distinct market condition the actor has learned to recognise.",
        f"",
    ]

    for i, p in enumerate(top_patterns, 1):
        md_lines.append(_format_pattern_md(p, i))

    md_lines += [
        f"---",
        f"",
        f"## Part 2 — Decision Rules (Type A)",
        f"",
        f"Rules are extracted by distilling the actor into a shallow decision tree.",
        f"Each rule is a human-readable IF/THEN statement expressed in the original",
        f"feature space (RSI, ATR, funding rate, etc.).",
        f"",
    ]

    for i, r in enumerate(top_rules, 1):
        md_lines.append(_format_rule_md(r, i))

    md_lines += [
        "---",
        "",
        "*This report was auto-generated by `btc_actor.report_writer`. "
        "Past performance does not guarantee future results.*",
        "",
    ]

    md_text  = "\n".join(md_lines)
    md_path  = out_path / "strategies_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"[ReportWriter] Markdown -> {md_path}")

    # ----------------------------------------------------------------
    # 3. Print top-5 to console
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TOP DISCOVERED STRATEGIES")
    print("=" * 60)
    print("\n--- Patterns ---")
    for p in top_patterns[:5]:
        print(f"  [{p.direction:5s}] {p.name:<40s}  "
              f"win={p.win_rate:.1%}  ret={p.mean_return:+.4f}  n={p.n_samples}")
    print("\n--- Rules ---")
    for r in top_rules[:5]:
        conds = "  AND  ".join(r.conditions[:2])
        print(f"  [{r.action:5s}] IF {conds[:60]:<60s}  "
              f"win={r.win_rate:.1%}  ret={r.mean_return:+.4f}")
    print("=" * 60 + "\n")

    return {"json": str(json_path), "markdown": str(md_path)}
