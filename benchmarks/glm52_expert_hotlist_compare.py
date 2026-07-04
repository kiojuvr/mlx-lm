#!/usr/bin/env python3
"""Compare GLM-5.2 routed-expert hotlists.

The parser accepts both mlx_lm's text hotlist format:

    layer expert hits weight

and ds4's generated C initializer format:

    {layer, expert},
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


C_PAIR_RE = re.compile(r"\{\s*(\d+)\s*,\s*(\d+)\s*\}")
TEXT_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)(?:\s+(\d+)(?:\s+([-+0-9.eE]+))?)?\s*$"
)
DEFAULT_TOP_NS = (16, 32, 64, 128, 256, 512, 1024, 4096)


@dataclass(frozen=True)
class HotlistEntry:
    layer: int
    expert: int
    rank: int
    hits: int | None = None
    weight: float | None = None

    @property
    def pair(self) -> tuple[int, int]:
        return (self.layer, self.expert)


def parse_top_ns(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("--top-ns values must be positive")
        values.append(value)
    if not values:
        raise ValueError("--top-ns must include at least one value")
    return values


def parse_hotlist_text(text: str) -> list[HotlistEntry]:
    entries: list[HotlistEntry] = []
    seen: set[tuple[int, int]] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("/*"):
            continue

        c_match = C_PAIR_RE.search(stripped)
        if c_match:
            layer = int(c_match.group(1))
            expert = int(c_match.group(2))
            pair = (layer, expert)
            if pair not in seen:
                seen.add(pair)
                entries.append(HotlistEntry(layer, expert, len(entries) + 1))
            continue

        text_match = TEXT_ROW_RE.match(stripped)
        if text_match:
            layer = int(text_match.group(1))
            expert = int(text_match.group(2))
            hits = int(text_match.group(3)) if text_match.group(3) else None
            weight = float(text_match.group(4)) if text_match.group(4) else None
            pair = (layer, expert)
            if pair not in seen:
                seen.add(pair)
                entries.append(
                    HotlistEntry(layer, expert, len(entries) + 1, hits, weight)
                )
    return entries


def parse_hotlist_file(path: Path) -> list[HotlistEntry]:
    entries = parse_hotlist_text(path.read_text(encoding="utf-8"))
    if not entries:
        raise ValueError(f"{path} did not contain any hotlist entries")
    return entries


def _mean(values: Iterable[int]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def compare_hotlists(
    left: list[HotlistEntry],
    right: list[HotlistEntry],
    *,
    top_ns: Iterable[int] = DEFAULT_TOP_NS,
) -> dict:
    left_rank = {entry.pair: entry.rank for entry in left}
    right_rank = {entry.pair: entry.rank for entry in right}
    rows = []
    for top_n in top_ns:
        left_pairs = [entry.pair for entry in left[:top_n]]
        right_pairs = [entry.pair for entry in right[:top_n]]
        left_set = set(left_pairs)
        right_set = set(right_pairs)
        overlap = left_set & right_set
        union = left_set | right_set
        rows.append(
            {
                "top_n": top_n,
                "left_count": len(left_set),
                "right_count": len(right_set),
                "overlap": len(overlap),
                "left_coverage": len(overlap) / len(left_set) if left_set else 0.0,
                "right_coverage": len(overlap) / len(right_set) if right_set else 0.0,
                "jaccard": len(overlap) / len(union) if union else 0.0,
                "mean_left_rank_in_right": _mean(
                    right_rank[pair] for pair in left_set if pair in right_rank
                ),
                "mean_right_rank_in_left": _mean(
                    left_rank[pair] for pair in right_set if pair in left_rank
                ),
            }
        )
    return {
        "left_entries": len(left),
        "right_entries": len(right),
        "left_unique_layers": len({entry.layer for entry in left}),
        "right_unique_layers": len({entry.layer for entry in right}),
        "rows": rows,
    }


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def format_table(summary: dict) -> str:
    headers = (
        "top_n",
        "left_count",
        "right_count",
        "overlap",
        "left_coverage",
        "right_coverage",
        "jaccard",
        "mean_left_rank_in_right",
        "mean_right_rank_in_left",
    )
    lines = ["\t".join(headers)]
    for row in summary["rows"]:
        lines.append("\t".join(_format_value(row.get(key)) for key in headers))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument(
        "--top-ns",
        default=",".join(str(value) for value in DEFAULT_TOP_NS),
        help="Comma-separated top-N cutoffs to compare.",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    top_ns = parse_top_ns(args.top_ns)
    left = parse_hotlist_file(args.left)
    right = parse_hotlist_file(args.right)
    summary = compare_hotlists(left, right, top_ns=top_ns)
    summary.update(
        {
            "left": str(args.left),
            "right": str(args.right),
            "left_label": args.left_label,
            "right_label": args.right_label,
        }
    )

    print(format_table(summary))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
