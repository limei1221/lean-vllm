"""Mean, min and max across repeated sweeps that benchmarks/repeat_sweep.sh wrote, per engine and offered load.

uv run python benchmarks/aggregate_reps.py results/.../deepseek [more dirs...]
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = [
    ("goodput", "goodput", "{:.2f}"),
    ("slo_goodput", "slo_good", "{:.2f}"),
    ("output_tok_s", "tok/s", "{:.0f}"),
    ("ttft_p50", "ttft_p50", "{:.3f}"),
    ("ttft_p99", "ttft_p99", "{:.3f}"),
    ("tpot_p50", "tpot_p50", "{:.4f}"),
    ("itl_p99", "itl_p99", "{:.4f}"),
    ("e2e_p99", "e2e_p99", "{:.2f}"),
]


def load(directory: Path) -> dict[tuple[str, float], list[dict]]:
    """{(engine, rate): [row per repetition]} from every summary.json below directory."""
    points = defaultdict(list)
    for path in sorted(directory.glob("**/summary.json")):
        summary = json.loads(path.read_text())
        for row in summary["rows"]:
            points[(summary["engine"], row["request_rate"])].append(row)
    return points


def cell(values: list[float], spec: str) -> str:
    """Mean, and the range when there are repetitions."""
    mean = statistics.mean(values)
    if len(values) == 1:
        return spec.format(mean)
    return f"{spec.format(mean)} ({spec.format(min(values))}–{spec.format(max(values))})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.dirs:
        points = load(directory)
        print(f"\n### {directory}\n")
        print("| engine | rate | n | " + " | ".join(heading for _, heading, _ in METRICS) + " |")
        print("| --- | ---: | ---: | " + " | ".join("---:" for _ in METRICS) + " |")
        for (engine, rate), rows in sorted(points.items(), key=lambda item: (item[0][1], item[0][0])):
            cells = []
            for key, _, spec in METRICS:
                values = [row[key] for row in rows if row.get(key) is not None]
                cells.append(cell(values, spec) if values else "--")
            incomplete = sum(row["completed"] < (200 if rate == 1 else 1000) for row in rows)
            note = f" ({incomplete} incomplete)" if incomplete else ""
            print(f"| {engine}{note} | {rate:g} | {len(rows)} | " + " | ".join(cells) + " |")
        # Engine gap per rate on the mean goodput.
        rates = sorted({rate for _, rate in points})
        for rate in rates:
            lean, vllm = points.get(("lean-vllm", rate)), points.get(("vllm", rate))
            if lean and vllm:
                gap = statistics.mean(r["goodput"] for r in lean) / statistics.mean(r["goodput"] for r in vllm) - 1
                tpot = statistics.mean(r["tpot_p50"] for r in lean) / statistics.mean(r["tpot_p50"] for r in vllm) - 1
                print(f"- rate {rate:g}: lean-vLLM goodput {gap:+.1%}, median TPOT {tpot:+.1%} vs vLLM")


if __name__ == "__main__":
    main()
