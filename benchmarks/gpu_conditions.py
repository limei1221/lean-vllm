"""Mean busy SM clock, temperature and power per engine, from repeat_sweep.sh's nvidia-smi log and run timestamps.

uv run python benchmarks/gpu_conditions.py results/.../deepseek
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def number(text: str) -> float:
    return float(text.strip().split()[0])


for directory in map(Path, sys.argv[1:]):
    samples = []
    with open(directory / "nvidia-smi.log") as handle:
        for row in csv.DictReader(handle, skipinitialspace=True):
            when = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
            samples.append(
                (
                    when,
                    number(row["utilization.gpu [%]"]),
                    number(row["clocks.current.sm [MHz]"]),
                    number(row["temperature.gpu"]),
                    number(row["power.draw [W]"]),
                )
            )
    per_engine = defaultdict(list)
    for path in directory.glob("**/summary.json"):
        summary = json.loads(path.read_text())
        for run in summary["rows"]:
            start, end = datetime.fromisoformat(run["started_at"]), datetime.fromisoformat(run["finished_at"])
            # Busy samples only: the clock idles down between requests at low load.
            per_engine[summary["engine"]] += [s for s in samples if start <= s[0] <= end and s[1] >= 50]
    print(f"### {directory}")
    for engine, rows in sorted(per_engine.items()):
        clock, temp, power = (statistics.mean(r[i] for r in rows) for i in (2, 3, 4))
        print(f"- {engine}: {len(rows)} busy samples, SM clock {clock:.0f} MHz, {temp:.0f} C, {power:.0f} W")
