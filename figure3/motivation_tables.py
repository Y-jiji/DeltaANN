#!/usr/bin/env python3
"""Rebuild the plotted values of Figures 1, 2 and 3 from the raw experiment outputs.

Standard library only. Each command writes one CSV to stdout:

    motivation_tables.py figure1 <recallqps.csv>
    motivation_tables.py figure2 <directory holding twohop017_<dataset>_d64.csv>
    motivation_tables.py figure3 <directory holding recallqps019_<dataset>.csv>
"""

import bisect
import csv
import math
import os
import sys

DATASETS = ("sift", "deep", "spacev", "msturing", "t2i")

# Figure 1 draws a line through most beams and a marker on every fourth beam. These beams appear
# in neither. The list was read from the figure file and cannot be derived from the raw CSV.
NOT_DRAWN = {
    ("sift", "original"): (70, 180, 200, 225, 275, 300, 375, 1000),
    ("sift", "made-symmetric"): (180, 200, 225, 275, 300, 1000),
    ("deep", "original"): (180, 200, 225, 275, 300, 375, 1000),
    ("deep", "made-symmetric"): (200, 275, 300, 1000),
    ("spacev", "original"): (180, 200, 225, 275, 300, 375, 1000),
    ("spacev", "made-symmetric"): (200, 275, 300, 1000),
    ("msturing", "original"): (180, 200, 225, 275, 300, 375, 750, 1000),
    ("msturing", "made-symmetric"): (200, 275, 300, 1000),
    ("t2i", "original"): (200, 225, 275, 300, 1000),
    ("t2i", "made-symmetric"): (200, 275, 300, 1000),
}
GRAPH_NAMES = (("orig", "original"), ("bidir", "made-symmetric"))

# Figure 2: the tally covers this many inserted points per dataset.
POINTS = 10_000

# Figure 3: the panels, and each line as (graph, codebook letter, prune label, codebook label).
# Codebook A is the old one and B the new one. Graph BB searched with B is the 1x baseline.
PANELS = (("SPACEV", "spacev"), ("MSTuring", "msturing"))
LINES = (
    ("AB", "A", "old PQ re-prune", "old codebook"),
    ("AB", "B", "old PQ re-prune", "new codebook"),
    ("AE", "B", "exact re-prune", "new codebook"),
)


def writer():
    """A CSV writer on stdout with plain newlines."""
    return csv.writer(sys.stdout, lineterminator="\n")


def figure1(path):
    """Rows of the raw recall-QPS CSV that Figure 1 draws, values unchanged."""
    with open(path, newline="") as source:
        rows = list(csv.DictReader(source))
    out = writer()
    out.writerow(("dataset", "graph", "beam", "recall", "qps"))
    for dataset in DATASETS:
        for raw, name in GRAPH_NAMES:
            kept = [r for r in rows if r["ds"] == dataset and r["graph"] == raw
                    and int(r["beam"]) not in NOT_DRAWN[(dataset, name)]]
            for row in sorted(kept, key=lambda r: int(r["beam"])):
                out.writerow((dataset, name, row["beam"], row["recall"], row["qps"]))


def figure2(directory):
    """Average candidates per insert in each stacked segment of Figure 2, two decimals."""
    out = writer()
    out.writerow(("dataset", "selected_navigable_from_selected", "selected_other", "discarded"))
    for dataset in DATASETS:
        navigable = other = discarded = 0
        with open(os.path.join(directory, f"twohop017_{dataset}_d64.csv"), newline="") as source:
            for row in csv.DictReader(source):
                if row["status"] == "kept":
                    navigable += int(row["fromkept"])
                    other += int(row["viapruned"])
                else:
                    discarded += int(row["nodes"])
        out.writerow((dataset, f"{navigable / POINTS:.2f}", f"{other / POINTS:.2f}",
                      f"{discarded / POINTS:.2f}"))


def curve(rows, graph, book):
    """(recall, QPS) of a graph searched with a codebook by PQ traversal, by rising recall."""
    return sorted((float(r["recall"]), float(r["qps"])) for r in rows
                  if r["graph"] == graph and r["book"] == book and r["mode"] == "pq")


def baseline_qps(recall, xs, ys):
    """QPS of the baseline at `recall`: piecewise linear in recall over log QPS (natural log)."""
    if recall == xs[-1]:
        return math.exp(ys[-1])
    j = bisect.bisect_right(xs, recall) - 1
    return math.exp((ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]) * (recall - xs[j]) + ys[j])


def figure3(directory):
    """Each line's QPS over the baseline's QPS at the same recall, at the line's own recalls.

    A point whose recall lies outside the baseline's recall range is not plotted.
    """
    out = writer()
    out.writerow(("dataset", "prune", "codebook", "recall", "relative_qps"))
    for panel, dataset in PANELS:
        with open(os.path.join(directory, f"recallqps019_{dataset}.csv"), newline="") as source:
            rows = list(csv.DictReader(source))
        base = curve(rows, "BB", "B")
        xs = [recall for recall, _ in base]
        ys = [math.log(qps) for _, qps in base]
        for graph, book, prune, codebook in LINES:
            for recall, qps in curve(rows, graph, book):
                if xs[0] <= recall <= xs[-1]:
                    relative = qps / baseline_qps(recall, xs, ys)
                    out.writerow((panel, prune, codebook, f"{recall:.4f}", f"{relative:.4f}"))


COMMANDS = {"figure1": figure1, "figure2": figure2, "figure3": figure3}

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    COMMANDS[sys.argv[1]](sys.argv[2])
