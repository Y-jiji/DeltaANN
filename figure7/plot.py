#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR = {"DeltaANN": "#FF0066", "OdinANN": "#0000FF", "CONDA": "#6600CC"}
MARKER = {"DeltaANN": "o", "OdinANN": "^", "CONDA": "D"}


def rows(path):
    return list(csv.DictReader(open(path)))


DATASETS = ["SIFT100M", "DEEP100M", "SPACEV100M", "MSTuring100M"]
RECALL = ["0.90", "0.90", "0.85", "0.80"]
T, Q = rows("throughput.csv"), rows("index_quality.csv")
fig, ax = plt.subplots(2, 4, figsize=(12, 5), constrained_layout=True)
for i, d in enumerate(DATASETS):
    for s in sorted({r["system"] for r in T}, key=list(COLOR).index):
        p = [r for r in T if r["dataset"] == d and r["system"] == s]
        ax[0, i].plot([float(r["hours"]) for r in p], [float(r["throughput_k_ops_per_s"]) for r in p], color=COLOR[s], marker=MARKER[s], markevery=8, fillstyle="none", label=s)
        p = [r for r in Q if r["dataset"] == d and r["system"] == s]
        ax[1, i].plot([float(r["batches_done"]) for r in p], [float(r["qps_at_fixed_recall_k"]) for r in p], color=COLOR[s], marker=MARKER[s], fillstyle="none", label=s)
    ax[0, i].set(title=d, xlabel="Time (hours)", ylabel="Throughput (K ops/s)", ylim=(0, None))
    ax[1, i].set(xlabel="Batches Done", ylabel=f"QPS at Recall {RECALL[i]} (K)", ylim=(0, None))
ax[0, 0].legend()
fig.savefig("figure7.pdf")
