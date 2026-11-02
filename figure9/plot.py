#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def bars(a, x, series):
    w = 0.8 / len(series)
    for i, (label, v, c) in enumerate(series):
        a.bar([k + (i - (len(series) - 1) / 2) * w for k in range(len(x))], v, w, color=c, label=label)
    a.set_xticks(range(len(x)), x)


T = list(csv.DictReader(open("a_odinann_time_per_batch.csv")))
S = list(csv.DictReader(open("b_ssd_reads_per_delete.csv")))
x = [r["dataset"] for r in T]
rest = [float(r["rest_of_batch_s"]) for r in T]
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8), constrained_layout=True)
ax[0].bar(x, rest, color="#9999FF", label="Rest of Batch")
ax[0].bar(x, [float(r["merge_s"]) for r in T], bottom=rest, color="#0000FF", label="Merge")
ax[0].set(ylabel="Time per Batch (s)", title="(a) OdinANN's batch time")
ax[0].legend()
bars(ax[1], x, [("DeltaANN", [float(r["deltaann"]) for r in S], "#FF0066"), ("CONDA", [float(r["conda"]) for r in S], "#6600CC")])
ax[1].set(ylabel="SSD Reads per Delete", title="(b) SSD reads")
ax[1].legend()
fig.savefig("figure9.pdf")
