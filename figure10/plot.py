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


H = list(csv.DictReader(open("a_hops_per_insert.csv")))
D = list(csv.DictReader(open("b_distances_per_insert.csv")))
x = [r["dataset"] for r in H]
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8), constrained_layout=True)
bars(ax[0], x, [("DeltaANN", [float(r["deltaann"]) for r in H], "#FF0066"), ("OdinANN", [float(r["odinann"]) for r in H], "#0000FF")])
ax[0].set(ylabel="Hops per Insert", title="(a) Hops")
ax[0].legend()
bars(ax[1], x, [("DeltaANN PQ", [float(r["deltaann_pq"]) for r in D], "#FF0066"), ("DeltaANN exact", [float(r["deltaann_exact"]) for r in D], "#FF99C2"), ("OdinANN PQ", [float(r["odinann_pq"]) for r in D], "#0000FF"), ("OdinANN exact", [float(r["odinann_exact"]) for r in D], "#9999FF")])
ax[1].set(ylabel="Distances per Insert", yscale="log", ylim=(10, 100000), title="(b) Distances")
ax[1].legend(fontsize=6)
fig.savefig("figure10.pdf")
