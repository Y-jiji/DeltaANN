#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = list(csv.DictReader(open("relative_qps.csv")))
SERIES = [("old PQ re-prune", "old codebook", "#FF0066", "s", "full"), ("old PQ re-prune", "new codebook", "#FF0066", "s", "none"), ("exact re-prune", "new codebook", "black", "o", "none")]
fig, ax = plt.subplots(1, 2, figsize=(6, 2.8), constrained_layout=True)
for a, d in zip(ax, ["SPACEV", "MSTuring"]):
    for prune, book, c, m, fill in SERIES:
        p = [r for r in R if r["dataset"] == d and r["prune"] == prune and r["codebook"] == book]
        a.plot([float(r["recall"]) for r in p], [float(r["relative_qps"]) for r in p], color=c, marker=m, ms=4, fillstyle=fill, label=f"{prune}, {book}")
    a.axhline(1, color="black", lw=0.8)
    a.set(title=d, xlabel="Recall@10", ylim=(0.25, 2))
ax[0].set_ylabel("Relative QPS")
ax[0].legend(fontsize=6)
fig.savefig("figure3.pdf")
