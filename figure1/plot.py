#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = list(csv.DictReader(open("recall_qps.csv")))
E = {r["dataset"]: r["edges_removed_percent"] for r in csv.DictReader(open("edges_removed.csv"))}
fig, ax = plt.subplots(1, 5, figsize=(13, 2.8), constrained_layout=True)
for a, d in zip(ax, ["sift", "deep", "spacev", "msturing", "t2i"]):
    for g, c, m, label in (("original", "#555555", "^", "original"), ("made-symmetric", "#FF0066", "s", f"-{E[d]}% edges")):
        p = [r for r in R if r["dataset"] == d and r["graph"] == g]
        a.plot([float(r["recall"]) for r in p], [float(r["qps"]) for r in p], color=c, marker=m, ms=3, label=label)
    a.set(title=d, xlabel="Recall@10", ylim=(0, None))
    a.legend()
ax[0].set_ylabel("Throughput (QPS)")
fig.savefig("figure1.pdf")
