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


A, B = rows("a_insert_delete_search.csv"), rows("b_insert_search.csv")
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8), constrained_layout=True)
for s in ("DeltaANN", "OdinANN"):
    p = [r for r in A if r["system"] == s]
    ax[0].plot([float(r["hours"]) for r in p], [float(r["throughput_k_ops_per_s"]) for r in p], color=COLOR[s], marker=MARKER[s], markevery=8, fillstyle="none", label=s)
    p = [r for r in B if r["system"] == s]
    ax[1].plot([int(r["batch"]) for r in p], [float(r["throughput_k_ops_per_s"]) for r in p], color=COLOR[s], marker=MARKER[s], markevery=10, fillstyle="none", label=s)
ax[0].set(xlabel="Time (hours)", ylabel="Throughput (K ops/s)", ylim=(0, 14), title="(a) Insert-Delete-Search")
ax[1].set(xlabel="Batches Done", ylabel="Throughput (K ops/s)", ylim=(0, 14), title="(b) Insert-Search")
ax[0].legend()
fig.savefig("figure11.pdf")
