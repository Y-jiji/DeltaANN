#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = list(csv.DictReader(open("recall_at_l100.csv")))
x = [int(r["batches_done"]) for r in R]
fig, ax = plt.subplots(figsize=(5, 2.4), constrained_layout=True)
ax.plot(x, [float(r["deltaann"]) for r in R], color="#FF0066", marker="o", fillstyle="none", label="DeltaANN")
ax.plot(x, [float(r["deltaann_without_prunesearch"]) for r in R], color="#FF0066", marker="s", label="DeltaANN w/o PruneSearch")
ax.plot(x, [float(r["odinann"]) for r in R], color="#0000FF", marker="^", fillstyle="none", label="OdinANN")
ax.axvline(99, color="black", lw=0.6)
ax.set(xlabel="Batches Done", ylabel="Recall@10 at L = 100")
ax.legend(fontsize=7)
fig.savefig("figure12.pdf")
