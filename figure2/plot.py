#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = list(csv.DictReader(open("candidates_per_insert.csv")))[::-1]
y = [r["dataset"] for r in R]
a0 = [float(r["selected_navigable_from_selected"]) for r in R]
a1 = [float(r["selected_other"]) for r in R]
a2 = [float(r["discarded"]) for r in R]
fig, ax = plt.subplots(figsize=(5, 2.6), constrained_layout=True)
ax.barh(y, a0, color="#00CCFF", label="selected, navigable from selected")
ax.barh(y, a1, left=a0, color="#004D61", label="other selected")
ax.barh(y, a2, left=[u + v for u, v in zip(a0, a1)], facecolor="white", edgecolor="#FF0000", hatch="/////", label="discarded")
ax.set(xlabel="Candidates per Insert")
ax.legend(ncols=3, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1))
fig.savefig("figure2.pdf")
