#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = list(csv.DictReader(open("throughput_and_ssd_pages.csv")))
x = [r["variant"] for r in R]
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8), constrained_layout=True)
ax[0].bar(x, [float(r["throughput_k_ops_per_s"]) for r in R], color="#FF0066")
ax[0].set(ylabel="Throughput (K ops/s)", title="(a) Throughput")
k = range(len(x))
ax[1].bar([i - 0.2 for i in k], [float(r["pages_read_per_replacement"]) for r in R], 0.4, color="black", label="Pages read")
ax[1].bar([i + 0.2 for i in k], [float(r["pages_written_per_replacement"]) for r in R], 0.4, color="#FF0066", label="Pages written")
ax[1].set_xticks(k, x)
ax[1].set(ylabel="Pages per Replacement", title="(b) SSD pages")
ax[1].legend()
fig.savefig("figure13.pdf")
