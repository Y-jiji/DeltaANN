#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = list(csv.DictReader(open("ab_repair_budget.csv")))
B = list(csv.DictReader(open("cd_mask_margin.csv")))
C = list(csv.DictReader(open("ef_tasks_in_flight.csv")))
RED = dict(color="#FF0066", marker="o", fillstyle="none")
fig, ax = plt.subplots(3, 2, figsize=(7, 7), constrained_layout=True)
c = [float(r["c"]) for r in A]
ax[0, 0].plot(c, [float(r["throughput_k_ops_per_s"]) for r in A], **RED)
ax[0, 0].set(xlabel="Repair Budget Factor C", ylabel="Throughput (K ops/s)", ylim=(0, None), title="(a) Throughput")
ax[0, 1].plot(c, [float(r["reprunes_per_replacement_at_batch_10"]) for r in A], label="DeltaANN", **RED)
ax[0, 1].plot(c, [float(r["bound_2_over_c_minus_1"]) for r in A], color="black", marker="o", fillstyle="none", label="Bound 2/(C-1)")
ax[0, 1].set(xlabel="Repair Budget Factor C", ylabel="Re-prunes per Replacement", yscale="log", title="(b) Re-prunes")
ax[0, 1].legend()
b = [float(r["beta"]) for r in B]
ax[1, 0].plot(b, [float(r["pages_read_per_replacement"]) for r in B], **RED)
ax[1, 0].set(xlabel="Mask Margin beta", ylabel="Pages per Replacement", ylim=(0, None), title="(c) SSD reads")
ax[1, 1].plot(b, [float(r["qps_at_recall_0_90"]) for r in B], **RED)
ax[1, 1].set(xlabel="Mask Margin beta", ylabel="QPS at Recall 0.9", ylim=(0, None), title="(d) Search QPS")
t = [int(r["tasks_per_thread"]) for r in C]
ax[2, 0].plot(t, [float(r["throughput_k_ops_per_s"]) for r in C], **RED)
ax[2, 0].set(xlabel="Tasks in Flight per Thread", ylabel="Throughput (K ops/s)", xscale="log", xticks=t, xticklabels=t, ylim=(0, None), title="(e) Throughput")
ax[2, 1].plot(t, [float(r["replacement_latency_ms"]) for r in C], label="Replacement", **RED)
ax[2, 1].plot(t, [float(r["search_latency_ms"]) for r in C], color="black", marker="s", fillstyle="none", label="Search")
ax[2, 1].set(xlabel="Tasks in Flight per Thread", ylabel="Latency (ms)", xscale="log", xticks=t, xticklabels=t, ylim=(0, None), title="(f) Latency")
ax[2, 1].legend()
fig.savefig("figure14.pdf")
