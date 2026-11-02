# /// script
# requires-python = ">=3.11,<3.14"
# ///
# One table per batch and one per checkpoint over #029's variants. usage: summary.py <tag>... <batches.csv>
# <quality.csv>: per tag the files analyze.py and counters.py wrote in this directory, recall_<tag>.csv,
# tput_<tag>.csv and counters_<tag>.csv (a missing file read as no rows); batches.csv rows variant, batch, seconds,
# rate, then insert_<c> for the churn ops' and search_<c> for the searches' mean counters c of COUNTERS; quality.csv
# rows variant, checkpoint, recall50 (recall@10 at beam 50, the mix's beam) and qps90 (QPS at recall@10 TARGET,
# linear between the two beams around it, empty if no beam reaches it)
import csv
import os
import sys

COUNTERS = ["hop", "cold", "dist", "wrote", "full", "sdc", "laten_us"]
TARGET = 0.90
if len(sys.argv) < 4:
    sys.exit("usage: summary.py <tag>... <batches.csv> <quality.csv>")
tags, batches_csv, quality_csv = sys.argv[1:-2], sys.argv[-2], sys.argv[-1]

with open(batches_csv, "w") as out:
    out.write(",".join(["variant", "batch", "seconds", "rate"] + [f"{o}_{c}" for o in ("insert", "search")
                                                                  for c in COUNTERS]) + "\n")
    for tag in tags:
        tput = {}
        if os.path.exists(f"tput_{tag}.csv"):
            tput = {int(r["batch"]): r for r in csv.DictReader(open(f"tput_{tag}.csv"))}
        count = {}
        if os.path.exists(f"counters_{tag}.csv"):
            count = {(int(r["batch"]), r["op"]): r for r in csv.DictReader(open(f"counters_{tag}.csv"))}
        for b in sorted(set(tput) | {b for b, _ in count}):
            cells = [tag, str(b), tput.get(b, {}).get("seconds", ""), tput.get(b, {}).get("rate", "")]
            for op in ("insert", "search"):
                cells += [count.get((b, op), {}).get(c, "") for c in COUNTERS]
            out.write(",".join(cells) + "\n")

with open(quality_csv, "w") as out:
    out.write("variant,checkpoint,recall50,qps90\n")
    for tag in tags:
        if not os.path.exists(f"recall_{tag}.csv"):
            continue
        curves = {}
        for r in csv.DictReader(open(f"recall_{tag}.csv")):
            curves.setdefault(int(r["checkpoint"]), []).append((int(r["beam"]), float(r["recall"]), float(r["qps"])))
        for c, curve in sorted(curves.items()):
            curve.sort()
            at50 = [rec for beam, rec, _ in curve if beam == 50]
            qps = ""
            for (_, r0, q0), (_, r1, q1) in zip(curve, curve[1:]):
                if r0 < TARGET <= r1:
                    qps = f"{q0 + (q1 - q0) * (TARGET - r0) / (r1 - r0):.1f}"
                    break
            out.write(f"{tag},{c},{at50[0] if at50 else ''},{qps}\n")
