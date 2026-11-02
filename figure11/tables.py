#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow"]
# ///
"""usage: tables.py <runs dir> <out dir>"""
import csv, math, os, re, sys

RUNS, OUT = sys.argv[1], sys.argv[2]
DS = [("sift", "SIFT"), ("deep", "DEEP"), ("spacevs", "SPACEV"), ("msturings", "MSTuring")]
LEVEL = {"sift": 0.90, "deep": 0.90, "spacevs": 0.85, "msturings": 0.80}


def p(name):
    return os.path.join(RUNS, name)


def out(path, header, rows):
    path = os.path.join(OUT, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(path, len(rows))


def tp(name):
    return {int(r["batch"]): float(r["seconds"]) for r in csv.DictReader(open(p(f"tp_{name}.csv")))}


def sweeps(name):
    s = {}
    for r in csv.DictReader(open(p(f"recall_{name}.csv"))):
        s.setdefault(int(r["checkpoint"]), []).append((int(r["beam"]), float(r["recall"]), float(r["qps"])))
    return s


def level(sweep, t):
    """QPS at recall t: log-linear between the first pair of beams spanning t."""
    rows = sorted(sweep)
    for (_, r0, q0), (_, r1, q1) in zip(rows, rows[1:]):
        if r0 < t <= r1:
            return math.exp(math.log(q0) + (t - r0) / (r1 - r0) * (math.log(q1) - math.log(q0)))
    return None


def counters(name, n, cols):
    """Mean of each sampled insert's counters over the first n batches."""
    import numpy as np
    import pyarrow as pa
    src = pa.memory_map(p(f"{name}.arrows"))
    try:
        t = pa.ipc.open_file(src).read_all()
    except pa.ArrowInvalid:
        src.seek(0)
        t = pa.ipc.open_stream(src).read_all()
    tags = t["tag"].to_numpy(zero_copy_only=False).astype(str)
    ops = t["op"].to_numpy(zero_copy_only=False).astype(str)
    b = np.array([int(re.search(r"_(\d+)$", x)[1]) + 1 for x in tags])
    res = {}
    for op in ("insert", "search"):
        keep = (b <= n) & (ops == op)
        res[op] = {c: float(np.nan_to_num(t[c].to_numpy(zero_copy_only=False).astype(float))[keep].mean()) if c in t.column_names else 0.0 for c in cols}
    return res


def fig11():
    rows = []
    for system, run in (("DeltaANN", "ourschurn_sift1b"), ("OdinANN", "odinchurn_sift1b")):
        t = 0.0
        for b, s in sorted(tp(run).items()):
            t += s
            rows.append([system, b, f"{t / 3600:.4f}", f"{3e6 / s / 1e3:.4f}"])
    out("a_insert_delete_search.csv", ["system", "batch", "hours", "throughput_k_ops_per_s"], rows)
    rows = []
    for system, run in (("DeltaANN", "oursgrow_sift1b"), ("OdinANN", "odingrow_sift1b")):
        rows += [[system, b, f"{2e6 / s / 1e3:.4f}"] for b, s in sorted(tp(run).items())]
    out("b_insert_search.csv", ["system", "batch", "throughput_k_ops_per_s"], rows)


fig11()
