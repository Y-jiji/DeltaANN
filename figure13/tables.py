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


def fig13_14():
    B = {}
    for r in csv.DictReader(open(p("batches.csv"))):
        B.setdefault(r["variant"], {})[int(r["batch"])] = float(r["seconds"])
    Q = {(r["variant"], int(r["checkpoint"])): float(r["qps90"]) for r in csv.DictReader(open(p("quality.csv")))}

    def C(t, op, col, n):
        v = [(float(r[col]), float(r["rows"])) for r in csv.DictReader(open(p(f"counters_{t}.csv"))) if r["op"] == op and int(r["batch"]) <= n]
        return sum(a * b for a, b in v) / sum(b for _, b in v)

    def last(t, col):
        return [float(r[col]) for r in csv.DictReader(open(p(f"counters_{t}.csv"))) if r["op"] == "insert"][-1]

    def thr(t, n):
        return 3 * n * 1e6 / sum(B[t][b] for b in range(1, n + 1)) / 1e3

    out("throughput_and_ssd_pages.csv", ["variant", "throughput_k_ops_per_s", "pages_read_per_replacement", "pages_written_per_replacement"],
        [[a, f"{thr(t, 2):.3f}", f"{C(t, 'insert', 'cold', 2):.1f}", f"{C(t, 'insert', 'wrote', 2):.1f}"] for a, t in (("Full", "full"), ("No PS", "blunt"), ("No BR", "inf"), ("Neither", "both"))])


fig13_14()
