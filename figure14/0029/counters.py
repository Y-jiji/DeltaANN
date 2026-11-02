# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy", "pyarrow"]
# ///
# Profile counters per churn batch and op of one #029 instantmix run. usage: counters.py <log> <counters.csv>: the log
# the Arrow IPC stream instantmix wrote (read up to its last whole batch, as analyze.py does); rows, one per batch b
# (stage `mix_<b-1>`) and op (`insert`, a churn op: a removal and the insert into its slot, or `search`, a L=50 search
# beside them), of the op rows kept (one of every RATE): batch, op, rows, then the mean of each counter of COUNTERS
# over them, and laten_us the mean latency in microseconds
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

COUNTERS = ["hop", "cold", "dist", "estim", "pass", "mask", "adc", "wrote", "full", "sdc"]
if len(sys.argv) != 3:
    sys.exit("usage: counters.py <log> <counters.csv>")
log, out_csv = sys.argv[1], sys.argv[2]

sums: dict[tuple[int, str], np.ndarray] = {}
reader = pa.ipc.open_stream(pa.memory_map(log))
while True:
    try:
        batch = reader.read_next_batch()
    except (StopIteration, pa.ArrowInvalid, OSError):
        break
    batch = batch.filter(pc.starts_with(batch.column("tag"), "mix_"))
    if not batch.num_rows:
        continue
    tags = batch.column("tag").to_numpy(zero_copy_only=False).astype(str)
    ops = batch.column("op").to_numpy(zero_copy_only=False).astype(str)
    cols = np.stack([batch.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
                     for c in ["laten", *COUNTERS]], axis=1)
    for tag in np.unique(tags):
        b = int(re.fullmatch(r"mix_(\d+)", tag)[1]) + 1
        for op in ("insert", "search"):
            pick = (tags == tag) & (ops == op)
            if not pick.any():
                continue
            row = np.concatenate([[pick.sum()], cols[pick].sum(axis=0)])
            sums[(b, op)] = sums.get((b, op), 0) + row

with open(out_csv, "w") as out:
    out.write(",".join(["batch", "op", "rows", "laten_us", *COUNTERS]) + "\n")
    for (b, op), row in sorted(sums.items()):
        n = row[0]
        means = row[1:] / n
        out.write(f"{b},{op},{int(n)},{means[0] / 1e3:.1f}," + ",".join(f"{m:.3f}" for m in means[1:]) + "\n")
