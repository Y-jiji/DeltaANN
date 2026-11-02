# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy", "pyarrow"]
# ///
# Recall@10 and QPS per checkpoint and beam, and time per batch, of one #029 instantmix run, #028/analyze.py as is.
# out linearly. usage: analyze.py <log> <gt.arrows> <M> <recall.csv> <tput.csv>: the log the Arrow IPC stream
# instantmix wrote (read up to its last whole batch, so a log still being written reads as its finished sweeps), the
# ground truth gtbrute's top-10 of every BLOCK rows in block order, M the base's span in blocks; a churn log (any
# `mix_<b>` stage) scores checkpoint c against blocks [c, c + M), a grow log (`grow_<b>` stages) against [0, M + c)
import os
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

TOPK = 10
BLOCK = int(os.environ.get("BLOCK", 1_000_000))
NONE = int(np.iinfo(np.uint64).max)
if len(sys.argv) != 6:
    sys.exit("usage: analyze.py <log> <gt.arrows> <M> <recall.csv> <tput.csv>")
log, truth, span, recall_csv, tput_csv = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]

# [1] the log: stage times, the kind of run, the sweeps' hits by query
laten: dict[str, int] = {}
stages: list[tuple[int, float, int]] = []
kinds: set[str] = set()
held: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
reader = pa.ipc.open_stream(pa.memory_map(log))
while True:
    try:
        batch = reader.read_next_batch()
    except (StopIteration, pa.ArrowInvalid, OSError):
        break
    op = batch.column("op")
    stage = batch.filter(pc.equal(op, "stage"))
    for tag, spent, rss in zip(*(stage.column(c).to_pylist() for c in ("tag", "laten", "maxrss"))):
        laten[tag] = spent
        hit = re.fullmatch(r"(mix|grow)_(\d+)", tag)
        if hit:
            kinds.add(hit[1])
            stages.append((int(hit[2]) + 1, spent / 1e9, -1 if rss is None else rss))
    batch = batch.filter(pc.equal(op, "search"))
    if not batch.num_rows:
        continue
    names = batch.column("tag").to_numpy(zero_copy_only=False)
    keep = ~np.char.startswith(names.astype(str), "mix_") & ~np.char.startswith(names.astype(str), "grow_")
    if not keep.any():
        continue
    batch = batch.filter(pa.array(keep))
    names = names[keep]
    index = batch.column("index").to_numpy()
    cut = pc.list_slice(batch.column("results"), 0, TOPK)
    lengths = pc.fill_null(pc.list_value_length(cut), 0).to_numpy(zero_copy_only=False)
    values = pc.list_flatten(cut).field("value").to_numpy(zero_copy_only=False).astype(np.uint64)
    hits = np.full((len(lengths), TOPK), NONE, dtype=np.uint64)
    hits[np.repeat(np.arange(len(lengths)), lengths),
         np.arange(len(values)) - np.repeat(np.cumsum(lengths) - lengths, lengths)] = values
    for name in np.unique(names):
        pick = names == name
        held.setdefault(str(name), []).append((index[pick], hits[pick]))
churn = "mix" in kinds
stages.sort()
sweeps: list[tuple[int, int, np.ndarray, float]] = []
for name, parts in held.items():
    hit = re.fullmatch(r"sweep_b(\d+)_k(\d+)_r\d+", name)
    assert hit, f"{log}: search tag {name}"
    index = np.concatenate([p[0] for p in parts])
    rows = np.concatenate([p[1] for p in parts])
    sweeps.append((int(hit[1]), int(hit[2]), rows[np.argsort(index)], len(index) / (laten[name] / 1e9)))
last = max((c for c, _, _, _ in sweeps), default=0)

# [2] the ground truth's blocks 0 to M + last - 1, shaped (block, query, TOPK)
ids_list, dists_list = [], []
for gt in pa.ipc.open_stream(pa.memory_map(truth)):
    if len(ids_list) == span + last:
        break
    block = gt.column("block").to_numpy()
    assert (block == len(ids_list)).all(), f"{truth}: block {block[0]} where {len(ids_list)} is due"
    assert (gt.column("query").to_numpy() == np.arange(gt.num_rows)).all(), f"{truth}: query order"
    ids_list.append(gt.column("rows").flatten().to_numpy().reshape(-1, TOPK))
    dists_list.append(gt.column("dists").flatten().to_numpy().reshape(-1, TOPK))
assert len(ids_list) == span + last, f"{truth}: {len(ids_list)} blocks, want {span + last}"
ids, dists = np.stack(ids_list), np.stack(dists_list)
del ids_list, dists_list

# [3] recall per (checkpoint, beam): the live blocks' merged top-10, each sweep pass's hits against it
live: dict[int, np.ndarray] = {}
passes: dict[tuple[int, int], list[tuple[float, float]]] = {}
for c, beam, found_rows, qps in sweeps:
    if c not in live:
        lo, hi = (c, c + span) if churn else (0, span + c)
        cand = np.concatenate(list(ids[lo:hi]), axis=1)
        near = np.concatenate(list(dists[lo:hi]), axis=1)
        live[c] = np.take_along_axis(cand, np.argpartition(near, TOPK - 1, axis=1)[:, :TOPK], axis=1)
        del cand, near
    assert len(found_rows) == live[c].shape[0], f"checkpoint {c} beam {beam}: {len(found_rows)} queries"
    found = (found_rows[:, :, None] == live[c][:, None, :]).any(axis=2).sum(axis=1)
    passes.setdefault((c, beam), []).append((float(found.mean()) / TOPK, qps))

# [4] the two CSVs
with open(recall_csv, "w") as out:
    out.write("checkpoint,beam,recall,qps\n")
    for (c, beam), runs in sorted(passes.items()):
        out.write(f"{c},{beam},{np.mean([r for r, _ in runs]):.5f},{np.median([q for _, q in runs]):.1f}\n")
with open(tput_csv, "w") as out:
    out.write("batch,stage,seconds,rate,maxrss\n")
    for b, seconds, rss in stages:
        out.write(f"{b},batch,{seconds:.3f},{BLOCK / seconds:.1f},{'' if rss < 0 else rss}\n")
