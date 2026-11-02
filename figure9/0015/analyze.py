# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy", "pyarrow"]
# ///
"""Recall@10 and QPS per checkpoint and beam, and time per batch, of one #015 mix run.

usage: analyze.py <log> <gt.arrows> <M> <recall.csv> <tput.csv>, the log an Arrow IPC stream odinmix, freshmix,
fnctmix or instantmix wrote, the ground truth one `gtbrute` wrote, M the base's span in BLOCK rows.
"""

import os
import re
import sys
from collections.abc import Iterator
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

TOPK: Final[int] = 10
BLOCK: Final[int] = int(os.environ.get("BLOCK", 1_000_000))
NONE: Final[int] = int(np.iinfo(np.uint64).max)


def blocks(path: str, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows and distances of the top-TOPK of blocks 0 to count - 1 in the ground truth stream `path`, one batch
    per block of rows [b * BLOCK, (b + 1) * BLOCK) in block order, shaped (block, query, TOPK), rows global."""
    ids, dists = [], []
    for batch in pa.ipc.open_stream(pa.memory_map(path)):
        if len(ids) == count:
            break
        block = batch.column("block").to_numpy()
        assert (block == len(ids)).all(), f"{path}: block {block[0]} where {len(ids)} is due"
        assert (batch.column("query").to_numpy() == np.arange(batch.num_rows)).all(), f"{path}: query order"
        ids.append(batch.column("rows").flatten().to_numpy().reshape(-1, TOPK))
        dists.append(batch.column("dists").flatten().to_numpy().reshape(-1, TOPK))
    assert len(ids) == count, f"{path}: {len(ids)} blocks, want {count}"
    return np.stack(ids), np.stack(dists)


def window(ids: np.ndarray, dists: np.ndarray, first: int) -> np.ndarray:
    """Top-TOPK rows per query over blocks `first` to the last of `ids`, shaped (query, TOPK)."""
    cand = np.concatenate(list(ids[first:]), axis=1)
    near = np.concatenate(list(dists[first:]), axis=1)
    pick = np.argpartition(near, TOPK - 1, axis=1)[:, :TOPK]
    return np.take_along_axis(cand, pick, axis=1)


def matrix(lists: pa.Array) -> np.ndarray:
    """The `value` of the first TOPK hits of each result list, padded with NONE."""
    cut = pc.list_slice(lists, 0, TOPK)
    lengths = pc.fill_null(pc.list_value_length(cut), 0).to_numpy(zero_copy_only=False)
    values = pc.list_flatten(cut).field("value").to_numpy(zero_copy_only=False).astype(np.uint64)
    out = np.full((len(lengths), TOPK), NONE, dtype=np.uint64)
    row = np.repeat(np.arange(len(lengths)), lengths)
    col = np.arange(len(values)) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    out[row, col] = values
    return out


def stream(path: str) -> Iterator[pa.RecordBatch]:
    """The record batches of the Arrow IPC stream `path`, read up to its last whole batch, so a log still being
    written or cut off reads as its finished runs."""
    reader = pa.ipc.open_stream(pa.memory_map(path))
    while True:
        try:
            yield reader.read_next_batch()
        except (StopIteration, pa.ArrowInvalid, OSError):
            return


def scan(path: str) -> tuple[bool, list[tuple[int, float, int]], list[tuple[int, int, np.ndarray, float]]]:
    """Whether the log `path` replaces rows (any `mix_<b>` stage) or only grows (`grow_<b>`), its batches as
    (b + 1, seconds, maxrss or -1), and its sweeps `sweep_b<c>_k<k>_r<r>` as (c, k, hits by query, QPS)."""
    laten: dict[str, int] = {}
    stages: list[tuple[int, float, int]] = []
    kinds: set[str] = set()
    held: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for batch in stream(path):
        op = batch.column("op")
        stage = batch.filter(pc.equal(op, "stage"))
        for tag, spent, rss in zip(*(stage.column(c).to_pylist() for c in ("tag", "laten", "maxrss"))):
            laten[tag] = spent
            hit = re.fullmatch(r"(mix|grow)_(\d+)", tag)
            if hit:
                kinds.add(hit[1])
                stages.append((int(hit[2]) + 1, spent / 1e9, -1 if rss is None else rss))
        batch = batch.filter(pc.equal(op, "search"))
        names = batch.column("tag").to_numpy(zero_copy_only=False)
        index = batch.column("index").to_numpy()
        hits = matrix(batch.column("results")) if batch.num_rows else np.empty((0, TOPK), np.uint64)
        for name in np.unique(names):
            if not name.startswith(("mix_", "grow_")):
                pick = names == name
                held.setdefault(name, []).append((index[pick], hits[pick]))
    sweeps = []
    for name, parts in held.items():
        hit = re.fullmatch(r"sweep_b(\d+)_k(\d+)_r\d+", name)
        assert hit, f"{path}: search tag {name}"
        index = np.concatenate([p[0] for p in parts])
        rows = np.concatenate([p[1] for p in parts])
        sweeps.append((int(hit[1]), int(hit[2]), rows[np.argsort(index)], len(index) / (laten[name] / 1e9)))
    return "mix" in kinds, sorted(stages), sweeps


def recall(sweeps: list[tuple[int, int, np.ndarray, float]], churn: bool, truth: str,
           span: int, grow: int) -> list[tuple[int, int, float, float]]:
    """(checkpoint, beam, mean recall, median QPS) of `sweeps` against blocks 0 to span + C - 1 of the ground
    truth `truth`, C the last checkpoint, rows [c - grow, c + span) blocks live at checkpoint c under `churn`, from
    0 while c is under `grow`, else [0, span + c)."""
    ids, dists = blocks(truth, span + max((c for c, _, _, _ in sweeps), default=0))
    live: dict[int, np.ndarray] = {}
    passes: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for c, beam, hits, qps in sweeps:
        if c not in live:
            lo, hi = (max(0, c - grow), c + span) if churn else (0, span + c)
            live[c] = window(ids[:hi], dists[:hi], lo)
        assert len(hits) == live[c].shape[0], f"checkpoint {c} beam {beam}: {len(hits)} queries"
        found = (hits[:, :, None] == live[c][:, None, :]).any(axis=2).sum(axis=1)
        passes.setdefault((c, beam), []).append((float(found.mean()) / TOPK, qps))
    return [(c, beam, float(np.mean([r for r, _ in runs])), float(np.median([q for _, q in runs])))
            for (c, beam), runs in sorted(passes.items())]


def main(args: list[str]) -> None:
    """Write the recall csv args[3] and the batch time csv args[4] of the log args[0], against the ground
    truth args[1] over a base of args[2] blocks, the first args[5] batches, 0 if absent, only inserting."""
    if len(args) not in (5, 6):
        sys.exit("usage: analyze.py <log> <gt.arrows> <M> <recall.csv> <tput.csv> [<G>]")
    churn, stages, sweeps = scan(args[0])
    with open(args[3], "w") as out:
        out.write("checkpoint,beam,recall,qps\n")
        for c, beam, found, qps in recall(sweeps, churn, args[1], int(args[2]), int(args[5]) if len(args) == 6 else 0):
            out.write(f"{c},{beam},{found:.5f},{qps:.1f}\n")
    with open(args[4], "w") as out:
        out.write("batch,stage,seconds,rate,maxrss\n")
        for b, seconds, rss in stages:
            out.write(f"{b},batch,{seconds:.3f},{BLOCK / seconds:.1f},{'' if rss < 0 else rss}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
