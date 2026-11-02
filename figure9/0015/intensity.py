"""Per batch (numbered from 1 as analyze.py does) of a run, from its arrows log and its MON file: the batch's seconds, reads and
writes completed per second, the CPU's busy share, and over the sampled insert rows the mean pages written (`wrote`)
and vertex reads (`hop`). argv: arrows, mon, output csv."""

import re
import sys
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

STAGE: Final[re.Pattern] = re.compile(r"(grow|mix)_(\d+)")


def table(path: str) -> pa.Table:
    """The arrows log at `path`, file or stream format, with the columns this script reads."""
    src = pa.memory_map(path)
    try:
        t = pa.ipc.open_file(src).read_all()
    except pa.ArrowInvalid:
        src.seek(0)
        t = pa.ipc.open_stream(src).read_all()
    return t.select(["tag", "op", "start", "laten", "wrote", "hop"])


def monitor(path: str) -> np.ndarray:
    """The MON samples at `path` as rows of epoch, busy, idle, reads, writes."""
    rows = [line.split()[1:6] for line in open(path) if line.startswith("MON ") and len(line.split()) >= 6]
    return np.array([[float(x) for x in row] for row in rows])


def window(mon: np.ndarray, start: float, end: float) -> tuple[float, float, float]:
    """Reads/s, writes/s and the busy share between the first sample at or after `start` and the last at or before
    `end`, NaN when fewer than two samples fall inside."""
    inside = mon[(mon[:, 0] >= start) & (mon[:, 0] <= end)]
    if len(inside) < 2:
        return float("nan"), float("nan"), float("nan")
    d = inside[-1] - inside[0]
    return d[3] / d[0], d[4] / d[0], d[1] / (d[1] + d[2])


def main(arrows: str, mon: str, out: str) -> None:
    """Write the per-batch intensity of the run logged in `arrows` and `mon` to `out`."""
    t, m = table(arrows), monitor(mon)
    stages = t.filter(pc.equal(t["op"], "stage"))
    inserts = t.filter(pc.equal(t["op"], "insert"))
    with open(out, "w") as f:
        f.write("batch,seconds,reads,writes,busy,wrote,hop,sampled\n")
        for tag, start, laten in zip(*(stages[c].to_pylist() for c in ("tag", "start", "laten"))):
            hit = STAGE.fullmatch(tag)
            if hit is None:
                continue
            s, e = start / 1e9, (start + laten) / 1e9
            reads, writes, busy = window(m, s, e)
            rows = inserts.filter(pc.equal(inserts["tag"], tag))
            wrote = pc.mean(rows["wrote"]).as_py() if rows.num_rows else float("nan")
            hop = pc.mean(rows["hop"]).as_py() if rows.num_rows else float("nan")
            f.write(f"{int(hit.group(2)) + 1},{e - s:.1f},{reads:.0f},{writes:.0f},{busy:.4f},{wrote:.3f},{hop:.2f},{rows.num_rows}\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
