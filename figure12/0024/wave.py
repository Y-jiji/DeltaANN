#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""usage: wave.py <bin> <i8|u8|f32> <dim> <rows> <out>: the first rows of 14 sweeps over the rows sorted by dimension 0,
output row q of sweep j being sorted row j + 14i, with i = q on even j and i = count_j - 1 - q on odd j."""
import sys

import numpy as np

src, kind, dim, rows, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
K = 14
a = np.memmap(src, dtype={"i8": np.int8, "u8": np.uint8, "f32": np.float32}[kind], mode="r").reshape(-1, dim)
order = np.argsort(np.asarray(a[:, 0]), kind="stable")
left = rows
with open(out, "wb") as f:
    for j in range(K):
        ids = order[j::K]
        if j % 2:
            ids = ids[::-1]
        ids = ids[:left]
        for s in range(0, len(ids), 1 << 20):
            f.write(np.asarray(a[ids[s:s + (1 << 20)]]).tobytes())
        left -= len(ids)
        if not left:
            break
