# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=2"]
# ///
"""Reorder the inserted rows of a headerless stream, keeping its base.

argv: the input stream, the output stream, the output's source-row index (.npy, uint32), the bytes per row, the base
rows, the seed. Copies rows [0, base) as they are, the rows the loaded graph is built on, then writes rows [base, n)
in the order of a permutation drawn with the seed, in chunks of 1M rows; index[i] is the input row of output row i.
"""

import os
import sys

import numpy as np

source, target, index, width, base, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
CHUNK = 1_000_000
size = os.path.getsize(source)
assert size % width == 0, f"{source}: {size} bytes, not rows of {width}"
rows = size // width
assert base < rows < 2**32, f"{rows} rows, base {base}"
order = np.concatenate([np.arange(base, dtype=np.uint32),
                        base + np.random.default_rng(seed).permutation(rows - base).astype(np.uint32)])
with open(source, "rb") as f:
    f.seek(base * width)
    tail = np.frombuffer(f.read((rows - base) * width), dtype=np.uint8).reshape(rows - base, width)
with open(source, "rb") as f, open(target, "wb") as out:
    left = base * width
    while left:
        piece = f.read(min(left, CHUNK * width))
        out.write(piece)
        left -= len(piece)
    for start in range(base, rows, CHUNK):
        out.write(tail[order[start:min(start + CHUNK, rows)] - base].tobytes())
np.save(index, order)
assert os.path.getsize(target) == size, f"{target}: {os.path.getsize(target)} bytes, want {size}"
print(f"rows {rows} base {base} seed {seed} first moved {order[base:base + 4].tolist()}")
