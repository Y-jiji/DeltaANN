# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy"]
# ///
"""Flatten a ParlayANN graph into a fixed-width edge list.

argv: the ParlayANN graph (u32 n, R, n degrees, then the neighbor lists back to back) and the
output path. Writes n rows of R u32 neighbors, row i for node i, unused slots set to PAD, and
nothing else: no header, no degrees. Streams BLOCK rows at a time, both files memory-mapped.
"""

import sys
from typing import Final

import numpy as np

PAD: Final[int] = 0xFFFFFFFF
BLOCK: Final[int] = 1_000_000


def main() -> None:
    """Read the ParlayANN graph at argv[1] and write its padded n x R edge list to argv[2]."""
    graph, output = sys.argv[1], sys.argv[2]
    words = np.memmap(graph, dtype="<u4", mode="r")
    n, width = int(words[0]), int(words[1])
    degree = words[2 : 2 + n]
    offset = np.zeros(n + 1, dtype=np.uint64)
    np.cumsum(degree, dtype=np.uint64, out=offset[1:])
    total = int(offset[-1])
    edges = words[2 + n :]
    assert edges.size == total, f"graph holds {edges.size} edges, degrees sum to {total}"
    assert int(degree.max(initial=0)) <= width, f"a degree exceeds the max degree {width}"
    flat = np.memmap(output, dtype="<u4", mode="w+", shape=(n, width))
    column = np.arange(width, dtype=np.uint32)
    for lo in range(0, n, BLOCK):
        hi = min(lo + BLOCK, n)
        block = np.full((hi - lo, width), PAD, dtype="<u4")
        block[column < degree[lo:hi, None]] = edges[int(offset[lo]) : int(offset[hi])]
        flat[lo:hi] = block
    flat.flush()


if __name__ == "__main__":
    main()
