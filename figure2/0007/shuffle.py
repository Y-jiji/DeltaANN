# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy"]
# ///
"""Shuffle the inserted half of a headerless 20M-point file.

argv: the input file, the output file, the bytes per row. Copies the first `BASE` rows, the points
the loaded graph is built on, as they are, then writes the next `BASE` rows, the points the experiment inserts,
in the order of a permutation drawn with the fixed `SEED`, in chunks of `CHUNK` rows.
"""

import sys
from typing import Final

import numpy as np

BASE: Final[int] = 10_000_000
SEED: Final[int] = 7
CHUNK: Final[int] = 1_000_000


def main(args: list[str]) -> None:
    """Write the shuffled copy of `args[0]` to `args[1]`, rows of `args[2]` bytes."""
    source, target, width = args[0], args[1], int(args[2])
    rows = np.memmap(source, dtype=np.uint8, mode="r", shape=(2 * BASE, width))
    order = BASE + np.random.default_rng(SEED).permutation(BASE)
    with open(target, "wb") as out:
        for start in range(0, BASE, CHUNK):
            out.write(rows[start:min(start + CHUNK, BASE)].tobytes())
        for start in range(0, BASE, CHUNK):
            out.write(rows[order[start:start + CHUNK]].tobytes())


if __name__ == "__main__":
    main(sys.argv[1:])
