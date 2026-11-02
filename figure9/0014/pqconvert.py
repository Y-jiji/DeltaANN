# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy"]
# ///
"""Write a fnct PQ codebook as a PipeANN pivots file.

argv: the codebook `pqtrain` wrote (raw little-endian f32, `book[(band * CENTERS + i) * BAND..]` the
BAND dims of centroid i of a band, codes against the zero point), the output path, and optionally the
dims of a point, DIMS (msturing's and spacev's 100) by default, 128 for sift, and the dims per band,
BAND (8) by default. Band b becomes PipeANN chunk b over dims [b * BAND, min((b + 1) * BAND, dims)); the
padding dims past dims must be zero in the book and are dropped. The centroid is zero and the dims stay
in order.
"""

import sys
from typing import Final

import numpy as np

DIMS: Final[int] = 100
BAND: Final[int] = 8
CENTERS: Final[int] = 256
SECTOR: Final[int] = 4096


def block(array: np.ndarray) -> bytes:
    """One PipeANN bin block as save_bin writes it: u32 rows, u32 cols, then the array."""
    rows, cols = array.shape
    return np.array([rows, cols], dtype="<u4").tobytes() + np.ascontiguousarray(array).tobytes()


def main(book: str, pivots: str, dims: int, band: int) -> None:
    """Read the fnct codebook at `book` over points of `dims`, `band` dims a band, and write its PipeANN pivots
    file to `pivots`."""
    bands = -(-dims // band)
    cube = np.fromfile(book, dtype="<f4").reshape(bands, CENTERS, band)
    assert not cube[-1, :, dims - (bands - 1) * band :].any(), "padding dims of the last band are not zero"
    full = cube.transpose(1, 0, 2).reshape(CENTERS, bands * band)[:, :dims]
    offsets = np.minimum(np.arange(bands + 1) * band, dims).astype("<u4")
    parts = [block(full), block(np.zeros((dims, 1), dtype="<f4")),
             block(np.arange(dims, dtype="<u4")[:, None]), block(offsets[:, None])]
    cumul = np.cumsum([SECTOR] + [len(part) for part in parts]).astype("<u8")
    head = block(cumul[:, None])
    with open(pivots, "wb") as out:
        out.write(head + bytes(SECTOR - len(head)) + b"".join(parts))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else DIMS,
         int(sys.argv[4]) if len(sys.argv) > 4 else BAND)
