"""Array exchange with Julia.

Arrays are raw little-endian binary files in column-major (Fortran) order, described by an
`arrays.json` manifest giving dtype and shape *in Julia order*.  Arrays returned by `read` therefore
have Julia's shape; transpose (`.T`) to get the usual PyTorch (batch-first) layout.
"""

import json
from pathlib import Path

import numpy as np

MANIFEST = "arrays.json"


def read(dirpath):
    """Read all arrays in `dirpath`, returning a dict of name => Fortran-ordered numpy array."""
    d = Path(dirpath)
    meta = json.loads((d / MANIFEST).read_text())
    return {
        k: np.fromfile(d / m["file"], dtype=np.dtype(m["dtype"]).newbyteorder("<")).reshape(
            m["shape"], order="F"
        )
        for k, m in meta.items()
    }


def write(dirpath, arrays):
    """Write a dict of name => numpy array, where each array has Julia's shape (e.g. `pred.T`)."""
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    meta = {}
    for k, a in arrays.items():
        f = f"{k}.bin"
        a = np.asarray(a)
        a.ravel(order="F").astype(a.dtype.newbyteorder("<"), copy=False).tofile(d / f)
        meta[k] = {"file": f, "dtype": a.dtype.name, "shape": list(a.shape)}
    (d / MANIFEST).write_text(json.dumps(meta, indent=2))
