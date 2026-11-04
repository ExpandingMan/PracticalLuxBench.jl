"""Shared utilities for the PyTorch benchmarks: devices, timing, workspace and results IO."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

# keep triton's current default and silence its warning about the default changing
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
# disable inductor's persistent on-disk caches so that compile times do not depend on previous runs
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

import torch  # noqa: E402

from . import arrayio


def parseargs(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("task", choices=["train", "infer"])
    p.add_argument("--workspace", required=True, help="directory with config.json and data/")
    p.add_argument("--device", required=True, choices=["cpu", "cuda"])
    p.add_argument("--out", required=True, help="path of the output JSON file")
    p.add_argument("--pred", default=None, help="directory to write test set predictions to (infer)")
    p.add_argument("--compile", action="store_true", help="compile the model with torch.compile")
    return p.parse_args()


def loadworkspace(path):
    """Return the benchmark config dict and the dict of input arrays."""
    ws = Path(path)
    cfg = json.loads((ws / "config.json").read_text())
    return cfg, arrayio.read(ws / "data")


def getdevice(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(name)


def sync(device):
    """Block until all queued work on `device` is complete."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def totensor(a, device):
    """Convert a Julia-ordered array to a batch-first contiguous tensor on `device`."""
    return torch.from_numpy(a.T.copy()).to(device)


def timereps(f, mintime=0.25, minreps=10, maxreps=1000):
    """Mirror of Julia's `_timereps`: wall-clock seconds for each call of `f` (which must block)."""
    ts = []
    t0 = time.perf_counter()
    while len(ts) < minreps or (time.perf_counter() - t0 < mintime and len(ts) < maxreps):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return ts


def maybecompile(model, compile):
    """`torch.compile` with static shapes (recompiling for each new shape, like Reactant)."""
    return torch.compile(model, dynamic=False) if compile else model


def info(device, compile):
    """Backend details recorded with the results."""
    d = {
        "torch_compile": compile,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "platform": device.type,
    }
    if device.type == "cuda":
        d["cuda"] = torch.version.cuda
        d["device_name"] = torch.cuda.get_device_name(device)
    return d


def saveresults(path, res):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(res, indent=2))
