"""PyTorch side of the MLP regression benchmark (`MLPBenchmark` in Julia).

Run from Julia via `uv run`; reads the config, data and initial parameters from the workspace.
"""

import numpy as np
import torch
from torch import nn

from plb import arrayio, loop
from plb.common import getdevice, info, loadworkspace, parseargs, saveresults, totensor


def buildmodel(cfg):
    """Same architecture as `luxmodel(::MLPBenchmark)`; layer `i` here is `layer_i` in Lux."""
    dims = [cfg["nin"]] + [cfg["width"]] * cfg["depth"]
    layers = []
    for i in range(cfg["depth"]):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU(approximate="tanh")]
    layers.append(nn.Linear(cfg["width"], cfg["nout"]))
    return nn.Sequential(*layers)


def loadparams(model, arrays):
    """Copy the Lux initial parameters into `model`.  Lux weights are `(out, in)` like torch."""
    linear = [m for m in model if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for i, m in enumerate(linear, start=1):
            m.weight.copy_(torch.from_numpy(np.ascontiguousarray(arrays[f"layer_{i}.weight"])))
            m.bias.copy_(torch.from_numpy(np.ascontiguousarray(arrays[f"layer_{i}.bias"])))
    return model


def setup(cfg, arrays, device):
    return loadparams(buildmodel(cfg), arrays).to(device)


def train(cfg, arrays, device, compile):
    (x, y) = (totensor(arrays["xtrain"], device), totensor(arrays["ytrain"], device))
    (xtest, ytest) = (totensor(arrays["xtest"], device), totensor(arrays["ytest"], device))
    lossfn = nn.MSELoss()

    def evaluate(m):
        m.eval()
        with torch.no_grad():
            return {"test_loss": lossfn(m(xtest), ytest).item()}

    (res, _) = loop.train(
        cfg,
        lambda: setup(cfg, arrays, device),
        loop.batches(x, y, cfg["batchsize"]),
        lossfn,
        evaluate,
        device,
        compile,
    )
    return {**info(device, compile), **res}


def infer(cfg, arrays, device, preddir, compile):
    xtest = totensor(arrays["xtest"], device)
    (runs, pred) = loop.infer(cfg, setup(cfg, arrays, device), xtest, xtest, device, compile)
    arrayio.write(preddir, {"pred": pred.T})
    return {**info(device, compile), "runs": runs}


def main():
    args = parseargs("PyTorch MLP regression benchmark")
    (cfg, arrays) = loadworkspace(args.workspace)
    device = getdevice(args.device)
    if args.task == "train":
        res = train(cfg, arrays, device, args.compile)
    else:
        if args.pred is None:
            raise ValueError("--pred is required for the infer task")
        res = infer(cfg, arrays, device, args.pred, args.compile)
    saveresults(args.out, res)


if __name__ == "__main__":
    main()
