"""PyTorch side of the U-net segmentation benchmark (`UNetBenchmark` in Julia).

Run from Julia via `uv run`; reads the config, data and initial parameters from the workspace.
Julia arrays are `(W, H, C, N)`, so after transposing they are the usual `(N, C, H, W)`.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from plb import arrayio, loop
from plb.common import getdevice, info, loadworkspace, parseargs, saveresults, totensor


def convblock(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
    )


class Level(nn.Module):
    """Mirror of `_unetlevel` in Julia; parameters are registered in the same order as in Lux."""

    def __init__(self, cin, chs, i):
        super().__init__()
        self.enc = convblock(cin, chs[i])
        self.inner = None
        if i < len(chs) - 1:
            self.inner = nn.Sequential(
                nn.MaxPool2d(2),
                Level(chs[i], chs, i + 1),
                nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2),
            )
            self.dec = convblock(2 * chs[i], chs[i])

    def forward(self, x):
        x = self.enc(x)
        if self.inner is None:
            return x
        return self.dec(torch.cat([self.inner(x), x], dim=1))


def nclasses(cfg):
    return 1 + len(cfg["shapes"])


def buildmodel(cfg):
    chs = cfg["channels"]
    return nn.Sequential(Level(cfg["nin"], chs, 0), nn.Conv2d(chs[0], nclasses(cfg), 1))


def loadparams(model, arrays, order):
    """Copy the Lux initial parameters into `model`, matching them by order.

    Reversing the dimensions of each Lux array gives the torch layout, e.g. conv weights
    `(kW, kH, in, out)` become `(out, in, kH, kW)` and transposed conv weights `(kW, kH, out, in)`
    become `(in, out, kH, kW)`.  BatchNorm running statistics have the same defaults in both.
    """
    params = list(model.parameters())
    if len(params) != len(order):
        raise ValueError(f"torch model has {len(params)} parameters, Lux has {len(order)}")
    with torch.no_grad():
        for p, name in zip(params, order):
            a = arrays[name].T
            if tuple(a.shape) != tuple(p.shape):
                raise ValueError(f"shape mismatch for {name}: Lux {a.shape}, torch {tuple(p.shape)}")
            p.copy_(torch.from_numpy(np.ascontiguousarray(a)))
    return model


def setup(cfg, arrays, order, device):
    return loadparams(buildmodel(cfg), arrays, order).to(device)


def loaddata(arrays, device):
    xs = {k: totensor(arrays[k], device) for k in ("xtrain", "xtest")}
    ys = {k: totensor(arrays[k], device).long() for k in ("ytrain", "ytest")}
    return {**xs, **ys}


def segeval(cfg, testbatches, lossfn):
    """Returns `evaluate(model)`: test loss, pixel accuracy and per class IoU (as in Julia)."""
    K = nclasses(cfg)

    def evaluate(m):
        m.eval()
        (losses, inter, npred, ntrue) = ([], 0, 0, 0)
        with torch.no_grad():
            for xb, yb in testbatches:
                logits = m(xb)
                losses.append(lossfn(logits, yb))
                p = nn.functional.one_hot(logits.argmax(1), K)
                t = nn.functional.one_hot(yb, K)
                inter = inter + (p * t).sum((0, 1, 2))
                npred = npred + p.sum((0, 1, 2))
                ntrue = ntrue + t.sum((0, 1, 2))
        (inter, npred, ntrue) = (v.double().cpu().numpy() for v in (inter, npred, ntrue))
        iou = inter / (npred + ntrue - inter)
        return {
            "test_loss": torch.stack(losses).double().mean().item(),
            "test_accuracy": inter.sum() / ntrue.sum(),
            "test_iou": iou.tolist(),
            "test_miou": iou.mean(),
        }

    return evaluate


def examples(cfg, model, arrays, data):
    """Example inputs, labels and predicted class maps, as nested lists in Julia orientation."""
    n = cfg["nexamples"]
    model.eval()
    with torch.no_grad():
        pred = model(data["xtest"][:n]).argmax(1).cpu().numpy()
    # nested lists of an (H, W) array have W innermost, the first Julia dimension
    return {
        "example_input": [arrays["xtest"][:, :, 0, i].T.tolist() for i in range(n)],
        "example_label": [arrays["ytest"][:, :, i].T.tolist() for i in range(n)],
        "example_pred": [pred[i].tolist() for i in range(n)],
    }


def train(cfg, arrays, order, device, compile):
    data = loaddata(arrays, device)
    lossfn = nn.CrossEntropyLoss()
    testbatches = loop.batches(data["xtest"], data["ytest"], cfg["eval_batchsize"])
    (res, model) = loop.train(
        cfg,
        lambda: setup(cfg, arrays, order, device),
        loop.batches(data["xtrain"], data["ytrain"], cfg["batchsize"]),
        lossfn,
        segeval(cfg, testbatches, lossfn),
        device,
        compile,
    )
    return {**info(device, compile), **res, **examples(cfg, model, arrays, data)}


def infer(cfg, arrays, order, device, preddir, compile):
    xtest = totensor(arrays["xtest"], device)
    model = setup(cfg, arrays, order, device)
    (runs, pred) = loop.infer(cfg, model, xtest, xtest[: cfg["ncheck"]], device, compile)
    arrayio.write(preddir, {"pred": pred.T})
    return {**info(device, compile), "runs": runs}


def main():
    args = parseargs("PyTorch U-net segmentation benchmark")
    (cfg, arrays) = loadworkspace(args.workspace)
    order = json.loads((Path(args.workspace) / "data" / "params.json").read_text())
    device = getdevice(args.device)
    # let cuDNN pick the fastest convolution algorithms for the fixed input shapes
    torch.backends.cudnn.benchmark = True
    if args.task == "train":
        res = train(cfg, arrays, order, device, args.compile)
    else:
        if args.pred is None:
            raise ValueError("--pred is required for the infer task")
        res = infer(cfg, arrays, order, device, args.pred, args.compile)
    saveresults(args.out, res)


if __name__ == "__main__":
    main()
