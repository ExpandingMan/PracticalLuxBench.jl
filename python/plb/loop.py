"""Training and inference loops shared by the PyTorch benchmarks, mirroring `src/lux.jl`."""

import copy
import time

import numpy as np
import torch

from .common import maybecompile, sync, timereps


def makeopt(cfg, model):
    """Adam with the same hyperparameters as `Optimisers.Adam` in Lux."""
    return torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], betas=(0.9, 0.999), eps=1e-8)


def step(model, opt, lossfn, batch):
    """One optimization step, returning the (device resident) loss."""
    (xb, yb) = batch
    opt.zero_grad(set_to_none=True)
    loss = lossfn(model(xb), yb)
    loss.backward()
    opt.step()
    return loss.detach()


def epoch(model, opt, lossfn, batches, order):
    """One epoch over `batches` in the given `order`, returning the mean loss."""
    model.train()
    losses = [step(model, opt, lossfn, batches[i]) for i in order]
    # .item() blocks until all steps are complete
    return torch.stack(losses).double().mean().item()


def batches(x, y, n):
    """Split along the first (batch) dimension into minibatches, dropping any remainder."""
    return [(x[i * n : (i + 1) * n], y[i * n : (i + 1) * n]) for i in range(x.shape[0] // n)]


def train(cfg, newmodel, trainbatches, lossfn, evaluate, device, compile):
    """Train a model, returning the results dict and the trained (possibly compiled) model.

    `newmodel()` must return a new model on `device` with the initial parameters, and
    `evaluate(model)` a dict of test metrics.
    """
    model = newmodel()
    cmodel = maybecompile(model, compile)
    init = copy.deepcopy(model.state_dict())

    # Two untimed rounds of two single-step epochs with throwaway optimizers, mirroring the Lux
    # compile exercise.  The first, on a separate model, absorbs one-time initialization (CUDA
    # context, cuBLAS handles, kernel loading, compiler startup); the second, on the model which is
    # then trained, is the cost of compiling (or starting) training in a warm process.  The initial
    # state is then restored so that the timed epochs include no compilation.
    def warmup(m):
        sync(device)
        t0 = time.perf_counter()
        opt = makeopt(cfg, m)
        epoch(m, opt, lossfn, trainbatches, [0])
        epoch(m, opt, lossfn, trainbatches, [0])
        sync(device)
        return time.perf_counter() - t0

    tcold = warmup(maybecompile(newmodel(), compile))
    tcomp = warmup(cmodel)
    model.load_state_dict(init)

    # the first evaluation also compiles the evaluation graph
    t0 = time.perf_counter()
    evaluate(cmodel)
    tc = time.perf_counter() - t0

    opt = makeopt(cfg, cmodel)
    rng = np.random.default_rng(cfg["seed"] + 2)
    (trainloss, metrics, epochtimes) = ([], [], [])
    for _ in range(cfg["epochs"]):
        sync(device)
        t0 = time.perf_counter()
        trainloss.append(epoch(cmodel, opt, lossfn, trainbatches, rng.permutation(len(trainbatches))))
        epochtimes.append(time.perf_counter() - t0)
        metrics.append(evaluate(cmodel))
    res = {
        "cold_start_time": tcold,
        "compile_time": tcomp,
        "eval_compile_time": tc,
        "epoch_times": epochtimes,
        "train_loss": trainloss,
        **{k: [m[k] for m in metrics] for k in metrics[0]},
    }
    return (res, cmodel)


def infer(cfg, model, x, xcheck, device, compile):
    """Time inference for each of `cfg["infer_batchsizes"]` on the leading samples of `x`.

    Returns the per batch size results and the predictions on `xcheck` as a numpy array.
    """
    model = maybecompile(model.eval(), compile)

    def call(xb):
        model(xb)
        sync(device)

    runs = []
    with torch.inference_mode():
        for n in cfg["infer_batchsizes"]:
            xb = x[:n]
            # with torch.compile the first call for each shape includes compilation
            t0 = time.perf_counter()
            call(xb)
            tw = time.perf_counter() - t0
            runs.append(
                {
                    "batchsize": n,
                    "compile_time": 0.0,
                    "first_call_time": tw,
                    "times": timereps(lambda: call(xb)),
                }
            )
        pred = model(xcheck).cpu().numpy()
    return (runs, pred)
