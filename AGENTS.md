# PracticalLuxBench
This package is for running practical training and inference examples with Lux.jl and
PyTorch and comparing the results to see how close we are to Lux being a viable
alternative to PyTorch.

Agents will have to check online documentation for Lux.jl, Reactant.jl and possibly
pytorch.


## Benchmarks
- Fast and simple benchmarks are done mostly for sanity checking.  They should consist of
  regressions with MLP's of a few layers.
- We are primarily interested in doing U-net image segmentation benchmarks.  Examples
  should be small enough so that training doesn't take much more than about half an hour
  at most.
- Benchmarks are done on synthetic data, all methods for generating the synthetic data
  should be done in Julia.  Synthetic data should accept seed or RNG arguments for
  reproducibility.

## Python
- All python should be run with `uv` or `uvx`.  It is ok to assume the host machine has
  these and error if they are not present.
- Python package management is to be handled by `uv`.
- Benchmarking code is in the Julia package, python benchmarks should be run via shell
  commands within Julia, see `Cmd` and `run`.
- We expect most benchmarks to run from the 100's of milliseconds to tens of minutes, so
  it's not necessary to do super precies benchmarks with BenchmarkTools.
- Be careful to exclude overhead converting Julia to Python and vice versa.  Python
  benchmarks should occur within a single call to python and data loading should be
  excluded from the benchmark.

## Lux
- Julia will have excessive compilation times on the first run.  Where possible, try to
  avoid including compilation in benchmarks by re-running, or running a compilation
  "exercise" first.
- GPU runs should use Reactant.  CPU runs should have an option whether to use reactant.


## Results
- Results should be output in JSON files.  They can be read back for plotting.

## Plotting
- We should be able to run plotting separately from benchmarks.  Plots should load results
  from the JSON file outputs.
- There should be separate plot figures for training and inference.  Each should have
  several useful panels which help to compare performance and also confirm that the model
  converged correctly.

## GPU
- The simple MLP benchmarks should be run on *both* CPU and GPU.
- The more complicated U-net examples can be run on GPU only, in both python and Julia.
