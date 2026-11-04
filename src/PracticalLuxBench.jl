module PracticalLuxBench

using Random, Statistics, Dates, LinearAlgebra
using Lux, Reactant, Optimisers, Zygote
using JSON
using GLMakie

include("backends.jl")
include("arrayio.jl")
include("results.jl")
include("python.jl")
include("benchmark.jl")
include("lux.jl")
include("mlp.jl")
include("unet.jl")
include("plots.jl")

export Backend, LuxBackend, TorchBackend
export Benchmark, MLPBenchmark, UNetBenchmark
export benchmark, train, infer, plotresults, loadresults

function __init__()
    # By default XLA preallocates 75% of GPU memory for the lifetime of the process, which leaves too
    # little for the PyTorch subprocesses.  Allocate on demand instead (as PyTorch does), unless the
    # user has configured this explicitly.
    haskey(ENV, "XLA_REACTANT_GPU_PREALLOCATE") || (Reactant.XLA.XLA_REACTANT_GPU_PREALLOCATE[] = false)
end

end # module PracticalLuxBench
