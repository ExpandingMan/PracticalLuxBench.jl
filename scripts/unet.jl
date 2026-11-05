#====================================================================================================
Run and/or plot the U-net image segmentation benchmark (GPU only).

    julia --project=. -t auto scripts/unet.jl [run|plot|all] [--quick] [--dir DIR] [--backends lux-gpu,torch-gpu,...]

Backends are `lux-gpu`, `torch-gpu` and `torch-gpu-eager`; `lux-gpu` and `torch-gpu` are run by default.

`--quick` runs a much smaller problem, useful for checking that everything works.
====================================================================================================#
using PracticalLuxBench
using PracticalLuxBench: resultsdir

function _parsebackend(s::AbstractString)
    parts = split(s, "-")
    (fw, d) = (parts[1], Symbol(parts[2]))
    flag = length(parts) < 3 ? nothing : parts[3]
    fw == "lux" && return LuxBackend(d; reactant=flag ≠ "noreactant")
    fw == "torch" && return TorchBackend(d; compile=flag ≠ "eager")
    error("unknown backend $s")
end

function parseargs(args)
    cmd = isempty(args) || startswith(first(args), "--") ? "all" : first(args)
    cmd ∈ ("run", "plot", "all") || error("command must be one of run, plot, all; got $cmd")
    _value(flag, default) = (i = findfirst(==(flag), args); isnothing(i) ? default : args[i + 1])
    backends = map(_parsebackend, split(_value("--backends", "lux-gpu,torch-gpu"), ","))
    (; cmd, quick="--quick" ∈ args, dir=_value("--dir", resultsdir()), backends)
end

function (@main)(args)
    opts = parseargs(args)
    # XLA's allocator never returns memory, and after training it can grow to its cap (default 0.75 of
    # free memory) during Lux inference, starving the PyTorch subprocesses which run afterwards.  Lux
    # needs only ≈1.7 GB here; 0.5 leaves room for PyTorch on an 8 GB GPU.  The GPU client is created
    # lazily so this takes effect if set before the first GPU run.
    haskey(ENV, "XLA_REACTANT_GPU_MEM_FRACTION") || (PracticalLuxBench.Reactant.XLA.XLA_REACTANT_GPU_MEM_FRACTION[] = 0.5)
    b = if opts.quick
        UNetBenchmark(;
            imagesize=64,
            channels=[8, 16, 32],
            ntrain=256,
            ntest=64,
            epochs=3,
            infer_batchsizes=[1, 4, 16],
        )
    else
        UNetBenchmark()
    end
    opts.cmd ∈ ("run", "all") && benchmark(b; backends=opts.backends, dir=opts.dir)
    opts.cmd ∈ ("plot", "all") && plotresults("unet"; dir=opts.dir)
    0
end
