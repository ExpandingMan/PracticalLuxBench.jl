#====================================================================================================
Run and/or plot the MLP regression benchmark.

    julia --project=. -t auto scripts/mlp.jl [run|plot|all] [--quick] [--dir DIR] [--backends lux-cpu,torch-gpu,...]

Backends are `lux-cpu`, `lux-cpu-noreactant`, `lux-gpu`, `torch-cpu`, `torch-gpu`, `torch-cpu-eager` and
`torch-gpu-eager`.  All but the eager PyTorch backends are run by default.

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
    bs = _value("--backends", "lux-cpu,lux-cpu-noreactant,lux-gpu,torch-cpu,torch-gpu")
    backends = map(_parsebackend, split(bs, ","))
    (; cmd, quick="--quick" ∈ args, dir=_value("--dir", resultsdir()), backends)
end

function (@main)(args)
    opts = parseargs(args)
    b = if opts.quick
        MLPBenchmark(; width=64, ntrain=2^12, ntest=2^10, epochs=3, infer_batchsizes=[1, 64, 2^10])
    else
        MLPBenchmark()
    end
    opts.cmd ∈ ("run", "all") && benchmark(b; backends=opts.backends, dir=opts.dir)
    opts.cmd ∈ ("plot", "all") && plotresults("mlp"; dir=opts.dir)
    0
end
