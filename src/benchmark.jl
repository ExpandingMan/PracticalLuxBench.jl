
"""
    Benchmark

A benchmark problem: a model, synthetic data and training/inference settings.  Concrete subtypes
implement

- `name(b)`: a short name used for result files.
- `pyscript(b)`: the python script (relative to [`pydir`](@ref)) implementing the PyTorch side.
- `gendata(b)`: the synthetic data as a `NamedTuple` of arrays.
- `luxmodel(b)`: the Lux model.
- `objective(b)`, `evalfunc(b)`, `summarize(b, outputs)`, `evalbatchsize(b)` and optionally
  `checkinputs(b, data)` and `trainextras(be, b, m, ts, dev, data)`, used for Lux training and
  inference (see `src/lux.jl`).

Benchmarks also have fields `seed`, `batchsize`, `epochs`, `learning_rate` and `infer_batchsizes`.

Benchmarks are serialized to JSON (as their fields) to pass configuration to python.
"""
abstract type Benchmark end

"""
    initparams(b::Benchmark)

Initial parameters and states of the Lux model.  These are shared with PyTorch so that all
backends start from identical weights.
"""
initparams(b::Benchmark) = Lux.setup(Xoshiro(b.seed + 1), luxmodel(b))

_flatparams(a::AbstractArray, prefix::AbstractString) = [prefix=>Array(a)]
function _flatparams(nt::NamedTuple, prefix::AbstractString="")
    ps = map(collect(keys(nt))) do k
        _flatparams(nt[k], isempty(prefix) ? string(k) : string(prefix, ".", k))
    end
    reduce(vcat, ps; init=Pair{String,Array}[])
end

"""
    exportarrays(b::Benchmark, data, ps)

All arrays that python needs, keyed by name.  Parameters are keyed by their path in the Lux
parameter tree, e.g. `"layer_1.weight"`.
"""
function exportarrays(b::Benchmark, data, ps)
    d = Dict{String,Array}(map(k -> string(k)=>data[k], collect(keys(data))))
    merge(d, Dict(_flatparams(ps)))
end

"""
    _reference(b::Benchmark, data, ps, st)

Float64 predictions on `checkinputs(b, data)` from plain Lux (no Reactant) on the CPU, used as the ground
truth when checking the numerical agreement of each backend's inference.
"""
function _reference(b::Benchmark, data, ps, st)
    m = luxmodel(b)
    first(m(Float64.(checkinputs(b, data)), f64(ps), Lux.testmode(st)))
end

"""
    prederror(ŷ, ref)

Summary statistics of the discrepancy between predictions `ŷ` and reference predictions `ref`.
"""
function prederror(ŷ::AbstractArray, ref::AbstractArray)
    δ = Float64.(ŷ) .- ref
    Dict{String,Any}(
        "max_abs"=>maximum(abs, δ),
        "rms"=>sqrt(mean(abs2, δ)),
        "ref_rms"=>sqrt(mean(abs2, ref)),
        "max_rel"=>maximum(abs, δ) / maximum(abs, ref),
    )
end

_flags(be::TorchBackend) = be.compile ? ["--compile"] : String[]

function train(be::TorchBackend, b::Benchmark, ws::Workspace, data, ps, st)
    out = outpath(ws, be, "train")
    runpython(pyscript(b), "train", "--workspace", ws.dir, "--device", device(be), "--out", out,
        _flags(be)...,
    )
    merge(_resultheader(be), loadresults(out))
end

function infer(be::TorchBackend, b::Benchmark, ws::Workspace, data, ps, st)
    out = outpath(ws, be, "infer")
    pd = preddir(ws, be)
    runpython(pyscript(b), "infer", "--workspace", ws.dir, "--device", device(be), "--out", out, "--pred", pd,
        _flags(be)...,
    )
    res = merge(_resultheader(be), loadresults(out))
    res["prediction_error"] = prederror(readarrays(pd)["pred"], _reference(b, data, ps, st))
    res
end

_taskfunc(task::AbstractString) = task == "train" ? train : task == "infer" ? infer : throw(ArgumentError("unknown task $task"))

"""
    defaultbackends(b::Benchmark)

The backends on which `b` is run by default.
"""
defaultbackends(::Benchmark) = defaultbackends()

"""
    benchmark(b::Benchmark; backends=defaultbackends(b), tasks=("train", "infer"), dir=resultsdir())

Run benchmark `b` on each of `backends` for each of `tasks`, writing results to `dir/\$(name)_\$(task).json`.  Returns the paths of the result files.

All backends receive identical synthetic data and identical initial parameters.
"""
function benchmark(b::Benchmark;
    backends=defaultbackends(b),
    tasks=("train", "infer"),
    dir::AbstractString=resultsdir(),
)
    data = gendata(b)
    (ps, st) = initparams(b)
    sys = systeminfo()
    mktempdir() do tmp
        ws = Workspace(tmp)
        JSON.json(configpath(ws), b; pretty=true)
        writearrays(datadir(ws), exportarrays(b, data, ps))
        # the parameter names in the order of the Lux parameter tree
        JSON.json(joinpath(datadir(ws), "params.json"), first.(_flatparams(ps)))
        map(collect(tasks)) do task
            f = _taskfunc(task)
            runs = map(backends) do be
                @info "running $(name(b)) $task on $(label(be))"
                # copies guard against backends which mutate parameters in place
                f(be, b, ws, data, deepcopy(ps), deepcopy(st))
            end
            res = Dict{String,Any}("benchmark"=>name(b), "task"=>task, "config"=>b, "system"=>sys, "runs"=>runs)
            saveresults(resultspath(dir, name(b), task), res)
        end
    end
end
