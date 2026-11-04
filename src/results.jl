
"""
    resultsdir()

The default directory for benchmark results and plots.
"""
resultsdir() = pkgdir(@__MODULE__, "results")

_elapsed(t0::Integer) = (time_ns() - t0) / 1e9

function _gpuname()
    isnothing(Sys.which("nvidia-smi")) && return nothing
    try
        readchomp(`nvidia-smi --query-gpu=name --format=csv,noheader`)
    catch
        nothing
    end
end

"""
    systeminfo()

Information about the host and Julia packages, recorded with each set of results.
"""
function systeminfo()
    Dict{String,Any}(
        "date"=>string(now()),
        "hostname"=>gethostname(),
        "julia"=>string(VERSION),
        "lux"=>string(pkgversion(Lux)),
        "reactant"=>string(pkgversion(Reactant)),
        "cpu"=>first(Sys.cpu_info()).model,
        "cpu_threads"=>Sys.CPU_THREADS,
        "julia_threads"=>Threads.nthreads(),
        "blas_threads"=>BLAS.get_num_threads(),
        "gpu"=>_gpuname(),
    )
end

"""
    _timereps(f; mintime=0.25, minreps=10, maxreps=1000)

Call `f()` repeatedly, returning a vector of the wall-clock time in seconds of each call.  Calls are
made at least `minreps` times and until either `mintime` seconds have elapsed or `maxreps` calls have
been made.  `f` must block until its computation is complete.  The python side uses identical logic.
"""
function _timereps(f; mintime::Real=0.25, minreps::Integer=10, maxreps::Integer=1000)
    ts = Float64[]
    t0 = time_ns()
    while length(ts) < minreps || (_elapsed(t0) < mintime && length(ts) < maxreps)
        t = time_ns()
        f()
        push!(ts, _elapsed(t))
    end
    ts
end

resultspath(dir::AbstractString, name::AbstractString, task::AbstractString) = joinpath(dir, string(name, "_", task, ".json"))

function saveresults(path::AbstractString, res)
    mkpath(dirname(path))
    JSON.json(path, res; pretty=true, allownan=true)
    path
end

"""
    loadresults(path)
    loadresults(name, task; dir=resultsdir())

Load benchmark results from a JSON file, e.g. `loadresults("mlp", "train")`.
"""
loadresults(path::AbstractString) = JSON.parsefile(path; allownan=true)
loadresults(name::AbstractString, task::AbstractString; dir::AbstractString=resultsdir()) = loadresults(resultspath(dir, name, task))
