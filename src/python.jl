
"""
    pydir()

Directory of the `uv` managed python project containing the PyTorch benchmarks.
"""
pydir() = pkgdir(@__MODULE__, "python")

function _uv()
    uv = Sys.which("uv")
    isnothing(uv) && error("`uv` was not found on the PATH; it is required to run the PyTorch benchmarks")
    uv
end

"""
    runpython(script, args...)

Run `script` (relative to [`pydir`](@ref)) with `uv run` in the package's python project, passing
`args` as command line arguments.  Throws if the process fails.
"""
function runpython(script::AbstractString, args...)
    d = pydir()
    cmd = `$(_uv()) run --quiet --project $d python $(joinpath(d, script)) $(map(string, args))`
    run(cmd)
end

"""
    Workspace

A temporary directory shared with the python processes, containing the benchmark config
(`config.json`) and the input arrays (`data/`).
"""
struct Workspace
    dir::String
end

configpath(ws::Workspace) = joinpath(ws.dir, "config.json")
datadir(ws::Workspace) = joinpath(ws.dir, "data")
outpath(ws::Workspace, b::Backend, task::AbstractString) = joinpath(ws.dir, string(label(b), "_", task, ".json"))
preddir(ws::Workspace, b::Backend) = joinpath(ws.dir, string(label(b), "_pred"))
