#====================================================================================================
Arrays are exchanged with python as raw little-endian binary files in column-major order, with a
`arrays.json` manifest giving the element type and the shape *in Julia order*.  The python side
(`python/plb/arrayio.py`) reads them with `order="F"` so that shapes agree on both sides.
====================================================================================================#

_dtype(::Type{Float32}) = "float32"
_dtype(::Type{Float64}) = "float64"
_dtype(::Type{Int32}) = "int32"

function _jltype(s::AbstractString)
    s == "float32" && return Float32
    s == "float64" && return Float64
    s == "int32" && return Int32
    error("unsupported dtype $s")
end

_manifestpath(dir::AbstractString) = joinpath(dir, "arrays.json")

"""
    writearrays(dir, arrays)

Write each `name => array` pair of `arrays` to `dir/name.bin` and write a manifest describing them
to `dir/arrays.json`.  Returns `dir`.
"""
function writearrays(dir::AbstractString, arrays)
    mkpath(dir)
    meta = Dict(map(collect(arrays)) do (k, a)
        a = Array(a)
        f = string(k, ".bin")
        write(joinpath(dir, f), a)
        k => Dict("file"=>f, "dtype"=>_dtype(eltype(a)), "shape"=>collect(size(a)))
    end)
    JSON.json(_manifestpath(dir), meta; pretty=true)
    dir
end

"""
    readarrays(dir)

Read all arrays in `dir` written with [`writearrays`](@ref) (or its python counterpart), returning
a `Dict` of `name => array`.
"""
function readarrays(dir::AbstractString)
    meta = JSON.parsefile(_manifestpath(dir))
    Dict(map(collect(meta)) do (k, m)
        a = Array{_jltype(m["dtype"])}(undef, Int.(m["shape"])...)
        read!(joinpath(dir, m["file"]), a)
        String(k) => a
    end)
end
