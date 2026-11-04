"""
    Backend

A framework + device combination on which a [`Benchmark`](@ref) can be run.
"""
abstract type Backend end

function _checkdevice(d::Symbol)
    d ∈ (:cpu, :gpu) || throw(ArgumentError("device must be `:cpu` or `:gpu`, got `:$d`"))
    d
end

"""
    LuxBackend(device::Symbol; reactant=true)

Lux.jl on `device` which is `:cpu` or `:gpu`.  If `reactant`, the model is compiled with Reactant.jl
(XLA) and differentiated with Enzyme; otherwise it runs natively and is differentiated with Zygote.
GPU runs always use Reactant.
"""
struct LuxBackend <: Backend
    device::Symbol
    reactant::Bool

    function LuxBackend(d::Symbol; reactant::Bool=true)
        _checkdevice(d)
        (d == :gpu && !reactant) && throw(ArgumentError("GPU runs of Lux always use Reactant"))
        new(d, reactant)
    end
end

"""
    TorchBackend(device::Symbol; compile=true)

PyTorch run in a `uv` managed python environment, on `device` which is `:cpu` or `:gpu`.  If
`compile`, the model is compiled with `torch.compile` (with static shapes, like Reactant);
otherwise PyTorch runs in eager mode.
"""
struct TorchBackend <: Backend
    device::Symbol
    compile::Bool

    TorchBackend(d::Symbol; compile::Bool=true) = new(_checkdevice(d), compile)
end

framework(::LuxBackend) = "lux"
framework(::TorchBackend) = "torch"

"""
    label(b::Backend)

A short identifier for the backend, e.g. `"lux-gpu"`, used to tag results.
"""
label(b::Backend) = string(framework(b), "-", b.device)
label(b::LuxBackend) = b.reactant ? string("lux-", b.device) : string("lux-", b.device, "-noreactant")
label(b::TorchBackend) = b.compile ? string("torch-", b.device) : string("torch-", b.device, "-eager")

usesreactant(b::LuxBackend) = b.reactant
usesreactant(::TorchBackend) = false

"""
    compiled(b::Backend)

Whether the backend compiles the model (Reactant for Lux, `torch.compile` for PyTorch).
"""
compiled(b::LuxBackend) = b.reactant
compiled(b::TorchBackend) = b.compile

function defaultbackends()
    [LuxBackend(:cpu), LuxBackend(:cpu; reactant=false), LuxBackend(:gpu), TorchBackend(:cpu), TorchBackend(:gpu)]
end

"""
    device(b::LuxBackend)

Return the Lux device object for `b`.  For Reactant backends, this also sets Reactant's default backend.
"""
function device(b::LuxBackend)
    b.reactant || return cpu_device()
    Reactant.set_default_backend(b.device == :gpu ? "gpu" : "cpu")
    reactant_device(; force=true)
end

device(b::TorchBackend) = b.device == :gpu ? "cuda" : "cpu"

# common fields of every result entry
function _resultheader(b::Backend)
    Dict{String,Any}(
        "label"=>label(b),
        "framework"=>framework(b),
        "device"=>string(b.device),
        "reactant"=>usesreactant(b),
        "compiled"=>compiled(b),
    )
end

"""
    ad(b::LuxBackend)

The automatic differentiation backend used for training with `b`.
"""
ad(b::LuxBackend) = b.reactant ? AutoEnzyme() : AutoZygote()
