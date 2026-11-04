
"""
    MLPBenchmark(; kwargs...)

A small regression benchmark with a multi-layer perceptron, mostly intended for sanity checking.

The synthetic target is a random sum of sinusoids `y = A sin.(Ω x .+ φ) + σ ε` with
`x, ε ~ N(0, I)`, normalized so that `var(y) ≈ 1`.  The noise variance `σ^2` is therefore the
lowest achievable expected test loss.

The model has `depth` hidden layers of size `width` with `gelu` (tanh approximation) activations,
and is trained with Adam on the mean squared error.
"""
Base.@kwdef struct MLPBenchmark <: Benchmark
    nin::Int = 16
    nout::Int = 1
    width::Int = 256
    depth::Int = 3
    nmodes::Int = 32
    frequency::Float64 = 1.5
    noise::Float64 = 0.05
    ntrain::Int = 2^16
    ntest::Int = 2^13
    batchsize::Int = 512
    epochs::Int = 30
    learning_rate::Float64 = 1e-3
    infer_batchsizes::Vector{Int} = [1, 16, 256, 4096, 2^13]
    seed::Int = 1
end

name(::MLPBenchmark) = "mlp"
pyscript(::MLPBenchmark) = "mlp.py"

"""
    gendata(rng, b::MLPBenchmark)

Generate training and test data for `b`.  Inputs are `nin × n` and targets `nout × n`.
"""
function gendata(rng::AbstractRNG, b::MLPBenchmark)
    Ω = Float32(b.frequency / sqrt(b.nin)) .* randn(rng, Float32, b.nmodes, b.nin)
    φ = Float32(2π) .* rand(rng, Float32, b.nmodes)
    A = Float32(sqrt(2 / b.nmodes)) .* randn(rng, Float32, b.nout, b.nmodes)
    σ = Float32(b.noise)
    sample(n) = let x = randn(rng, Float32, b.nin, n)
        (x, A * sin.(Ω * x .+ φ) .+ σ .* randn(rng, Float32, b.nout, n))
    end
    (xtrain, ytrain) = sample(b.ntrain)
    (xtest, ytest) = sample(b.ntest)
    (; xtrain, ytrain, xtest, ytest)
end
gendata(b::MLPBenchmark) = gendata(Xoshiro(b.seed), b)

function luxmodel(b::MLPBenchmark)
    hidden = map(_ -> Dense(b.width => b.width, gelu), 1:(b.depth - 1))
    Chain(Dense(b.nin => b.width, gelu), hidden..., Dense(b.width => b.nout))
end

_mseloss(m, ps, st, x, y) = mean(abs2, first(m(x, ps, st)) .- y)

objective(::MLPBenchmark) = MSELoss()
evalfunc(::MLPBenchmark) = _mseloss
evalbatchsize(b::MLPBenchmark) = b.ntest
summarize(::MLPBenchmark, outs) = Dict{String,Any}("test_loss"=>mean(outs))
