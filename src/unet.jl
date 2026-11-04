
"""
    UNetBenchmark(; kwargs...)

A U-net image segmentation benchmark on synthetic images, run on the GPU only.

Each single-channel `imagesize × imagesize` image contains `minshapes` to `maxshapes` randomly
placed, sized and rotated shapes on a smooth random background with Gaussian noise of standard
deviation `noise`.  The kinds of shapes are given by `shapes` (any of `"disk"`, `"square"` and
`"triangle"`).  Shape intensities are drawn independently of their kind, so the kind can only be
inferred from the geometry.  Each pixel is labeled `0` for background or `k` for the `k`th kind of
shape, so there are `1 + length(shapes)` classes.

The U-net has one level per entry of `channels`, each with two 3×3 convolutions with batch
normalization and ReLU; downsampling is by 2×2 max pooling and upsampling by 2×2 transposed
convolution with stride 2.  It is trained with Adam on the pixel-wise cross entropy.
"""
Base.@kwdef struct UNetBenchmark <: Benchmark
    imagesize::Int = 128
    nin::Int = 1
    shapes::Vector{String} = ["disk", "square", "triangle"]
    minshapes::Int = 2
    maxshapes::Int = 6
    minradius::Float64 = 6.0
    maxradius::Float64 = 20.0
    noise::Float64 = 0.1
    channels::Vector{Int} = [32, 64, 128, 256]
    ntrain::Int = 2048
    ntest::Int = 256
    batchsize::Int = 16
    eval_batchsize::Int = 32
    epochs::Int = 20
    learning_rate::Float64 = 1e-3
    infer_batchsizes::Vector{Int} = [1, 4, 16, 64]
    ncheck::Int = 8
    nexamples::Int = 2
    seed::Int = 1
end

name(::UNetBenchmark) = "unet"
pyscript(::UNetBenchmark) = "unet.py"

nclasses(b::UNetBenchmark) = 1 + length(b.shapes)

defaultbackends(::UNetBenchmark) = [LuxBackend(:gpu), TorchBackend(:gpu)]

#====================================================================================================
Synthetic data
====================================================================================================#

"""
    Shape

A shape which can be painted into a synthetic segmentation image.  Concrete shapes have a `center`,
a circumradius `radius` and possibly an `angle`, and implement `inside(s, p)` for points `p`.
"""
abstract type Shape end

struct DiskShape <: Shape
    center::NTuple{2,Float64}
    radius::Float64
end

struct SquareShape <: Shape
    center::NTuple{2,Float64}
    radius::Float64
    angle::Float64
end

struct TriangleShape <: Shape
    center::NTuple{2,Float64}
    radius::Float64
    angle::Float64
end

function _shapetype(s::AbstractString)
    s == "disk" && return DiskShape
    s == "square" && return SquareShape
    s == "triangle" && return TriangleShape
    throw(ArgumentError("unknown shape $s"))
end

_shape(::Type{DiskShape}, c, r, θ) = DiskShape(c, r)
_shape(::Type{T}, c, r, θ) where {T<:Shape} = T(c, r, θ)

_rotate((u, v), θ) = (cos(θ) * u - sin(θ) * v, sin(θ) * u + cos(θ) * v)

inside(s::DiskShape, p) = hypot((p .- s.center)...) ≤ s.radius

function inside(s::SquareShape, p)
    (u, v) = _rotate(p .- s.center, -s.angle)
    max(abs(u), abs(v)) ≤ s.radius / √2
end

# equilateral; the inradius is half the circumradius
function inside(s::TriangleShape, p)
    d = p .- s.center
    all(j -> d[1] * cos(s.angle + 2π * j / 3) + d[2] * sin(s.angle + 2π * j / 3) ≤ s.radius / 2, 0:2)
end

"""
    paint!(img, lbl, s::Shape, v, k)

Set pixels of `img` inside `s` to `v` and the corresponding pixels of `lbl` to `k`.
"""
function paint!(img::AbstractMatrix, lbl::AbstractMatrix, s::Shape, v, k)
    (rx, ry) = map(s.center, size(img)) do c, n
        max(1, floor(Int, c - s.radius)):min(n, ceil(Int, c + s.radius))
    end
    for j ∈ ry, i ∈ rx
        if inside(s, (i, j))
            img[i, j] = v
            lbl[i, j] = k
        end
    end
    img
end

"""
    segimage(rng, b::UNetBenchmark)

Generate one image and its label map for `b`.
"""
function segimage(rng::AbstractRNG, b::UNetBenchmark)
    n = b.imagesize
    # smooth background: a random linear gradient with values in [0, 0.3]
    (a, gx, gy) = (0.1 * rand(rng), 0.1 * rand(rng), 0.1 * rand(rng))
    img = Float32.(a .+ 0.3 .* (gx .* (1:n) .+ gy .* (1:n)') ./ n)
    lbl = zeros(Int32, n, n)
    foreach(1:rand(rng, b.minshapes:b.maxshapes)) do _
        k = rand(rng, 1:length(b.shapes))
        r = b.minradius + (b.maxradius - b.minradius) * rand(rng)
        c = (r + (n - 2r) * rand(rng), r + (n - 2r) * rand(rng))
        s = _shape(_shapetype(b.shapes[k]), c, r, 2π * rand(rng))
        paint!(img, lbl, s, Float32(0.35 + 0.65 * rand(rng)), Int32(k))
    end
    img .+= Float32(b.noise) .* randn(rng, Float32, n, n)
    (img, lbl)
end

"""
    gendata(rng, b::UNetBenchmark)

Generate training and test data for `b`.  Images are `imagesize × imagesize × 1 × n` and label maps
`imagesize × imagesize × n` with `Int32` class indices starting from `0`.
"""
function gendata(rng::AbstractRNG, b::UNetBenchmark)
    sample(m) = let ims = map(_ -> segimage(rng, b), 1:m)
        (reshape(stack(first.(ims)), b.imagesize, b.imagesize, 1, m), stack(last.(ims)))
    end
    (xtrain, ytrain) = sample(b.ntrain)
    (xtest, ytest) = sample(b.ntest)
    (; xtrain, ytrain, xtest, ytest)
end
gendata(b::UNetBenchmark) = gendata(Xoshiro(b.seed), b)

#====================================================================================================
Model
====================================================================================================#

# cross-correlation (like PyTorch) so that weights can be shared directly
_conv(k::Integer, ch::Pair; kw...) = Conv((k, k), ch; pad=k ÷ 2, cross_correlation=true, kw...)

function _convblock(cin::Integer, cout::Integer)
    Chain(_conv(3, cin => cout; use_bias=false), BatchNorm(cout, relu),
        _conv(3, cout => cout; use_bias=false), BatchNorm(cout, relu),
    )
end

_catchannels(x, skip) = cat(x, skip; dims=3)

# the part of the U-net at level `i` and below, mapping `cin` channels to `chs[i]` channels
function _unetlevel(cin::Integer, chs::AbstractVector, i::Integer)
    enc = _convblock(cin, chs[i])
    i == length(chs) && return enc
    inner = Chain(MaxPool((2, 2)), _unetlevel(chs[i], chs, i + 1),
        ConvTranspose((2, 2), chs[i + 1] => chs[i]; stride=2, cross_correlation=true),
    )
    Chain(enc, SkipConnection(inner, _catchannels), _convblock(2chs[i], chs[i]))
end

luxmodel(b::UNetBenchmark) = Chain(_unetlevel(b.nin, b.channels, 1), _conv(1, b.channels[1] => nclasses(b)))

#====================================================================================================
Loss and metrics
====================================================================================================#

# one-hot encoding of label maps `y` (`W × H × N`) along a new 3rd (channel) dimension
function _onehot(y, K::Integer)
    Float32.(reshape(y, size(y, 1), size(y, 2), 1, size(y, 3)) .== reshape(collect(Int32, 0:(K - 1)), 1, 1, K, 1))
end

# mean pixel-wise cross entropy of logits `ŷ` (`W × H × K × N`)
_segloss(ŷ, y) = -sum(_onehot(y, size(ŷ, 3)) .* logsoftmax(ŷ; dims=3)) / length(y)

function _segobjective(m, ps, st, (x, y))
    (ŷ, st) = m(x, ps, st)
    (_segloss(ŷ, y), st, (;))
end

# loss, and per class: true positives, predicted and true pixel counts
function _segeval(m, ps, st, x, y)
    ŷ = first(m(x, ps, st))
    oh = _onehot(y, size(ŷ, 3))
    p̂ = Float32.(ŷ .== maximum(ŷ; dims=3))
    d = (1, 2, 4)
    (_segloss(ŷ, y), vec(sum(p̂ .* oh; dims=d)), vec(sum(p̂; dims=d)), vec(sum(oh; dims=d)))
end

objective(::UNetBenchmark) = _segobjective
evalfunc(::UNetBenchmark) = _segeval
evalbatchsize(b::UNetBenchmark) = b.eval_batchsize
checkinputs(b::UNetBenchmark, data) = _firstn(data.xtest, b.ncheck)

function summarize(::UNetBenchmark, outs)
    (I, P, T) = map(k -> sum(o -> o[k], outs), (2, 3, 4))
    iou = I ./ (P .+ T .- I)
    Dict{String,Any}(
        "test_loss"=>mean(o -> o[1], outs),
        "test_accuracy"=>sum(I) / sum(T),
        "test_iou"=>iou,
        "test_miou"=>mean(iou),
    )
end

# class maps (0-based) of logits `ŷ` (`W × H × K × N`)
_classmaps(ŷ::AbstractArray) = Int32.(dropdims(map(c -> c[3] - 1, argmax(ŷ; dims=3)); dims=3))

# example inputs, labels and predictions from the trained model
function trainextras(be::LuxBackend, b::UNetBenchmark, m, ts, dev, data)
    x = _firstn(data.xtest, b.nexamples)
    (xd, st) = (dev(x), Lux.testmode(ts.states))
    p = _classmaps(Array(first(_compiled(be, m, xd, ts.parameters, st)(xd, ts.parameters, st))))
    y = _firstn(data.ytest, b.nexamples)
    Dict{String,Any}(
        "example_input"=>map(i -> x[:, :, 1, i], 1:b.nexamples),
        "example_label"=>map(i -> y[:, :, i], 1:b.nexamples),
        "example_pred"=>map(i -> p[:, :, i], 1:b.nexamples),
    )
end
