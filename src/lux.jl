#====================================================================================================
Lux training and inference, generic over benchmarks.  A `Benchmark` provides (see `Benchmark`)

- `objective(b)`: the objective for `Training.single_train_step!`.
- `evalfunc(b)`: `(m, ps, st, x, y) -> outputs` evaluated (on device) for each test batch.
- `summarize(b, outputs)`: test metrics (a `Dict` of host values) from the outputs for all batches.
- `evalbatchsize(b)`: batch size for evaluating the test set.
- `checkinputs(b, data)`: inputs used to check numerical agreement of inference.
- `trainextras(be, b, m, ts, dev, data)` (optional): additional training results.
====================================================================================================#

checkinputs(b::Benchmark, data) = data.xtest

trainextras(::LuxBackend, ::Benchmark, m, ts, dev, data) = Dict{String,Any}()

# device outputs to host values
_host(x) = Float64(x)
_host(x::AbstractArray) = Float64.(Array(x))
_host(x::Tuple) = map(_host, x)

# the first `n` samples (along the last dimension)
_firstn(x::AbstractArray, n::Integer) = copy(selectdim(x, ndims(x), 1:n))

# split along the last dimension into device-resident minibatches, dropping any remainder
function _batches(dev, x::AbstractArray, y::AbstractArray, n::Integer)
    nb = size(x, ndims(x)) ÷ n
    map(1:nb) do i
        idx = ((i - 1) * n + 1):(i * n)
        (dev(copy(selectdim(x, ndims(x), idx))), dev(copy(selectdim(y, ndims(y), idx))))
    end
end

# parameters are copied since training without Reactant updates them in place
function _trainstate(b::Benchmark, m, dev, ps, st)
    Training.TrainState(m, dev(deepcopy(ps)), dev(deepcopy(st)), Adam(Float32(b.learning_rate)))
end

function _trainstep(be::LuxBackend, b::Benchmark, ts::Training.TrainState, batch)
    Training.single_train_step!(ad(be), objective(b), batch, ts; return_gradients=Lux.False())
end

# a train state with the parameters of `fresh` which reuses the compiled train step of `warm`
function _withcache(fresh::Training.TrainState, warm::Training.TrainState)
    TS = Training.TrainState
    TS(map(k -> k ∈ (:cache, :objective_function) ? getfield(warm, k) : getfield(fresh, k), fieldnames(TS))...)
end

# one epoch over `batches` in the given `order`; returns the train state and the mean loss
function _epoch(be::LuxBackend, b::Benchmark, ts::Training.TrainState, batches, order)
    ℓs = Vector{Any}(undef, length(order))
    for (j, i) ∈ enumerate(order)
        (_, ℓ, _, ts) = _trainstep(be, b, ts, batches[i])
        ℓs[j] = ℓ
    end
    # converting the losses to host values blocks until all steps are complete
    (ts, mean(Float64, ℓs))
end

"""
    _compileexercise(be::LuxBackend, b, m, dev, ps, st, batches)

Run two throwaway rounds of two single-step epochs, each round on its own new train state, before
any timed training, so that the timed epochs include no compilation.  The first absorbs one-time
compilation of Julia code (so it is only representative for the first backend run in a process);
the second measures the cost of compiling a train step in a warm process.  Returns
`[(t₁, ts₁), (t₂, ts₂)]`, the elapsed times and resulting train states.
"""
function _compileexercise(be::LuxBackend, b::Benchmark, m, dev, ps, st, batches)
    map(1:2) do _
        t0 = time_ns()
        (ts, _) = _epoch(be, b, _trainstate(b, m, dev, ps, st), batches, [1])
        # the train state type changes once it holds a compiled step; exercise that code path too
        (ts, _) = _epoch(be, b, ts, batches, [1])
        (_elapsed(t0), ts)
    end
end

# model and evaluation functions, compiled if using Reactant
_compiled(be::LuxBackend, m, x, ps, st) = be.reactant ? (@compile sync=true m(x, ps, st)) : m

function _compiledeval(be::LuxBackend, b::Benchmark, m, ts::Training.TrainState, (x, y))
    f = evalfunc(b)
    be.reactant ? (@compile f(m, ts.parameters, Lux.testmode(ts.states), x, y)) : f
end

function _evaluate(b::Benchmark, f, m, ts::Training.TrainState, batches)
    st = Lux.testmode(ts.states)
    summarize(b, map(((x, y),) -> _host(f(m, ts.parameters, st, x, y)), batches))
end

# vector of `Dict`s of metrics to a `Dict` of vectors
_collectmetrics(ms::AbstractVector) = Dict{String,Any}(map(k -> k=>map(m -> m[k], ms), collect(keys(first(ms)))))

_platform(be::LuxBackend) = be.reactant ? Reactant.XLA.platform_name(Reactant.XLA.default_backend()) : "native"

function train(be::LuxBackend, b::Benchmark, ::Workspace, data, ps, st)
    dev = device(be)
    m = luxmodel(b)
    batches = _batches(dev, data.xtrain, data.ytrain, b.batchsize)
    testbatches = _batches(dev, data.xtest, data.ytest, evalbatchsize(b))

    ((tcold, _), (tcomp, tsw)) = _compileexercise(be, b, m, dev, ps, st, batches)
    ts = _trainstate(b, m, dev, ps, st)
    be.reactant && (ts = _withcache(ts, tsw))

    # compilation and first evaluation of the test set, as for PyTorch
    t0 = time_ns()
    f = _compiledeval(be, b, m, ts, first(testbatches))
    _evaluate(b, f, m, ts, testbatches)
    tc = _elapsed(t0)

    rng = Xoshiro(b.seed + 2)
    (ℓtrain, Δt, metrics) = (Float64[], Float64[], Dict{String,Any}[])
    for _ ∈ 1:b.epochs
        t0 = time_ns()
        (ts, ℓ) = _epoch(be, b, ts, batches, randperm(rng, length(batches)))
        push!(Δt, _elapsed(t0))
        push!(ℓtrain, ℓ)
        push!(metrics, _evaluate(b, f, m, ts, testbatches))
    end
    res = Dict{String,Any}(
        "platform"=>_platform(be),
        "cold_start_time"=>tcold,
        "compile_time"=>tcomp,
        "eval_compile_time"=>tc,
        "epoch_times"=>Δt,
        "train_loss"=>ℓtrain,
    )
    merge(_resultheader(be), res, _collectmetrics(metrics), trainextras(be, b, m, ts, dev, data))
end

function infer(be::LuxBackend, b::Benchmark, ::Workspace, data, ps, st)
    dev = device(be)
    m = luxmodel(b)
    (psd, std) = (dev(ps), dev(Lux.testmode(st)))
    runs = map(b.infer_batchsizes) do n
        x = dev(_firstn(data.xtest, n))
        t0 = time_ns()
        f = _compiled(be, m, x, psd, std)
        tc = _elapsed(t0)
        t0 = time_ns()
        f(x, psd, std)
        tw = _elapsed(t0)
        Dict{String,Any}(
            "batchsize"=>n,
            "compile_time"=>tc,
            "first_call_time"=>tw,
            "times"=>_timereps(() -> f(x, psd, std)),
        )
    end
    x̃ = dev(checkinputs(b, data))
    ŷ = Array(first(_compiled(be, m, x̃, psd, std)(x̃, psd, std)))
    merge(_resultheader(be), Dict{String,Any}(
        "platform"=>_platform(be),
        "runs"=>runs,
        "prediction_error"=>prederror(ŷ, _reference(b, data, ps, st)),
    ))
end
