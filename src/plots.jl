#====================================================================================================
Plots are made only from the JSON result files, so they can be produced independently of the
benchmarks.  Color encodes the framework and line style / marker encodes the device.
====================================================================================================#

_usesreactant(r) = get(r, "reactant", r["framework"] == "lux")

function _color(r)
    r["framework"] == "torch" && return "#eb6834"
    _usesreactant(r) ? "#2a78d6" : "#1baf7a"
end
_linestyle(r) = r["device"] == "gpu" ? :solid : :dash
_marker(r) = r["device"] == "gpu" ? :circle : :rect
function _fwname(r)
    r["framework"] == "torch" && return get(r, "compiled", false) ? "PyTorch+compile" : "PyTorch eager"
    _usesreactant(r) ? "Lux+Reactant" : "Lux, no Reactant"
end
_prettylabel(r) = string(_fwname(r), " (", uppercase(r["device"]), ")")

# eager PyTorch runs are not plotted; compiled PyTorch is the comparison of interest
_plotted(r) = !(r["framework"] == "torch" && !get(r, "compiled", false))

_seriesattrs(r) = (color=_color(r), linestyle=_linestyle(r), marker=_marker(r), label=_prettylabel(r))

function _lines!(ax, x, y, r)
    a = _seriesattrs(r)
    scatterlines!(ax, x, y; color=a.color, linestyle=a.linestyle, marker=a.marker, label=a.label,
        linewidth=2, markersize=9,
    )
end

_fmtval(v) = v isa AbstractFloat && isinteger(v) ? string(Int(v)) : string(v)
_fmtval(v::AbstractVector) = string("[", join(map(_fmtval, v), ","), "]")

function _title(res)
    task = res["task"] == "train" ? "training" : "inference"
    string(uppercase(res["benchmark"]), " benchmark — ", task)
end

function _subtitle(res)
    s = res["system"]
    runs = filter(r -> haskey(r, "torch"), res["runs"])
    gpu = isnothing(s["gpu"]) ? "no GPU" : s["gpu"]
    sys = string(s["cpu"], " / ", gpu, "  —  Julia ", s["julia"], ", Lux ", s["lux"], ", Reactant ", s["reactant"],
        isempty(runs) ? "" : string(", PyTorch ", first(runs)["torch"]),
    )
    c = res["config"]
    ks = sort(collect(keys(c)))
    string(sys, "\n", join(map(k -> string(k, "=", _fmtval(c[k])), ks), ", "))
end

function _header!(fig, res)
    Label(fig[0, :], _subtitle(res); fontsize=13, color=:gray30, tellwidth=false, word_wrap=true)
    Label(fig[-1, :], _title(res); fontsize=20, font=:bold, tellwidth=false)
end

function _noisefloor!(ax, res)
    c = res["config"]
    haskey(c, "noise") || return nothing
    hlines!(ax, [c["noise"]^2]; color=:gray50, linestyle=:dot, linewidth=1.5, label="noise floor σ²")
end

# log-scale bar chart of one value per run (or per run and group), with values printed above the bars
function _bars!(ax, runs, vals::AbstractVector{<:Real}; fmt=_fmtsec)
    _bars!(ax, runs, [vals]; fmt, alphas=[1.0])
end
function _bars!(ax, runs, groups::AbstractVector{<:AbstractVector};
    fmt=_fmtsec,
    alphas=range(0.45, 1.0; length=length(groups)),
)
    n = length(runs)
    v = Float64.(reduce(vcat, groups))
    x = repeat(1:n, length(groups))
    dodge = reduce(vcat, map(i -> fill(i, n), eachindex(groups)))
    color = reduce(vcat, map(α -> map(r -> (_color(r), α), runs), alphas))
    barplot!(ax, x, v; dodge, color, fillto=minimum(v) / 4, gap=0.3, dodge_gap=0.05, strokewidth=0,
        bar_labels=:y, label_formatter=fmt, label_size=11,
    )
    ax.xticks = (1:n, map(_prettylabel, runs))
    ax.xticklabelrotation = π / 8
    ylims!(ax, minimum(v) / 4, maximum(v) * 4)
end

_fmt3(x) = x ≥ 100 ? string(round(Int, x)) : string(round(x; sigdigits=3))

function _fmtsec(t)
    t ≥ 1 && return string(_fmt3(t), " s")
    t ≥ 1e-3 && return string(_fmt3(1e3 * t), " ms")
    string(_fmt3(1e6 * t), " μs")
end

_fmtsci(x) = string(round(x; sigdigits=2))

"""
    plottrain(res)

Figure summarizing training results `res` (as loaded with [`loadresults`](@ref)).  Dispatches on
the benchmark name; the generic figure is for regression benchmarks.
"""
plottrain(res) = plottrain(Val(Symbol(res["benchmark"])), res)

function plottrain(::Val, res)
    runs = filter(_plotted, res["runs"])
    fig = Figure(; size=(1600, 1000))
    ax = Dict(
        :train=>Axis(fig[1, 1]; title="Training loss (mean over epoch)", xlabel="epoch", ylabel="MSE", yscale=log10),
        :test=>Axis(fig[1, 2]; title="Test loss", xlabel="epoch", ylabel="MSE", yscale=log10),
        :wall=>Axis(fig[1, 3]; title="Test loss vs wall-clock training time", xlabel="cumulative training time (s)",
            ylabel="MSE", xscale=log10, yscale=log10,
        ),
        :epoch=>Axis(fig[2, 1]; title="Time per epoch", xlabel="epoch", ylabel="time (s)", yscale=log10),
        :median=>Axis(fig[2, 2]; title="Median epoch time", ylabel="time (s)", yscale=log10),
        :first=>Axis(fig[2, 3]; title="Untimed warmup: cold process (light) / warm compile (dark)",
            ylabel="time (s)", yscale=log10,
        ),
    )
    foreach(runs) do r
        e = 1:length(r["epoch_times"])
        _lines!(ax[:train], e, r["train_loss"], r)
        _lines!(ax[:test], e, r["test_loss"], r)
        _lines!(ax[:wall], cumsum(Float64.(r["epoch_times"])), r["test_loss"], r)
        _lines!(ax[:epoch], e, r["epoch_times"], r)
    end
    foreach(k -> _noisefloor!(ax[k], res), (:train, :test, :wall))
    _bars!(ax[:median], runs, map(r -> median(r["epoch_times"]), runs))
    _bars!(ax[:first], runs, [map(r -> r["cold_start_time"], runs), map(r -> r["compile_time"], runs)])
    Legend(fig[3, :], ax[:test]; orientation=:horizontal, framevisible=false)
    _header!(fig, res)  # after the axes so that it spans all columns
    fig
end

# JSON nested arrays (innermost = first dimension) to a matrix
_matrix(v) = reduce(hcat, map(c -> Float64.(c), v))

_classcolors(K) = ["#e6e5e0", "#e87ba4", "#6250d6", "#eda100", "#008300", "#e34948"][1:K]

# input, ground truth and each run's prediction for each example image
function _examples!(gl, res, runs)
    src = findfirst(r -> haskey(r, "example_input"), runs)
    isnothing(src) && return nothing
    (inputs, labels) = (runs[src]["example_input"], runs[src]["example_label"])
    shapes = res["config"]["shapes"]
    K = 1 + length(shapes)
    cmap = cgrad(_classcolors(K), K; categorical=true)
    _panel(i, j, title) = Axis(gl[i, j]; title, aspect=DataAspect(), titlesize=12)
    _labelmap!(ax, v) = heatmap!(ax, _matrix(v); colormap=cmap, colorrange=(-0.5, K - 0.5))
    foreach(eachindex(inputs)) do i
        heatmap!(_panel(i, 1, "input $i"), _matrix(inputs[i]); colormap=:grays)
        _labelmap!(_panel(i, 2, "ground truth"), labels[i])
        foreach(enumerate(runs)) do (j, r)
            haskey(r, "example_pred") && _labelmap!(_panel(i, 2 + j, _prettylabel(r)), r["example_pred"][i])
        end
    end
    foreach(hidedecorations!, filter(c -> c isa Axis, contents(gl)))
    names = vcat("background", shapes)
    Legend(gl[:, end + 1], map(c -> PolyElement(; color=c), _classcolors(K)), names, "class"; framevisible=false)
end

function plottrain(::Val{:unet}, res)
    runs = filter(_plotted, res["runs"])
    fig = Figure(; size=(1600, 1500))
    ax = Dict(
        :train=>Axis(fig[1, 1]; title="Training loss (mean over epoch)", xlabel="epoch", ylabel="cross entropy",
            yscale=log10,
        ),
        :test=>Axis(fig[1, 2]; title="Test loss", xlabel="epoch", ylabel="cross entropy", yscale=log10),
        :miou=>Axis(fig[1, 3]; title="Test mean IoU (higher is better)", xlabel="epoch", ylabel="mean IoU"),
        :wall=>Axis(fig[2, 1]; title="Test mean IoU vs wall-clock training time", xlabel="cumulative training time (s)",
            ylabel="mean IoU",
        ),
        :median=>Axis(fig[2, 2]; title="Median epoch time", ylabel="time (s)", yscale=log10),
        :first=>Axis(fig[2, 3]; title="Untimed warmup: cold process (light) / warm compile (dark)",
            ylabel="time (s)", yscale=log10,
        ),
    )
    foreach(runs) do r
        e = 1:length(r["epoch_times"])
        _lines!(ax[:train], e, r["train_loss"], r)
        _lines!(ax[:test], e, r["test_loss"], r)
        _lines!(ax[:miou], e, r["test_miou"], r)
        _lines!(ax[:wall], cumsum(Float64.(r["epoch_times"])), r["test_miou"], r)
    end
    _bars!(ax[:median], runs, map(r -> median(r["epoch_times"]), runs))
    _bars!(ax[:first], runs, [map(r -> r["cold_start_time"], runs), map(r -> r["compile_time"], runs)])
    Legend(fig[3, :], ax[:test]; orientation=:horizontal, framevisible=false)
    Label(fig[4, :], "Example test set segmentations after training"; font=:bold, fontsize=16)
    gl = GridLayout(fig[5, :])
    _examples!(gl, res, runs)
    rowsize!(fig.layout, 5, Relative(0.3))
    _header!(fig, res)  # after the axes so that it spans all columns
    fig
end

"""
    plotinfer(res)

Figure summarizing inference results `res` (as loaded with [`loadresults`](@ref)).
"""
function plotinfer(res)
    runs = filter(_plotted, res["runs"])
    fig = Figure(; size=(1400, 1000))
    ax = Dict(
        :latency=>Axis(fig[1, 1]; title="Latency (median)", xlabel="batch size", ylabel="time per batch (s)",
            xscale=log10, yscale=log10,
        ),
        :throughput=>Axis(fig[1, 2]; title="Throughput (median)", xlabel="batch size", ylabel="samples / s",
            xscale=log10, yscale=log10,
        ),
        :first=>Axis(fig[2, 1]; title="First call (incl. compilation)", xlabel="batch size", ylabel="time (s)",
            xscale=log10, yscale=log10,
        ),
        :error=>Axis(fig[2, 2]; title="Max relative error vs Float64 CPU reference (test set)",
            ylabel="max |ŷ - y_ref| / max |y_ref|", yscale=log10,
        ),
    )
    foreach(runs) do r
        n = map(s -> s["batchsize"], r["runs"])
        t = map(s -> median(s["times"]), r["runs"])
        _lines!(ax[:latency], n, t, r)
        _lines!(ax[:throughput], n, n ./ t, r)
        _lines!(ax[:first], n, map(s -> s["compile_time"] + s["first_call_time"], r["runs"]), r)
    end
    _bars!(ax[:error], runs, map(r -> r["prediction_error"]["max_rel"], runs); fmt=_fmtsci)
    Legend(fig[3, :], ax[:latency]; orientation=:horizontal, framevisible=false)
    _header!(fig, res)  # after the axes so that it spans all columns
    fig
end

"""
    plotresults(name="mlp"; dir=resultsdir(), save=true)

Load the training and inference results of benchmark `name` from `dir` and plot them.  If `save`,
the figures are written to `dir/\$(name)_train.png` and `dir/\$(name)_infer.png`.  Returns a
`NamedTuple` of the figures.
"""
function plotresults(name::AbstractString="mlp"; dir::AbstractString=resultsdir(), save::Bool=true)
    figs = (train=plottrain(loadresults(name, "train"; dir)), infer=plotinfer(loadresults(name, "infer"; dir)))
    save && foreach(k -> GLMakie.save(joinpath(dir, string(name, "_", k, ".png")), figs[k]), keys(figs))
    figs
end
