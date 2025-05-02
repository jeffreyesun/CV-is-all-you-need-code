
using Plots
using CSV
using DataFrames
using LaTeXStrings
include("main.jl")

default(
    fontfamily="Computer Modern",
    linewidth=2,
    markersize=5,
    size=(600,450),
    framestyle=:box,
    label=nothing,
    grid=true,
    foreground_color_legend=nothing
)

#########
# Setup #
#########

params = Params()

#######################
# Neural Net Solution #
#######################

# Narrow, Shallow #
#-----------------#
V_net = VNet(false, false)
el_n_s_500 = @elapsed errors = run_neural_VFI(V_net, params)
CSV.write("narrow_shallow_500.csv", DataFrame(;epoch=1:length(errors), error=errors))
# 174.0s

# Narrow, Deep #
#--------------#
V_net = VNet(false, true)
el_n_d_500 = @elapsed errors = run_neural_VFI(V_net, params)
CSV.write("narrow_deep_500.csv", DataFrame(;epoch=1:length(errors), error=errors))
# 207.6s

# Wide, Shallow #
#---------------#
V_net = VNet(true, false)
el_w_s_500 = @elapsed errors = run_neural_VFI(V_net, params)
CSV.write("wide_shallow_500.csv", DataFrame(;epoch=1:length(errors), error=errors))
#196.7s

# Wide, Deep #
#------------#
V_net = VNet(true, true)
el_w_d_500 = @elapsed errors = run_neural_VFI(V_net, params)
CSV.write("wide_deep_500.csv", DataFrame(;epoch=1:length(errors), error=errors))
#231.3s

# 2000 #

# Narrow, Shallow #
#-----------------#
V_net = VNet(false, false)
el_n_s_2000 = @elapsed errors = run_neural_VFI(V_net, params)
CSV.write("narrow_shallow_500.csv", DataFrame(;epoch=1:length(errors), error=errors))

# Narrow, Deep #
#--------------#
V_net = VNet(false, true)
el_n_d_2000 = @elapsed errors = run_neural_VFI(V_net, params)
CSV.write("narrow_deep_500.csv", DataFrame(;epoch=1:length(errors), error=errors))

# Wide, Shallow #
#---------------#
V_net = VNet(true, false)
el_w_s_2000 = @elapsed errors = run_neural_VFI(V_net, params)
CSV.write("wide_shallow_500.csv", DataFrame(;epoch=1:length(errors), error=errors))

# Wide, Deep #
#------------#
V_net = VNet(true, true)
el_w_d_2000 = @elapsed errors = run_neural_VFI(V_net, params)
CSV.write("wide_deep_500.csv", DataFrame(;epoch=1:length(errors), error=errors))


#######################################
# Reference Neural Net (Narrow, Deep) #
#######################################

V_net = VNet(false, true)
el_nd_10k = @elapsed errors_test = run_neural_VFI(V_net, params; n_epochs=10_000)
CSV.write("narrow_deep_10k.csv", DataFrame(;epoch=1:length(errors_test), error=errors_test))
#4085.3s

################
# K-S Solution #
################

"""TODO
- [x] Update a0, a1
- [x] Update V_start_big
- [x] Outer loop
"""
#params = Params()

V_start_big = rand(FLOAT_PRECISION, (N_K, N_Z, N_K_BAR, N_A))
a0 = FLOAT_PRECISION[0.7601, 0.7824]
a1 = FLOAT_PRECISION[0.9, 0.9]

#TODO big-T is different here, fix!
KS_errors = []
el_ks_500 = @elapsed for i=1:500
    md, path_data = simulate_path_KSV(params, V_start_big, a0, a1)
    a0, a1, V_start_big_new = update_aV(path_data)

    error = norm(V_start_big .- V_start_big_new)
    V_start_big = V_start_big_new

    (;V_end_path, μ_start_path) = path_data
    var_V = mean(var(vec(V_end), ProbabilityWeights(vec(μ_start))) for (V_end, μ_start)=zip(V_end_path, μ_start_path))
    error /= var_V
    push!(KS_errors, error)

    print(i, " ")
    println(error)
end
#22.57s

# 10k #
#-----#

V_start_big = rand(FLOAT_PRECISION, (N_K, N_Z, N_K_BAR, N_A))
a0 = FLOAT_PRECISION[0.7601, 0.7824]
a1 = FLOAT_PRECISION[0.9, 0.9]

KS_errors = []
el_ks_10k = @elapsed for i=1:10_000
    md, path_data = simulate_path_KSV(params, V_start_big, a0, a1)
    a0, a1, V_start_big_new = update_aV(path_data)

    error = norm(V_start_big .- V_start_big_new)
    V_start_big = V_start_big_new

    (;V_end_path, μ_start_path) = path_data
    var_V = mean(var(vec(V_end), ProbabilityWeights(vec(μ_start))) for (V_end, μ_start)=zip(V_end_path, μ_start_path))
    error /= var_V
    push!(KS_errors, error)

    print(i, " ")
    println(error)
end
#436s

################
# Plot Results #
################

err_ns_500 = CSV.File("narrow_shallow_500.csv").error
err_nd_500 = CSV.File("narrow_deep_500.csv").error
err_ws_500 = CSV.File("wide_shallow_500.csv").error
err_wd_500 = CSV.File("wide_deep_500.csv").error
err_ks_500 = KS_errors

"""TODO
- [ ] Save vals, time
- [ ] Beef up results with comparison over hyperparameters
"""
err_ns_500_ma = [mean(err_ns_500[i:i+10]) for i=1:490]
err_nd_500_ma = [mean(err_nd_500[i:i+10]) for i=1:490]
err_ws_500_ma = [mean(err_ws_500[i:i+10]) for i=1:490]
err_wd_500_ma = [mean(err_wd_500[i:i+10]) for i=1:490]
err_ks_500_ma = [mean(err_ks_500[i:i+10]) for i=1:490]

label = "learning_curve_smallmodel"
fig = plot(
    [err_ks_500_ma, err_ns_500_ma, err_nd_500_ma, err_ws_500_ma, err_wd_500_ma];
    yscale=:log10,
    xlabel="Epoch",
    ylabel="MSE Error (1 = var "*L"(V_{end}))",
    yticks=10. .^(-6:-1),
    label = [
        "Krusell-Smith V" "Narrower, Shallower Neural Net V" "Narrower Neural Net V" "Shallower Neural Net V" "Neural Net V"],
)

savefig(fig, "figures/$label.png")
CSV.write("$label.csv", DataFrame(;epoch=1:length(errors), error=errors))
#TODO y axis, etc., rolling avg?


# Plot for Presentation #
#-----------------------#

errors_test
KS_errors
err_ks_10k = KS_errors
err_nd_10k = errors_test[1:10000]

el_nd_10k
el_ks_10k

figloglog = plot(
    [err_ks_10k, err_nd_10k];
    yscale=:log10,
    xscale=:log10,
    linewidth=1,
    xlabel="Epoch",
    ylabel="MSE Error (Share of var "*L"(V_{end}))",
    yticks=10. .^(-6:-1),
    label = [
        "Krusell-Smith V" "Neural Net V"],
)
savefig(figloglog, "figures/loglog_comparison.png")

mean(log10.(err_nd_10k)[300:end])
mean(log10.(err_ks_10k)[300:end])
log10(mean(err_nd_10k[300:end]))
log10(mean(err_ks_10k[300:end]))
300/10_000*el_nd_10k
300/10_000*el_ks_10k

time_nd_10k = collect(1:10_000)./10_000 .*el_nd_10k
time_ks_10k = collect(1:10_000)./10_000 .*el_ks_10k

figloglog_time = plot(
    [time_ks_10k time_nd_10k],
    [err_ks_10k, err_nd_10k];
    yscale=:log10,
    xscale=:log10,
    linewidth=1,
    xlabel="Time (s)",
    ylabel="MSE Error (Share of var "*L"(V_{end}))",
    yticks=10. .^(-6:-1),
    label = [
        "Krusell-Smith V" "Neural Net V"],
)
savefig(figloglog_time, "figures/loglog_time_comparison.png")


