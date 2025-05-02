
include("main.jl")

# Initialize Parameters
params = Params()

# Solve model using Neural VFI
V_net = VNet()
times_NVFI = @elapsed losses_NVFI = run_neural_VFI(V_net, params; n_epochs=10_000)

# Solve model using Krusell-Smith method
times_KS, errors_KS = run_krusell_smith_method(params)
