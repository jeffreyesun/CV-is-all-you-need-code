
include("main.jl")

# Initialize Parameters
params = Params()

# Train Neural Network to find global solution
#NOTE! This may fail if the initial V_net is not feasible
# If it fails, try initializing V_net again and trying again
V_net = VNet(false, true)
losses = run_neural_VFI(V_net, params; n_epochs=10_000)
