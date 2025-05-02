
include("main.jl")

# Initialize Parameters
params = Params()

# Train Neural Network to find global solution
V_net = VNet()
losses = run_neural_VFI(V_net, params; n_epochs=10_000)
