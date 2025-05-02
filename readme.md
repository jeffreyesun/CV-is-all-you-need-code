# Continuation Value is All You Need

This repository contains an application of the Neural Value Function Iteration (Neural VFI) method to the Krusell-Smith model.

To get started, run `quickstart.jl`, which contains the following:
```
include("main.jl")

# Initialize Parameters
params = Params()

# Solve model using Neural VFI
V_net = VNet()
times_NVFI = @elapsed losses_NVFI = run_neural_VFI(V_net, params; n_epochs=10_000)

# Solve model using Krusell-Smith method
times_KS, errors_KS = run_krusell_smith_method(params)
```

This solves the model once using the Neural VFI method and once using the Krusell-Smith method, returning training times and losses.

The module `KrusselSmithModel` contains an optimized implementation of the Within-Period Problem (WPP) of the Krusell-Smith model. This is used by both the Neural VFI implementation in `NeuralVFISolution` and the Krusell-Smith method implementation in `krusell_smith_solution.jl`.

Currently, the most up-to-date overview of this method is the slide deck located (here)[https://jeffreyesun.com/CV_is_all_you_need_slides.pdf].
