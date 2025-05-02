
#module NeuralVFISolution

#export VNet
#export simulate_path_neuralV
#export pretrain_V!, train_epoch!
#export run_neural_VFI

using LinearAlgebra
using Statistics
using StatsBase
using Random

#include("../KrusellSmithModel/KrusellSmithModel.jl")
#using .KrusellSmithModel

include("neural_nets.jl")
include("mcmc_sampling.jl")
include("neural_VFI_algorithm.jl")

#end
