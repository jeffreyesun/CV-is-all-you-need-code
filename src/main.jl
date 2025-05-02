
using LinearAlgebra
using Statistics
using StatsBase
using Random

##############
# Load Model #
##############
include("KrusellSmithModel/KrusellSmithModel.jl")
using .KrusellSmithModel
include("NeuralVFISolution/NeuralVFISolution.jl")
#using .NeuralVFISolution

include("krusell_smith_solution.jl")
