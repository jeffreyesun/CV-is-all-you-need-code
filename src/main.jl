
using LinearAlgebra
using Statistics
using StatsBase
using Accessors
import Base: tail
import Base.Threads: @threads

#############
# Constants #
#############

const FLOAT_PRECISION = Float32

###############
# State Space #
###############

# Grid sizetarget_moments = get_empirical_aggregate_moments()
const N_K = 65
const N_Z = 3
const STATE_IDXs = (N_K, N_Z)
# State space dimension layout
const K_DIM, Z_DIM = 1:2
const N_DIMS = length(STATE_IDXs)

##############
# Load Model #
##############

include("helper/helper.jl")
include("types.jl")
include("stages/main.jl")
include("within_period_problem.jl")
include("neural_nets.jl")
include("simulation.jl")
include("global_solution.jl")
include("krusell_smith_solution.jl")
