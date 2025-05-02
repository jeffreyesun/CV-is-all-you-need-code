module KrusellSmithModel

# Constants
export FLOAT_PRECISION
export N_K, N_Z, STATE_IDXs
export K_DIM, Z_DIM, N_DIMS
export Π, A_GRID, β
# Data Structures
export Params, ModelData
# Functions
export initialize_model, initialize_path
export iterate_V, iterate_λ
export solve_within_period_problem!, apply_aggregate_shock
export add_state_to_sample!

using LinearAlgebra
using Statistics
using StatsBase

import Base.Threads: @threads

#############
# Constants #
#############

const FLOAT_PRECISION = Float32

###############
# State Space #
###############

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
include("simulate.jl")

end
