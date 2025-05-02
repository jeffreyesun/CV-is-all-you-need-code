
"""
Define structs and constants for the model.

Parameters: Define parameters of the Krusell-Smith Model.

State Space: Define the state grids for the model.

Preallocated Data: Define the container ModelData containing the model's:
    - Value functions
    - Population Distributions
    - Precomputed values
"""

import Distributions: Normal, LogNormal, pdf
import QuantEcon: tauchen
import Random: seed!

##############
# Parameters #
##############

# Standard Parameters #
#---------------------#
@kwdef struct Params{T}
    η::T = 0.9#2.0              # Risk aversion
    α::T = 0.36             # Capital share
    δ::T = 0.025            # Depreciation rate
end

const Π = [0.85 0.15; 0.1 0.9]
const A_vals = Float32[0.5, 1.0]
const β = 0.98 |> FLOAT_PRECISION


# Data, Non-Spatial #
#-------------------#
const WEALTH_MAX = 1e7 / 1e3 |> FLOAT_PRECISION
const Z_PERSISTENCE = 0.95 |> FLOAT_PRECISION
const Z_STD = 0.1 |> FLOAT_PRECISION


###############
# State Space #
###############

# State Space Grids #
#-------------------#

const Z_PROCESS = tauchen(N_Z, Float64(Z_PERSISTENCE), Z_STD, 0)
const WEALTH_GRID_FLAT = FLOAT_PRECISION.(exp.(range(0, stop=log(WEALTH_MAX./10), length=N_K)).*10 .- 10)
const LOGZ_GRID_FLAT = FLOAT_PRECISION.(Z_PROCESS.state_values .+ 6)
const βZ_T = FLOAT_PRECISION.(β.*Z_PROCESS.p .+ 1e-15)
const βZ_T_TRANSPOSE = collect(βZ_T')
const Z_T_TRANSPOSE = FLOAT_PRECISION.(Z_PROCESS.p' .+ 1e-15)
const Z_T = collect(Z_T_TRANSPOSE')

# Shaped Grids for Broadcasting #
#-------------------------------#
const WEALTH_GRID = pad_dims(WEALTH_GRID_FLAT; ndims_new=2)
#const WEALTH_NEXT_GRID = pad_dims(WEALTH_GRID_FLAT; left=1)
const LOGZ_GRID = pad_dims(LOGZ_GRID_FLAT; left=1, ndims_new=2)
const Z_GRID = exp.(LOGZ_GRID)
const WEALTH_NEXT_GRID = pad_dims(WEALTH_GRID_FLAT; left=2)

const X_GRID = (WEALTH_GRID, Z_GRID)


#####################
# Preallocated Data #
#####################

"""
Data for a single age group, intended to be reused for each age group calculation,
so that these values cannot be relied upon after the entire period has been solved.
"""
@kwdef struct ModelData{T<:Number}
    # Value Function
    ## Next
    V_end::Array{T, N_DIMS} = zeros(T, STATE_IDXs)
    ## Preshock
    V_preshock::Array{T, N_DIMS} = zeros(T, STATE_IDXs)
    ## Consume
    wealthi_postc_k_prec::Array{Int, N_DIMS} = zeros(Int, STATE_IDXs)
    V_consume::Array{T, N_DIMS} = zeros(T, STATE_IDXs)
    ## Income
    V_income::Array{T, N_DIMS} = zeros(T, STATE_IDXs)
    wealth_postinc_k_preinc::Array{T, N_DIMS} = zeros(T, STATE_IDXs)

    # Population Distribution
    λ_start::Array{T, N_DIMS} = zeros(T, STATE_IDXs)
    λ_prec::Array{T, N_DIMS} = zeros(T, STATE_IDXs)
    λ_preshock::Array{T, N_DIMS} = zeros(T, STATE_IDXs)
    λ_end::Array{T, N_DIMS} = zeros(T, STATE_IDXs)

    # Precomputed Stuff
    forbidden_states::Array{T, N_DIMS} = @. -Inf*(WEALTH_GRID < 0)
    u_indirect::Vector{Vector{T}} = [zeros(T, N_K) for _ in 1:N_K]
end
ModelData(; kwargs...) = ModelData{FLOAT_PRECISION}(; kwargs...)

function precompute!(md::ModelData, params::Params)
    u_indirect_big = get_indirect_u(params)
    @threads for ki=1:N_K
        md.u_indirect[ki] .= @view(u_indirect_big[ki,1,:])
    end
end

function ModelData(params::Params; kwargs...)
    md = ModelData{FLOAT_PRECISION}(; kwargs...)
    precompute!(md, params)
    return md
end

function initialize_model(params)
    md = ModelData(params)
    md.V_end .= rand(FLOAT_PRECISION, STATE_IDXs) .* 1000
    md.λ_start .= rand(FLOAT_PRECISION, STATE_IDXs)
    md.λ_start[end,:] .= 0
    md.λ_start ./= sum(md.λ_start)
    Ai = 2
    Λ_start = (;Ai, λ_start=md.λ_start)
    Λ_start.λ_start
    return md, Λ_start
end

function initialize_path()
    return (;
        V_end_path = Matrix{FLOAT_PRECISION}[],
        V_end_lookahead_path = Matrix{FLOAT_PRECISION}[],
        V_start_path = Matrix{FLOAT_PRECISION}[],
        λ_start_path = Matrix{FLOAT_PRECISION}[],
        Ai_path = Int[],
    )
end
