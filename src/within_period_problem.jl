"""
Solve the within-period problem (WPP) for the Krusell-Smith model.

iterate_V and iterate_λ are key exported functions that solve the WPP.

Collectively, these functions implement the function called Φ in the paper.
"""

#############################
# Household Model Functions #
#############################

# Utility #
#---------#
function get_indirect_u(params)
    (;η) = params
    u = @. (max(WEALTH_GRID - WEALTH_NEXT_GRID, 0)^(1-η) - 1) / (1-η)
    u[u.<0] .= -Inf
    return u
end

# Wealth #
#--------#

get_K_bar(λ_start) = sum(λ_start .* WEALTH_GRID)
get_K_bar(as::ModelData) = get_K_bar(as.λ_start)
get_L_bar(λ_start) = sum(λ_start .* Z_GRID)
get_L_bar(as::ModelData) = get_L_bar(as.λ_start)

function get_wealth_postinc(as, A, params)
    (;λ_start, wealth_postinc_k_preinc) = as
    (;α, δ) = params
    K = get_K_bar(as)
    L = get_L_bar(as)
    w = (1-α)*A*(K/L)^α
    r = α*A*(K/L)^(α-1)
    
    @. wealth_postinc_k_preinc = WEALTH_GRID*(1+r-δ) + w*Z_GRID
    return wealth_postinc_k_preinc
end


##############################
# Household Problem Solution #
##############################

# Backward #
#----------#
function iterate_V(V_end, Ai, prealloc, params)
    prealloc.V_end .= V_end
    A = A_vals[Ai]
    
    V_preshock = get_V_preshock(V_end, prealloc)
    V_consume = get_V_consume(V_preshock, prealloc)
    V_start = get_V_income(V_consume, prealloc, A, params)
    #TODO? enforce_borrowing_constraint!(V_income, prealloc)

    return V_start
end

# Forward #
#---------#
function iterate_λ(λ_start, sim_prealloc)
    sim_prealloc.λ_start .= λ_start

    λ_prec = get_λ_prec(λ_start, sim_prealloc)
    λ_preshock = get_λ_preshock(λ_prec, sim_prealloc)
    λ_end = get_λ_end(λ_preshock, sim_prealloc)

    return λ_end
end

function solve_within_period_problem!(as::ModelData, V_end, Λ_start, params)
    (;Ai) = Λ_start
    (;λ_start) = as
    (;Ai, λ_start) = Λ_start
    V_start = iterate_V(V_end, Ai, as, params)
    λ_end = iterate_λ(λ_start, as)
    Λ_end = (;Ai, λ_end)
    return V_start, Λ_end
end
