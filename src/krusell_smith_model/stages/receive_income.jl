
##################
# Receive Income #
##################

# Backward #
#----------#
"""
    Compute pre-income value, V_income.

Simply interpolate post-income value V_consume onto "post-income
wealth in terms of pre-income wealth," wealth_postinc_k_preinc.
"""
function get_V_income(V_consume, prealloc, A, params)
    (;V_income, wealth_postinc_k_preinc) = prealloc

    get_wealth_postinc(prealloc, A, params)

    # We have V_consume in terms of wealth_postinc (= wealth_consume)
    # Put that in terms of wealth_preinc by resampling at wealth_postinc for each wealth_preinc
    @threads for zi=1:N_Z
        @views reinterpolate!(
            V_income[:,zi], V_consume[:,zi], WEALTH_GRID_FLAT, wealth_postinc_k_preinc[:,zi],
            Val(-Inf)
        )
    end

    V_income .= min.(V_income, V_consume[end:end,:])

    return V_income
end

# Forward #
#---------#
function get_λ_prec(μ_start, sim_prealloc)
    (;wealth_postinc_k_preinc, λ_prec) = sim_prealloc

    @threads for zi=1:N_Z
        # Convert λ_postmarket (over pre-income wealth)
        # to λ_prec (over post-income wealth)
        @views convert_distribution!(
            λ_prec[:,zi],
            μ_start[:,zi],
            wealth_postinc_k_preinc[:,zi],
            WEALTH_GRID_FLAT)
    end
    #TODO Document why this replacement is necessary
    # It's dangerous because it could cause things to silently fail
    replace!(λ_prec, NaN=>0)

    return λ_prec
end
