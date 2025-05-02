
#######################
# Consumption-Savings #
#######################

# Backward #
#----------#
"""
    Solve consumption problem.

Get pre-consumption value V_consume by maximizing
utility + continuation value
over all possible choices of continuation value.
Save the choices as wealthi_postc_k_prec.

V_consume(x) = max_{g,h} u(g,h,ℓ(x)) + V_postconsume(x'(x,g,h))
"""
function get_V_consume(V_postconsume, prealloc)
    (;V_consume, wealthi_postc_k_prec, u_indirect) = prealloc

    Threads.@threads for zi=1:N_Z
        @views k1_argmax!(
            V_consume[:,zi],
            wealthi_postc_k_prec[:,zi],
            V_postconsume[:,zi],
            u_indirect
        )
    end

    return V_consume
end

# Forward #
#---------#
function get_λ_preshock(λ_prec, sim_prealloc)
    (;wealthi_postc_k_prec, λ_preshock) = sim_prealloc

    λ_preshock .= 0

    Threads.@threads for zi=1:N_Z
        for ki=1:N_K
            k1i = wealthi_postc_k_prec[ki,zi]
            λ_preshock[k1i,zi] += λ_prec[ki,zi]
        end
    end
    
    return λ_preshock
end
