
################
# Income Shock #
################


# Backward #
#----------#
"Compute V_preshock. Get values in terms of z by multiplying along z' by z_T"
function get_V_preshock(V_end, prealloc)
    (;V_preshock) = prealloc
    mul!(V_preshock, V_end, βZ_T_TRANSPOSE)
    return V_preshock
end

# Forward #
#---------#
function get_λ_end(λ_preshock, sim_prealloc)
    (;λ_end) = sim_prealloc

    mul!(λ_end, λ_preshock, Z_T)
    return λ_end
end
