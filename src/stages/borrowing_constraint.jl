
########################
# Borrowing Constraint #
########################

# Backward #
#----------#
"""
    Convert any values below the borrowing constraint to -Inf.
    Must be applied whenever the household makes a decision that affects
    the borrowing constraint. I.e. saving or buying a house.
"""
function enforce_borrowing_constraint!(V, prealloc)
    V .+= prealloc.forbidden_states
    return V
end
