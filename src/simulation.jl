

# Generate Training Data #
#------------------------#
"""
Simulate data using a ``dumb'' value function guess,
equal to the beginning-of-period value function.
"""
function simulate_path_dumbV(params::Params, T::Int, sample_t)
    # Initialize model
    md, Λ_start = initialize_model(params)
    # Initialize sample data
    state_samples = initialize_path()
    # Initialize V function guess
    V_start = md.V_end
    
    for t=1:T
        # Predict V_end in a ``dumb'' way, setting it equal to the current beginning-of-period value function
        V_end = V_start
        # Simulate period
        V_start, Λ_end = solve_within_period_problem!(md, V_end, Λ_start, params)
        # Possibly sample state
        t in sample_t && add_state_to_sample!(state_samples, md, Λ_end.Ai)
        # Update state
        Λ_start = apply_aggregate_shock(Λ_end)
    end
    return md, state_samples
end

"""
Simulate data using a neural network to predict V_end.
"""
function simulate_path_neuralV(params::Params, V_net::VNet, T::Int, sample_t; init_state=nothing)
    # Initialize model
    md, Λ_start = something(init_state, initialize_model(params))
    # Initialize sample and training data
    path_data = initialize_path()

    local Λ_end
    for t=1:T
        # Predict V_end using the neural network
        V_end = V_net(md, Λ_start.Ai)
        # Simulate period forward
        try
            V_start, Λ_end = solve_within_period_problem!(md, V_end, Λ_start, params)
        catch e
            throw("V_net is not feasible, please reinitialize and try again.")
        end
        # Possibly sample state and lookahead-predicted V_end
        t in sample_t && compute_and_store_training_data!(path_data, md, Λ_start, V_net, params)
        # Update state
        Λ_start = apply_aggregate_shock(Λ_end)
    end
    return md, Λ_end, path_data
end

"""
Simulate data using the Krusell-Smith value function approximator to predict V_end.
"""
function simulate_path_KSV(params, V_params, a0, a1; T=1000, sample_t=100:T)
    # Initialize model
    md, Λ_start = initialize_model(params)
    # Initialize sample data
    path_data = initialize_path()

    for t=1:T
        # Predict V_end using the Krusell-Smith value function approximator
        V_end = get_V_end_KS(md, Λ_start, V_params, a0, a1)
        # Simulate period forward
        V_start, Λ_end = solve_within_period_problem!(md, V_end, Λ_start, params)
        # Possibly sample state
        t in sample_t && add_state_to_sample!(path_data, md, Λ_end.Ai)
        # Update state
        Λ_start = apply_aggregate_shock(Λ_end)
    end
    return md, path_data
end

function compute_and_store_training_data!(training_data, md, Λ_start, V_net, params)
    (;Ai, λ_start) = Λ_start
    add_state_to_sample!(training_data, md, Ai)

    # Compute and store V_end_lookahead
    V_end_lookahead = get_V_end_lookahead(md, λ_start, Ai, V_net, params)
    push!(training_data.V_end_lookahead_path, V_end_lookahead)
    return training_data
end


# Guessing V_start_next from neural net #
#---------------------------------------#
function get_V_start_next(md, λ_start_next, Ai_next, V_net, params)
    @assert md.λ_start == λ_start_next
    V_end_next = V_net(md, Ai_next)
    return iterate_V(V_end_next, Ai_next, md, params)
end

function get_V_end_lookahead(md, λ_start_next, Ai, V_net, params)
    # Save current state and V_end
    V_end_save = copy(md.V_end)
    λ_start_save = copy(md.λ_start)

    # Look ahead to the following period
    md.λ_start .= λ_start_next

    # Simulate the next period for each possible aggregate shock A,
    # and compute the expected value of V_end, taking expectation over the
    # A
    V_end_lookahead = zeros(FLOAT_PRECISION, STATE_IDXs)
    for Ai_next in 1:length(A_vals)
        V_start_next = get_V_start_next(md, λ_start_next, Ai_next, V_net, params)
        V_end_lookahead += Π[Ai, Ai_next] * V_start_next
    end

    # Restore state and V_end to current period (rather than next period)

    md.λ_start .= λ_start_save
    iterate_V(V_end_save, Ai, md, params)
    return V_end_lookahead
end
