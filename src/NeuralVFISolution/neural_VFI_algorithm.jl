
"""
Find the global solution of the model using Neural VFI.

pretrain_V! - Pretrain the V_net using a ``dumb'' value function

train_epoch! - Train the V_net for one epoch

run_neural_VFI - Run the neural VFI algorithm
"""

function pretrain_V!(V_net, params; T=100, sample_t=50:100)
    pre_opt = Flux.setup(Adam(1e-8), V_net)

    md, path_data = simulate_path_dumbV(params, T, sample_t)
    (;λ_start_path, V_end_path, Ai_path) = path_data
    data = zip(λ_start_path, Ai_path, V_end_path)

    Flux.train!(V_net, data, pre_opt) do V_net, λ_start, Ai, V_end
        Λ_start = (;λ_start, Ai)
        return Flux.mse(vec(V_net(Λ_start)), vec(V_end))
    end
    return V_net
end

function train_epoch!(V_net, opt, params; T=100, sample_t=50:100, init_state=nothing)
    md, Λ_end, path_data = simulate_path_neuralV(params, V_net, T, sample_t; init_state)
    (;V_end_lookahead_path) = path_data

    (;λ_start_path, Ai_path) = path_data
    data = collect(zip(λ_start_path, Ai_path, V_end_lookahead_path))
    
    for i=1:10
        #batch = sample(data, length(sample_t)÷2)
        batch = data
        Flux.train!(V_net, batch, opt) do V_net, λ_start, Ai, V_end_lookahead
            Λ_start = (;λ_start, Ai)
            return Flux.mse(vec(V_net(Λ_start)), vec(V_end_lookahead))
        end
    end

    mses = (Flux.mse(V_net((;λ_start=d[1], Ai=d[2])), d[3]) for d=data)
    var_V = mean(var(vec(d[3]), ProbabilityWeights(vec(d[1]))) for d=data)
    last_error = mean(mses)/var_V
    init_state = (md, apply_aggregate_shock(Λ_end))
    return init_state, last_error
end

function run_neural_VFI(V_net, params; n_epochs=500)
    pretrain_V!(V_net, params)
    
    errors = []

    println("Training, Phase 1:")
    # Phase 1
    # Note: You might have to run this from `V_net = VNet()` a couple times.
    # Sometimes the initial VNet is not feasible and you get an error here
    opt = Flux.setup(Adam(1e-6), V_net)
    init_state = nothing
    for i=1:10
        if i%10 == 0
            md, Λ_end, _ = simulate_path_neuralV(params, V_net, 1000, [])
            init_state = (md, apply_aggregate_shock(Λ_end))
        end
        init_state, err = train_epoch!(V_net, opt, params; T=200, sample_t=100:2:200, init_state)
        push!(errors, err)
        print(i, " Loss: ")
        println(err)
    end

    # Phase 2
    println("Training, Phase 2:")
    opt = Flux.setup(Adam(1e-3), V_net)
    init_state = nothing

    for i=1:n_epochs
        if i%10 == 1
            md, Λ_end, _ = simulate_path_neuralV(params, V_net, 1000, []; init_state)
            md.λ_start ./= sum(md.λ_start)
            init_state = (md, apply_aggregate_shock(Λ_end))
        end
        init_state, err = train_epoch!(V_net, opt, params; T=100, sample_t=1:100, init_state)
        push!(errors, err)
        print(i, " Loss: ")
        println(err)
    end
    return errors
end
