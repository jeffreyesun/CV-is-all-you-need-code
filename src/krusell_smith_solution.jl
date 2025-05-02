const N_K_BAR = 5
const K_BAR_GRID = Vector{FLOAT_PRECISION}(1000:500:3000)
const LOG_K_BAR_GRID = reshape(log.(K_BAR_GRID), (1,1,N_K_BAR))
const N_A = 2
function get_K_bar_ind(K_bar)
    i = K_bar/100 - 9
    i = min(i, N_K_BAR-1)
    i = max(i, 1)
    return i
end

"Convention: Ai_next is realized between V_end and V_next is evaluated"
function get_V_end_KS(K_bar, Λ_start, V_params_KS, a0, a1)
    (;Ai) = Λ_start
    K_bar_next = exp(a0[Ai] + a1[Ai]*log(K_bar))
    K_bar_next_ind = get_K_bar_ind(K_bar_next)
    K_bar_next_ind_floor = Int(floor(K_bar_next_ind))
    ρ = K_bar_next_ind - K_bar_next_ind_floor

    V_next = [(1-ρ).*V_params_KS[:,:,K_bar_next_ind_floor,Ai_next] + ρ.*V_params_KS[:,:,K_bar_next_ind_floor+1,Ai_next] for Ai_next=1:N_A]
    return V_end = Π[Ai,:]'V_next
end

function get_V_end_KS(md::ModelData, Λ_start, V_params_KS, a0, a1)
    return get_V_end_KS(get_K_bar(md), Λ_start, V_params_KS, a0, a1)
end

# Update a #
#----------#

function get_aV(path_data, Ai_test)
    (;V_start_path, Ai_path, λ_start_path) = path_data
    Ai_mask = Ai_path .== Ai_test
    Ai_t_vec = filter(!=(length(Ai_mask)), findall(Ai_mask))
    Ai_log_K_bar = log.([get_K_bar(λ_start_path[Ai_t]) for Ai_t in Ai_t_vec])
    Ai_log_K_bar_next = log.([get_K_bar(λ_start_path[Ai_t + 1]) for Ai_t in Ai_t_vec])

    X = hcat(ones(length(Ai_log_K_bar)), Ai_log_K_bar)

    y_a = Ai_log_K_bar_next
    a0_Ai, a1_Ai = (X'X)\(X'y_a)

    Ai_V_start_path = V_start_path[Ai_t_vec]
    Ai_V_start_i_path_vec = [getindex.(Ai_V_start_path, i) for i in eachindex(Ai_V_start_path[1])]
    
    y_V = reduce(hcat, Ai_V_start_i_path_vec)
    V_start_Ai = (X'X)\X'y_V
    
    return a0_Ai, a1_Ai, V_start_Ai
end

function update_aV(path_data)
    aV_vec = get_aV.(Ref(path_data), 1:N_A)
    a0 = getindex.(aV_vec, 1)
    a1 = getindex.(aV_vec, 2)
    V_start_new = getindex.(aV_vec, 3)

    V_params_KS = zeros(FLOAT_PRECISION, (N_K, N_Z, N_K_BAR, N_A))
    for Ai_set=1:N_A
        V_start_new_Ai = V_start_new[Ai_set][1,:] .+ V_start_new[Ai_set][2,:].*LOG_K_BAR_GRID
        V_params_KS[:,:,:,Ai_set] = reshape(V_start_new_Ai, (STATE_IDXs...,N_K_BAR))
    end

    return a0, a1, V_params_KS
end
