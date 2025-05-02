import Distributions: Categorical

##############
# Simulation #
##############

iterate_A(Ai) = rand(Categorical(Π[Ai,:]))
function apply_aggregate_shock(Λ_end)
    (;Ai, λ_end) = Λ_end
    Ai_next = iterate_A(Ai)
    return Λ_start = (;Ai=Ai_next, λ_start=λ_end)
end

function add_state_to_sample!(path_data, md, Ai)
    (;V_end_path, V_start_path, λ_start_path, Ai_path) = path_data
    
    V_end_save = copy(md.V_end)
    V_start_save = copy(md.V_income)
    λ_start_save = copy(md.λ_start)
    Ai_save = Ai

    push!(V_end_path, V_end_save)
    push!(V_start_path, V_start_save)
    push!(λ_start_path, λ_start_save)
    push!(Ai_path, Ai_save)
    return path_data
end
