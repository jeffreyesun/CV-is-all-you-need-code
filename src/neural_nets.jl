
using Flux
import Flux.Scale

# Input Matrix #
#--------------#
function get_HH_in_mat()
    k_mat = zeros(Float32, STATE_IDXs)
    k_mat .= range(0, 1, length=N_K)
    z_mat = zeros(Float32, STATE_IDXs)
    z_mat .= range(0, 1, length=N_Z)'
    HH_in_mat = zeros(Float32, 2, prod(STATE_IDXs))
    HH_in_mat[1,:] .= vec(k_mat)
    HH_in_mat[2,:] .= vec(z_mat)
    return HH_in_mat
end

# Neural Nets #
get_GM_net() = Chain(
    Dense(2, 10, elu),
    Dense(10, 10, elu)
)
get_V_net(::Val{false}, ::Val{false}) = Chain(
    Parallel((a,b,c)->a.+b.+c, (
        Dense(10, 10, elu),
        Dense(2, 10, elu),
        Chain(
            Dense(1, 10, elu),
            Dense(10,10,elu),
        ),
    )),
    Dense(10, 8, elu),
    Dense(8, 5, elu),
    Dense(5, 1),
    Scale([100], [500]),
)

get_V_net(::Val{true}, ::Val{false}) = Chain(
    Parallel((a,b,c)->a.+b.+c, (
        Dense(10, 10, elu),
        Dense(2, 10, elu),
        Chain(
            Dense(1, 10, elu),
            Dense(10,10,elu),
        ),
    )),
    Dense(10, 15, elu),
    Dense(15, 10, elu),
    Dense(10, 1),
    Scale([100], [500]),
)

get_V_net(::Val{false}, ::Val{true}) = Chain(
    Parallel((a,b,c)->a.+b.+c, (
        Dense(10, 10, elu),
        Dense(2, 10, elu),
        Chain(
            Dense(1, 10, elu),
            Dense(10,10,elu),
        ),
    )),
    Dense(10, 8, elu),
    Dense(8, 8, elu),
    Dense(8, 5, elu),
    Dense(5, 1),
    Scale([100], [500]),
)

get_V_net(::Val{true}, ::Val{true}) = Chain(
    Parallel((a,b,c)->a.+b.+c, (
        Dense(10, 10, elu),
        Dense(2, 10, elu),
        Chain(
            Dense(1, 10, elu),
            Dense(10,10,elu),
        ),
    )),
    Dense(10, 15, elu),
    Dense(15, 15, elu),
    Dense(15, 10, elu),
    Dense(10, 1),
    Scale([100], [500]),
)

struct VNet{M,C,S}
    HH_in_mat::M
    GM_net::C
    V_net::S
end
function VNet(;wide=false, deep=true, seed=9483)
    Random.seed!(seed)
    HH_in_mat = get_HH_in_mat()
    GM_net = get_GM_net()
    V_net = get_V_net(Val(wide), Val(deep))
    return VNet(HH_in_mat, GM_net, V_net)
end
Flux.trainable(V_net::VNet) = (;V_net.GM_net, V_net.V_net)

function (V_net::VNet)(λ_start, Ai)
    (;HH_in_mat, GM_net, V_net) = V_net
    GM_mat = GM_net(HH_in_mat)
    GM = GM_mat*vec(λ_start)

    return reshape(V_net((GM, HH_in_mat, [Ai])), STATE_IDXs)
end
(V_net::VNet)(as::ModelData, Ai::Int) = V_net(as.λ_start, Ai)

Flux.@layer VNet
