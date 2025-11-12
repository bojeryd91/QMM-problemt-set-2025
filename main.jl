#=
    Main script
=#

using Parameters, Interpolations, Plots, LinearAlgebra, SparseArrays,
        Roots, Base.Threads, StatsBase, Printf

include("defineConstantsGridsEtc.jl")
include("helpFunctions.jl")

const_params = (σ=1.0, α=0.11)

### EGM
#   Using a guess for k_dec(k⁻, e), r_tp1, w_tp1, we can compute
#   λ_t = β * E_t[(1 + r_tp1) * u_prime(c_dec(k_tp1, e_tp1))]
#   and then c_t = u_prime_inv(λ_t)
function iterateEGM(cₜ₊₁s, params_in, rₜ, wₜ, rₜ₊₁)
    @unpack σ, β = params_in
    _, u_prime, u_prime_inv = createUtilityFunctions(σ)

    # For when ForwardDiff passes inputs
    this_type = eltype(cₜ₊₁s[1] + rₜ + wₜ + rₜ₊₁)

    # Given the consumption rule at t+1 per state (k⁻, e), compute
    # the Eₜ[λₜ₊₁(1+rₜ₊₁)] having chosen k = k⁻ in t
    c_decs = zeros(this_type, nA, nE)
    k_decs = zeros(this_type, nA, nE)

    for (idx_eₜ, eₜ) in enumerate(Egrid)
        
        c_endo = zeros(this_type, nA); k_endo = zeros(this_type, nA)
        for (idx_kₜ, kₜ) in enumerate(Kgrid)
            # In t, hh was in eₜ and chose kₜ
            cₜ₊₁s_by_e = cₜ₊₁s[idx_kₜ, :]
            λₜ₊₁s = u_prime.(cₜ₊₁s_by_e)
            # Period-t marginal utility given they chose kₜ:
            λₜ    = β * (1 + rₜ₊₁ - δ) * λₜ₊₁s'*Pₑ[idx_eₜ, :]
            cₜ    = u_prime_inv(λₜ)
            c_endo[idx_kₜ] = cₜ
            # Use BC to get period-t's k⁻ that made hh choose cₜ and kₜ
            k_endo[idx_kₜ] = (cₜ .+ kₜ .- eₜ*wₜ)./(1.0 + rₜ - δ)
        end

        # Interpolate to get c decisions on the exogenous Kgrid
        c_interp = LinearInterpolation(k_endo, c_endo, extrapolation_bc=Line())
        c_exog = c_interp.(Kgrid)
        k_exog = (1.0 + rₜ - δ).*Kgrid .+ eₜ*wₜ .- c_exog

        # For the cases when borrowing constraint was violated, Euler equation
        # does not apply and we have to set kₜ = kmin and use budget constraint
        # to get cₜ
        bc_viol = k_exog .< kmin
        k_exog[bc_viol] .= kmin
        c_exog[bc_viol] .= (1 + rₜ - δ)*Kgrid[bc_viol] .+ eₜ*wₜ .- kmin

        # Save to result matrices
        c_decs[:, idx_eₜ] = c_exog
        k_decs[:, idx_eₜ] = k_exog
    end
    return c_decs, k_decs
end

function solveSSforHHProblem(c_guess, wₛₛ, rₛₛ, params)
    max_iter = 1000; tol = 1.0e-6;
    iter = 0; err = 1.0e10

    while iter < max_iter && err > tol
        c_new, _ = iterateEGM(c_guess, params, rₛₛ, wₛₛ, rₛₛ)
        err = maximum(abs.(c_new .- c_guess))
        c_guess = c_new
        iter += 1
    end
    if iter == max_iter
        @warn "Warning: EGM did not converge in solveSSforHHProblem"
        kk
    end

    c_decs, k_decs = iterateEGM(c_guess, params, rₛₛ, wₛₛ, rₛₛ)

    return c_decs, k_decs, iter
end

function solveSSforHHProblem(wₛₛ, rₛₛ, params_in)
    # Construct a guess
    c_guess = zeros(nA, nE)
    for i in 1:nA, j in 1:nE
        c_guess[i, j] = 0.5 * (wₛₛ*Egrid[j] + rₛₛ*Kgrid[i])
    end
    return solveSSforHHProblem(c_guess, wₛₛ, rₛₛ, params_in)
end

function solveSSforHHProblem(params_in)
    @unpack rₛₛ, wₛₛ = params_in
    return solveSSforHHProblem(wₛₛ, rₛₛ, params_in)
end

function calibrateModel(KoverY_target, params_in)
    @unpack α = params_in

    # Compute labor supply (this is independent of the guess)
    Deₛₛ = inv_dist(Pₑ) # P(e = eⱼ) in steady state
    Lₛₛ = sum(Deₛₛ .* Egrid)

    # Get rₛₛ from firm's capital FOC
    rₛₛ = α/KoverY_target  # r = α*Y/K
    # Set Yₛₛ = 1 to get Kₛₛ   (only at steady state!)
    Kₛₛ = KoverY_target
    # This implies what Zₛₛ has to be  (again, only at steady state!)
    Zₛₛ = 1 / (Kₛₛ^α * Lₛₛ^(1.0-α))

    # By firm's FOCs, we can get the steady state wage given Lₛₛ
    wₛₛ = (1-α)*1.0/Lₛₛ

    # Find the β so that the households' savings imply the correct Kₛₛ given
    # the current guess for rₛₛ
    function getKssGivenTarget(β_guess)
        params_guess = merge(params_in, (; β=β_guess))
        _, k_ss, _ = solveSSforHHProblem(wₛₛ, rₛₛ, params_guess)

        ## Compute aggregate capital stock given the policy function implied
        #   by the current guess of (β, r)
        # (1) Get steady state distribution 
        Λ_impl = getTransitionMatrixFromPolicy(k_ss)
        Dₛₛ = inv_dist(Λ_impl)
        # (2) Compute aggregate capital stock
        # Dₛₛ = [P(k⁻₁, e₁); P(k⁻₁, e₂); ⋮; P(k⁻ₙₐ, eₙₑ)]
        K_agg = sum(Dₛₛ.*repeat(Kgrid, inner=nE))
        
        return K_agg/Kₛₛ - 1.0
    end
    β_min = 0.90; β_max = 1/(1+rₛₛ-δ)
    β_sol = find_zero(getKssGivenTarget, (β_min, β_max))

    # Get transition matrix and invariant distribution too
    params_final = merge(params_in, (; β=β_sol))
    cₛₛ, kₛₛ, _ = solveSSforHHProblem(wₛₛ, rₛₛ, params_final)
    Λₛₛ = getTransitionMatrixFromPolicy(kₛₛ)
    Dₛₛ = inv_dist(Λₛₛ)

    cali_params = (; Zₛₛ, Kₛₛ, Lₛₛ, cₛₛ, kₛₛ, rₛₛ, wₛₛ, Λₛₛ, Dₛₛ, α)
    return merge(cali_params, (; β=β_sol, σ=params_in.σ))
end

println("Calibrating the model and finding its steady state")
KoverY_target = 0.11/0.035
calibrated_model = calibrateModel(KoverY_target, const_params)
@unpack (Zₛₛ, Kₛₛ, Lₛₛ, cₛₛ, kₛₛ, Dₛₛ, Λₛₛ, rₛₛ, wₛₛ, α, σ) = calibrated_model

### Check aggregates and look at policy functions
tmp_Dₛₛ = vec(sum(reshape(Dₛₛ, nE, nA), dims=1))

##  Plot policy functions
sub_idx = Kgrid .<= 5.0
fig =
plot( Kgrid[sub_idx], cₛₛ[sub_idx, 1],   label="c(e₁)")
plot!(Kgrid[sub_idx], cₛₛ[sub_idx, 3],   label="c(e₂)")
plot!(Kgrid[sub_idx], cₛₛ[sub_idx, end], label="c(eₙₑ)")
title!("Steady state consumption decisions");
xlabel!("Assets k"); ylabel!("Consumption c")
display(fig)

fig =
plot(Kgrid, tmp_Dₛₛ.*100.0, label="");
title!("Steady state distribution over assets")
xlabel!("Assets k"); ylabel!("Density (%)")
display(fig)

sub_idx = Kgrid .<= 30.0
fig =
plot(Kgrid[sub_idx], tmp_Dₛₛ[sub_idx].*100.0, label="");
title!("Steady state distribution over assets")
xlabel!("Assets k"); ylabel!("Density (%)")
display(fig)

################################################################################
### Simulate model at steady state
################################################################################
function simulateAtSteadyState(N_hhs, model_in, T)
    @unpack cₛₛ, kₛₛ, Dₛₛ = model_in
    
    ### Interpolation objects
    k_itp = LinearInterpolation((Kgrid, Egrid), kₛₛ, extrapolation_bc=Line())

    ### Simulate each household over T periods
    CDF_Dₛₛ = cumsum(Dₛₛ)
    all_ks = zeros(N_hhs, T+1)
    idx_k₀s = Array{Int}(undef, N_hhs)
    @threads for ID in 1:N_hhs

        ### Sample initial states
        i_state = searchsortedfirst(CDF_Dₛₛ, rand())
        idx_k₀ = ceil(Int, i_state/nE)
        idx_k₀s[ID] = idx_k₀
        idx_eₜ = i_state - (idx_k₀ - 1)*nE
        eₜ = Egrid[idx_eₜ]; k⁻ₜ = Kgrid[idx_k₀]

        # Initialize storage
        ks = zeros(T+1)
        ks[1] = k⁻ₜ

        @inbounds for t in 2:T+1
            # Get saving decision today
            kₜ = k_itp(k⁻ₜ, eₜ)

            # Sample tomorrow's e state
            this_CDF_e = cumsum(Pₑ[idx_eₜ, :])
            idx_eₜ₊₁ = searchsortedfirst(this_CDF_e, rand())
            eₜ = Egrid[idx_eₜ₊₁]; idx_eₜ = idx_eₜ₊₁

            # Store results
            ks[t] = kₜ; k⁻ₜ   = kₜ
        end
        all_ks[ID, :] = ks
    end

    return vec(sum(all_ks, dims=1)./N_hhs)
end

K_path_sim = simulateAtSteadyState(100_000, calibrated_model, 500)

## What happens if we "simulate" using the transition matrix?
K_path_sim2 = zeros(length(K_path_sim))
K_path_sim2[1] = sum(Dₛₛ.*repeat(Kgrid, inner=nE))
Dₜ = Dₛₛ
for t = 2:lastindex(K_path_sim2)
    Dₜ = (Λₛₛ')*Dₜ
    K_path_sim2[t] = sum(Dₜ.*repeat(Kgrid, inner=nE))
end

fig =
plot( 1:length(K_path_sim), K_path_sim,  label="sim w/ hhs")
plot!(1:length(K_path_sim), fill(Kₛₛ, length(K_path_sim)),
                                         label="actual ss")
plot!(1:length(K_path_sim), K_path_sim2, label="sim using Λₛₛ", ls=:dash)
display(fig)

fig = histogram(K_path_sim2 .- Kₛₛ); display(fig)

################################################################################
### Sequence Space Jacobian to compute transition dynamics
################################################################################

### Get policy functions following a shock to r or w in T-1 (hhs become aware of
### the shock in perido 0) using Brute Force Method
function get_Ks_given_rws(r_path, w_path, params_in)

    @unpack cₛₛ, kₛₛ, Dₛₛ, rₛₛ = params_in

    # For when ForwardDiff passes its input
    this_type = eltype(r_path.+w_path)

    T = length(r_path)
    c_decs = zeros(this_type, T, nA, nE)
    k_decs = zeros(this_type, T, nA, nE)
    Ds     = zeros(this_type, T, nA*nE)
    Ks     = zeros(this_type, T)
    
    # Whatever the length of the path, assume that we then return to the steady
    #   state
    # So, in T, households expect rₛₛ and use the steady state policy when
    # forming expectations. Wage and interest rate is given by w_path[T] and
    #   r_path[T]
    c_T, k_T = iterateEGM(cₛₛ, params_in, r_path[end], w_path[end], rₛₛ)
    c_decs[end, :, :] = c_T; k_decs[end, :, :] = k_T

    # Iterate backwards to get policies for t = T-1, T-2, ..., 1, 0
    cₜ₊₁ = c_T
    for t = Iterators.reverse(1:T-1)
        cₜ, kₜ = iterateEGM(cₜ₊₁, params_in, r_path[t], w_path[t], r_path[t+1])
        c_decs[t, :, :] = cₜ; k_decs[t, :, :] = kₜ
        cₜ₊₁ = cₜ
    end

    # Iterate forwards to get distributions for t = 0, 1, ..., T
    Dₜ = Dₛₛ
    for t = 1:T
        Λₜ = getTransitionMatrixFromPolicy(k_decs[t, :, :])
        Dₜ = (Λₜ')*Dₜ
        Ds[t, :] = Dₜ
        Ks[t] = sum(Dₜ.*repeat(Kgrid, inner=nE))
    end

    return Ks
end

### Re-do steady-state simulation using above command
r_path = fill(rₛₛ, 300); w_path = fill(wₛₛ, length(r_path))
K_path = get_Ks_given_rws(r_path, w_path, calibrated_model)

fig = plot(K_path .- Kₛₛ)
title!("K_t - Kₛₛ when using general function to simulate")
display(fig)

### Adding a shock in period s = 1
dx = 10.0e-4
r_path = fill(rₛₛ, 300); r_path[1] = rₛₛ + dx
w_path = fill(wₛₛ, length(r_path))

K_path = get_Ks_given_rws(r_path, w_path, calibrated_model)

fig = plot(K_path, label="Kₜ")
plot!(fill(Kₛₛ, length(K_path)), label="Kₛₛ", ls=:dash)
display(fig)

################################################################################
### Compute Jacobian Jᴷʳₜ₀ using Brute Force method
T = 300; dx = rₛₛ*0.01

#=
### Iterate over all s ∈ {1, ..., T}
function getJacobianBF()
    done_counter = 0
    K_paths = Array{Float64}(undef, T, T)
    print("\e[2K\e[1G0.0% done")
    @threads for i_s in 1:T
        r_path = fill(rₛₛ, 300); r_path[i_s] = rₛₛ + dx
        K_path = get_Ks_given_rws(r_path, w_path, calibrated_model)
        K_paths[:, i_s] = K_path
        done_counter += 1
        str = @sprintf("%3.2f%% done", done_counter/T*100)
        print("\e[2K\e[1G", str)
    end
    print("\e[2K\e[1G100.0% done\n")
    return (K_paths .- Kₛₛ)./dx 
end

@time getJacobianBF

J_Kr_w_BF = getJacobianBF();

fig =
plot( J_Kr_w_BF[:,   1], label="s=1")
plot!(J_Kr_w_BF[:,  25], label="s=25")
plot!(J_Kr_w_BF[:,  50], label="s=50")
plot!(J_Kr_w_BF[:,  75], label="s=75")
plot!(J_Kr_w_BF[:, 100], label="s=100")
title!("Using manual differentiation")
display(fig)
=#

################################################################################
### Trying auto differentiation in only rₛ
using ForwardDiff
#=
r_path = fill(rₛₛ, 300); w_path = fill(wₛₛ, length(r_path))
function to_diff(rw_in)
    return get_Ks_given_rws(rw_in[1:T], rw_in[T+1:end], calibrated_model)
end
y = to_diff(vcat(r_path, w_path))

function getJacobianBF_w_ForwardDiff()
    return ForwardDiff.jacobian(to_diff, vcat(r_path, w_path))
end

@time getJacobianBF_w_ForwardDiff()

J_K_BF_w_ForwardDiff = getJacobianBF_w_ForwardDiff()
J_Kr = J_K_BF_w_ForwardDiff[:,   1:T]
J_Kw = J_K_BF_w_ForwardDiff[:, T+1:end]

fig =
plot( J_Kr[:,   1], label="s=1")
plot!(J_Kr[:,  25], label="s=25")
plot!(J_Kr[:,  50], label="s=50")
plot!(J_Kr[:,  75], label="s=75")
plot!(J_Kr[:, 100], label="s=100")
title!("Jᴷʳₜₛ, using ForwardDiff")
display(fig)

fig =
plot( J_Kw[:,   1], label="s=1")
plot!(J_Kw[:,  25], label="s=25")
plot!(J_Kw[:,  50], label="s=50")
plot!(J_Kw[:,  75], label="s=75")
plot!(J_Kw[:, 100], label="s=100")
title!("Jᴷʷₜₛ, using ForwardDiff")
display(fig)
=#
################################################################################
### Construct H matrices and compute impulse response do standard dev.
#   shock to TFP
#=
α = params_calibrated.α
∂rₜ∂Kₜ =    α *(α-1)*Zₛₛ*(Kₛₛ)^(α-2)*(Lₛₛ)^(-α+1)
∂wₜ∂Kₜ = (1-α)*( -α)*Zₛₛ*(Kₛₛ)^(α)  *(Lₛₛ)^(-α-1)
H_K = J_Kr.*∂rₜ∂Kₜ .+ J_Kw.*∂wₜ∂Kₜ - I

∂rₜ∂Zₜ =    α *(Kₛₛ)^(α-1)*(Lₛₛ)^(1-α)
∂wₜ∂Zₜ = (1-α)*(Kₛₛ)^(α)  *(Lₛₛ)^( -α)
H_Z   = J_Kr*∂rₜ∂Zₜ .+ J_Kw*∂wₜ∂Zₜ - I

### Produce shock sequence
ρ = 0.9; shock_size = 0.01; i_s=20
logZₜ = fill(log(Zₛₛ), T); logZₜ[i_s] = log(Zₛₛ) + shock_size
for i_t = i_s+1:T
    logZₜ[i_t] = (1-ρ)*log(Zₛₛ) + ρ*logZₜ[i_t-1]
end
Z_path = exp.(logZₜ)
plot(Z_path)
dz = Z_path .- Zₛₛ

G  = -inv(H_K)*H_Z
dK = G*dz
plot(dK)

K_path = Kₛₛ .+ dK
r_path = Z_path.*(K_path).^(α-1.0).*(α*Lₛₛ^(1-α))
w_path = Z_path.*(K_path).^(α).*((1-α)*Lₛₛ^(-α))

T_disp = 50
plot( (Z_path[1:T_disp] .- Zₛₛ)./Zₛₛ, label="Dev. in Zₜ")
plot!((K_path[1:T_disp] .- Kₛₛ)./Kₛₛ, label="Dev. in Kₜ")
plot!((r_path[1:T_disp] .- rₛₛ)./rₛₛ, label="Dev. in rₜ")
plot!((w_path[1:T_disp] .- wₛₛ)./wₛₛ, label="Dev. in wₜ")
title!("Deviations from steady state")

fig =
plot(0:(T_disp-1), G[1:T_disp, [5,10,15,20,25]],
            labels = ["s=5" "s=10"	"s=15" "s=20" "s=25"])
title!("News shock at time s")
display(fig)
=#
################################################################################
### Using Fake News algorithm
function get_Ys_and_Ds(rw_in, params_in, c_dec_ss=nothing)

    # For when ForwardDiff passes its input
    this_type = eltype(rw_in)

    if c_dec_ss === nothing
        cₛₛ, kₛₛ, _ = solveSSforHHProblem(params_in)
        Dₛₛ = inv_dist(getTransitionMatrixFromPolicy(kₛₛ))
    else
        println("Implement!"); kk
    end

    cₜ₊₁   = zeros(this_type, size(cₛₛ)); cₜ₊₁ .= cₛₛ
    K_path = zeros(this_type, T)
    D_path = zeros(this_type, length(Dₛₛ), T)
    
    # Given a change in period T's interest rate or wage, compute backwards
    # the changes in the aggregate K and distribution D
    for (i_t, t) in enumerate(Iterators.Reverse(1:T))
        if t == T # In the last period/first iteration, everything is back to
                  # steady state but today's wage or interest is different
            cₜ₊₁, kₜ = iterateEGM(cₛₛ,  params_in, rw_in[1], rw_in[2], rₛₛ)
        elseif t == T-1 # In the penultimate period/second iteration,
                        # today's wage and r are at steady state but
                        # tomorrow's interest rate might be different
            cₜ₊₁, kₜ = iterateEGM(cₜ₊₁, params_in, rₛₛ,      wₛₛ,      rw_in[1])
        else # Otherwise, wages and interest rates are at steady state but
             # tomorrow's policy function is different
            cₜ₊₁, kₜ = iterateEGM(cₜ₊₁, params_in, rₛₛ,      wₛₛ,      rₛₛ)
        end
        
        # Use '[:] to reshape k_t from 𝐑ⁿᵃ×𝐑ⁿᵉ to 𝐑ⁿᵉ ⁿᵃ, sorted first by e,
        # just like Dₛₛ
        K_path[i_t] = (kₜ'[:])'*Dₛₛ

        # Compute new in distribution using this periods transition matrix
        Λₜ = getTransitionMatrixFromPolicy(kₜ)
        D_path[:, i_t] .= Λₜ'*Dₛₛ
    end
    
    return vcat(K_path, D_path[:])
end

function get_J(Ks, Ds)
    ### Construct the Fake News matrix
    F = zeros(T, T)
    F[1, :] = Ks[:]
    E = kₛₛ'[:]
    for i_t = 2:T
        for i_s = 1:T
            F[i_t, i_s] = E'*Ds[:, :, i_s][:]
        end
        E = Λₛₛ*E
    end

    ### Construct the Jacobian using F
    J = zeros(T, T)
    J[1, :] .= F[1, :]; J[:, 1] .= F[:, 1]
    for i_s = 2:T
        @views J[2:T, i_s] .= J[1:T-1, i_s-1] .+ F[2:T, i_s]
    end

    return J
end

to_diff = function(rw_in)
    return get_Ys_and_Ds(rw_in, calibrated_model)
end
res = ForwardDiff.jacobian(to_diff, vcat(rₛₛ, wₛₛ))
Ks_r = res[1:T, 1]; Ks_w = res[1:T, 2]
Ds_r = reshape(res[T+1:end, 1], nE, nA, T)
Ds_w = reshape(res[T+1:end, 2], nE, nA, T)

Jᵏʳₜₛ = get_J(Ks_r, Ds_r)
Jᵏʷₜₛ = get_J(Ks_w, Ds_w)

#=
plot(F[:, 1])
plot( F[:, 25])
plot!(F[:, 50])
plot!(F[:, 75])
plot!(F[:, 100])
=#
fig =
plot( Jᵏʳₜₛ[:, 1])
plot!(Jᵏʳₜₛ[:, 26])
plot!(Jᵏʳₜₛ[:, 51])
plot!(Jᵏʳₜₛ[:, 76])
plot!(Jᵏʳₜₛ[:, 101])
display(fig)

################################################################################
### Construct H matrices and compute impulse response do standard dev.
#   shock to TFP
∂rₜ₊₁∂Kₜ =    α *(α-1)*Zₛₛ*(Kₛₛ)^(α-2)*(Lₛₛ)^(-α+1)
∂wₜ∂Kₜ   = (1-α)*( -α)*Zₛₛ*(Kₛₛ)^(α)  *(Lₛₛ)^(-α-1)
H_K      = Jᵏʳₜₛ.*∂rₜ₊₁∂Kₜ .+ Jᵏʷₜₛ.*∂wₜ∂Kₜ - I

∂rₜ₊₁∂Zₜ =    α *(Kₛₛ)^(α-1)*(Lₛₛ)^(1-α)
∂wₜ∂Zₜ   = (1-α)*(Kₛₛ)^(α)  *(Lₛₛ)^( -α)
H_Z      = Jᵏʳₜₛ*∂rₜ₊₁∂Zₜ .+ Jᵏʷₜₛ*∂wₜ∂Zₜ - I

### Produce shock sequence
#   log(Zₜ) = (1-ρ)⋅log(Zₛₛ) + ρ⋅log(Zₜ₋₁) + εₜ
ρ = 0.9; shock_size = 0.01; i_s=20
logZₜ = fill(log(Zₛₛ), T); logZₜ[i_s] = log(Zₛₛ) + shock_size
for i_t = i_s+1:T
    logZₜ[i_t] = (1-ρ)*log(Zₛₛ) + ρ*logZₜ[i_t-1]
end
Z_path = exp.(logZₜ)
plot(Z_path)
dz = Z_path .- Zₛₛ

G  = -inv(H_K)*H_Z
dK = G*dz
plot(dK)

K_path = Kₛₛ .+ dK
r_path = Z_path.*(K_path).^(α-1.0).*(   α *Lₛₛ^(1-α))
w_path = Z_path.*(K_path).^(α    ).*((1-α)*Lₛₛ^( -α))

T_disp = 50
fig =
plot( (Z_path[1:T_disp] .- Zₛₛ)./Zₛₛ, label="Dev. in Zₜ")
plot!((K_path[1:T_disp] .- Kₛₛ)./Kₛₛ, label="Dev. in Kₜ")
plot!((r_path[1:T_disp] .- rₛₛ)./rₛₛ, label="Dev. in rₜ")
plot!((w_path[1:T_disp] .- wₛₛ)./wₛₛ, label="Dev. in wₜ")
title!("Deviations from steady state")
display(fig)

fig =
plot(0:(T_disp-1), G[1:T_disp, [5,10,15,20,25]],
            labels = ["s=5" "s=10"	"s=15" "s=20" "s=25"])
title!("News shock at time s")
display(fig)