#=
    Main script
=#

using Parameters, Interpolations, Plots, LinearAlgebra, SparseArrays,
        Roots, Base.Threads, StatsBase, Printf

include("defineConstantsGridsEtc.jl")
include("helpFunctions.jl")

# Replace with named tuple?
@with_kw struct Params
    σ  = nothing; β  = nothing; α  = nothing;
    Zₛₛ = nothing; Kₛₛ = nothing; Lₛₛ = nothing;
    rₛₛ = nothing; wₛₛ = nothing
end
params = Params(σ=1.0, α=0.11)

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
        params_guess = Params(σ=params_in.σ, β=β_guess, α=params_in.α)
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
    params_final = Params(σ=params_in.σ, β=β_sol, α=params_in.α)
    cₛₛ, kₛₛ, _ = solveSSforHHProblem(wₛₛ, rₛₛ, params_final)
    Λₛₛ = getTransitionMatrixFromPolicy(kₛₛ)
    Dₛₛ = inv_dist(Λₛₛ)

    return β_sol, Zₛₛ, Kₛₛ, Lₛₛ, cₛₛ, kₛₛ, rₛₛ, wₛₛ, Λₛₛ, Dₛₛ
end

println("Calibrating the model and finding its steady state")
KoverY_target = 0.11/0.035
β_calibrated, Zₛₛ, Kₛₛ, Lₛₛ, cₛₛ, kₛₛ, rₛₛ, wₛₛ, Λₛₛ, Dₛₛ =
                    calibrateModel(KoverY_target, params)

params_calibrated = Params(β=β_calibrated, σ=params.σ, α=params.α,
                                                        rₛₛ=rₛₛ, wₛₛ=wₛₛ)

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
function simulateAtSteadyState(N_hhs, ss_in, T)
    @unpack cₛₛ, kₛₛ, Dₛₛ = ss_in
    
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

ss = (cₛₛ=cₛₛ, kₛₛ=kₛₛ, Zₛₛ=Zₛₛ, Kₛₛ=Kₛₛ, Lₛₛ=Lₛₛ, rₛₛ=rₛₛ, wₛₛ=wₛₛ, Dₛₛ=Dₛₛ)
K_path_sim = simulateAtSteadyState(100_000, ss, 500)

## What happens if we "simulate" using the transition matrix?
K_path_sim2 = zeros(length(K_path_sim))
K_path_sim2[1] = sum(Dₛₛ.*repeat(Kgrid, inner=nE))
Dₜ = Dₛₛ
for t = 2:length(K_path_sim2)
    Dₜ = (Λₛₛ')*Dₜ
    K_path_sim2[t] = sum(Dₜ.*repeat(Kgrid, inner=nE))
end

fig =
plot( 1:length(K_path_sim), K_path_sim, label="sim w/ hhs")
plot!(1:length(K_path_sim), fill(Kₛₛ, length(K_path_sim)), label="actual ss")
plot!(1:length(K_path_sim), K_path_sim2, label="sim using Λₛₛ", ls=:dash)
display(fig)

fig = histogram(K_path_sim2 .- Kₛₛ); display(fig)

################################################################################
### Sequence Space Jacobian to compute transition dynamics
################################################################################

### Get policy functions following a shock to r or w in T-1 (hhs become aware of
### the shock in perido 0) using Brute Force Method
function get_Ks_given_rws(r_path, w_path, params_in, c_dec_ss=nothing)

    # For when ForwardDiff passes its input
    this_type = eltype(r_path.+w_path)

    T = length(r_path)
    c_decs = zeros(this_type, T, nA, nE)
    k_decs = zeros(this_type, T, nA, nE)
    Ds     = zeros(this_type, T, nA*nE)
    Ks     = zeros(this_type, T)

    if c_dec_ss === nothing
        cₛₛ, kₛₛ, _ = solveSSforHHProblem(params_in)
        Dₛₛ = inv_dist(getTransitionMatrixFromPolicy(kₛₛ))
    else
        println("Implement!"); kk
    end
    
    # Whatever the length of the path, assume that we then return to the steady
    #   state
    # So, in T, households expect rₛₛ and use the steady state policy when
    # forming expectations. Wage and interest rate is given by w_path[T] and
    #   r_path[T]
    c_T, k_T = iterateEGM(cₛₛ, params_in, r_path[end], w_path[end], rₛₛ)
    c_decs[end, :, :] = c_T; k_decs[end, :, :] = k_T

    # Iterate backwards to get policies for t = T-1, T-2, ..., 1, 0
    cₜ₊₁ = c_T
    for t = T-1:-1:1
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
K_path = get_Ks_given_rws(r_path, w_path, params_calibrated)

fig = plot(K_path .- Kₛₛ)
title!("K_t - Kₛₛ when using general function to simulate")
display(fig)

### Adding a shock in period s = 1
dx = 10.0e-4
r_path = fill(rₛₛ, 300); r_path[1] = rₛₛ + dx
w_path = fill(wₛₛ, length(r_path))

K_path = get_Ks_given_rws(r_path, w_path, params_calibrated)

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
        K_path = get_Ks_given_rws(r_path, w_path, params_calibrated)
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
function to_diff(x) # ::AbstractVector{T}) where T
    return get_Ks_given_rws(x, w_path, params_calibrated)
end
y = to_diff(r_path)

J_Kr_v2 = ForwardDiff.jacobian(to_diff, r_path)