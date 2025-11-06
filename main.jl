#=
    Main script
=#

using Parameters, Interpolations, Plots, LinearAlgebra, SparseArrays,
        Roots, Base.Threads, StatsBase

include("defineConstantsGridsEtc.jl")
include("helpFunctions.jl")

# Replace with named tuple?
@with_kw struct Params
    σ; β; α
end
params = Params(σ=2.0, β=0.96, α=0.36)

### EGM
#   Using a guess for k_dec(k⁻, e), r_tp1, w_tp1, we can compute
#   λ_t = β * E_t[(1 + r_tp1) * u_prime(c_dec(k_tp1, e_tp1))]
#   and then c_t = u_prime_inv(λ_t)
function iterateEGM(c_tp1, params_in, wₜ, rₜ, rₜ₊₁)
    @unpack σ, β = params_in
    _, u_prime, u_prime_inv = createUtilityFunctions(σ)

    # Given the consumption rule at t+1 per state (k⁻, e), compute
    # the Eₜ[λₜ₊₁(1+rₜ₊₁)] having chosen k = k⁻ in t
    c_decs = zeros(nA, nE); k_decs = zeros(nA, nE)

    for (idx_eₜ, eₜ) in enumerate(Egrid)
        
        c_endo = zeros(nA)
        for idx_kₜ in eachindex(Kgrid)
            # In t, hh was in eₜ and chose kₜ
            cₜ₊₁s = c_tp1[idx_kₜ, :]
            λₜ₊₁s = u_prime.(cₜ₊₁s)
            λₜ    = β * (1 + rₜ₊₁) * λₜ₊₁s'*Pₑ[idx_eₜ, :]
            cₜ    = u_prime_inv(λₜ)
            c_endo[idx_kₜ] = cₜ
        end
        # Use BC to get the starting k⁻ that made hh choose cₜ and kₜ
        k_endo = (c_endo .+ Kgrid .- eₜ*wₜ)./(1.0 + rₜ)

        # Interpolate to get c decisions on the exogenous Kgrid
        c_interp = LinearInterpolation(k_endo, c_endo, extrapolation_bc=Line())
        c_exog = c_interp.(Kgrid)
        k_exog = (1.0 + rₜ).*Kgrid .+ eₜ*wₜ .- c_exog

        # For the cases when borrowing constraint was violated, Euler equation
        # does not apply and we have to set kₜ = 0.0 and use budget constraint
        # to get cₜ
        bc_viol = k_exog .<= 0.0
        k_exog[bc_viol] .= kmin
        c_exog[bc_viol] .= (1 + rₜ)*Kgrid[bc_viol] .+ eₜ*wₜ .- kmin

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
        c_new, _ = iterateEGM(c_guess, params, wₛₛ, rₛₛ, rₛₛ)
        err = maximum(abs.(c_new .- c_guess))
        c_guess = c_new
        iter += 1
    end
    if iter == max_iter
        @warn "Warning: EGM did not converge in solveSSforHHProblem"
        kk
    end

    c_decs, k_decs = iterateEGM(c_guess, params, wₛₛ, rₛₛ, rₛₛ)

    return c_decs, k_decs, iter
end

function solveSSforHHProblem(wₛₛ, rₛₛ, params)
    c_guess = zeros(nA, nE)
    for i in 1:nA, j in 1:nE
        c_guess[i, j] = 0.5 * (wₛₛ*Egrid[j] + rₛₛ*Kgrid[i])
    end
    return solveSSforHHProblem(c_guess, wₛₛ, rₛₛ, params)
end

function calibrateModel(r_target, params_in)
    @unpack α = params_in

    # Compute labor supply (independent of the guess)
    Deₛₛ = inv_dist(Pₑ) # P(e = eⱼ) in steady state
    Lₛₛ = sum(Deₛₛ .* Egrid)

    # Get Kₛₛ / Yₛₛ from firm's FOCs
    KoverYₛₛ = α/r_target
    # Set Yₛₛ = 1 to get Kₛₛ   (only at steady state!)
    Kₛₛ = KoverYₛₛ
    # This implies what Zₛₛ has to be  (again, only at steady state!)
    Zₛₛ = 1 / (Kₛₛ^(1 - α) * Lₛₛ^(-α))

    # By firm's FOCs, we can get the steady state wage given r_target
    wₛₛ = (1-α)*Zₛₛ*(Kₛₛ/Lₛₛ)^α

    # Find the β so that the households' savings imply the correct Kₛₛ given
    # the current guess for rₛₛ
    function getKssGiven_r_target(β_guess)
        params_guess = Params(σ=params_in.σ, β=β_guess, α=params_in.α)
        _, k_ss, _ = solveSSforHHProblem(wₛₛ, r_target, params_guess)

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
    β_min = 0.90; β_max = 1/(1+r_target)*1.01
    β_sol = find_zero(getKssGiven_r_target, (β_min, β_max))
    #β_sol = optimize(getKssGiven_r_target, β_min, β_max).minimizer

    return β_sol, Kₛₛ, Zₛₛ, wₛₛ, Lₛₛ
end

rₛₛ = 0.04
β_calibrated, Kₛₛ, Zₛₛ, wₛₛ, Lₛₛ = calibrateModel(rₛₛ, params)

### Get household policy functions given calibrated parameters
params_calibrated = Params(σ=params.σ, β=β_calibrated, α=params.α)
c_dec_ss, k_dec_ss, _ = solveSSforHHProblem(wₛₛ, rₛₛ, params)
Dₛₛ = inv_dist(getTransitionMatrixFromPolicy(k_dec_ss))
tmp_Dₛₛ = vec(sum(reshape(Dₛₛ, nE, nA), dims=1))

##  Plot policy functions
sub_idx = Kgrid .<= 5.0
fig =
plot( Kgrid[sub_idx], c_dec_ss[sub_idx, 1],   label="c(e₁)")
plot!(Kgrid[sub_idx], c_dec_ss[sub_idx, 3],   label="c(e₂)")
plot!(Kgrid[sub_idx], c_dec_ss[sub_idx, end], label="c(eₙₑ)")
title!("Steady state consumption decisions");
xlabel!("Assets k"); ylabel!("Consumption c")
display(fig)

fig =
plot(Kgrid, tmp_Dₛₛ.*100.0, label=""); title!("Steady state distribution over assets")
xlabel!("Assets k"); ylabel!("Density (%)")
display(fig)