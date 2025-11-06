#=
    This script defines several constants
=#

module DefineConstantsGridsEtc
    export ρₑ, σₑ, δ, nE, nA, kmin, kmax, Kgrid, Egrid, Pₑ,
            createUtilityFunctions

    using Distributions

    ρₑ = 0.966; σₑ = 0.5

    δ = 0.025

    nE = 7; nA = 500; kmin = 0.0; kmax = 200.0

    ### Exponential grid for capital holdings
    Kgrid = exp.(range(0.0, log(kmax-kmin+1.0), length=nA)) .+ kmin .- 1.0

    ### Tauchen method to discretize the endowement process
    #   As described at https://julia.quantecon.org/introduction_dynamics/finite_markov.html
    σ_ϵ = σₑ * sqrt(1 - ρₑ^2) # Standard deviation of the i.i.d. shocks
    m = 1; sₑ = 2*m*σₑ/(nE-1) # b/c it's std. dev. of e that is specified
    Egrid = range(-m*σₑ, stop=m*σₑ, step=sₑ)

    Pₑ = zeros(nE, nE); Φ(x) = cdf(Normal(0.0, σ_ϵ), x)
    for j = 1:nE
        if     j == 1
            Pₑ[:, j] .=      Φ((Egrid[1]  .- ρₑ*Egrid .+ sₑ/2))
        elseif j == nE
            Pₑ[:, j] .= 1 .- Φ((Egrid[nE] .- ρₑ*Egrid .- sₑ/2))
        else
            Pₑ[:, j] .=      Φ((Egrid[j]  .- ρₑ*Egrid .+ sₑ/2)) .-
                             Φ((Egrid[j]  .- ρₑ*Egrid .- sₑ/2))
        end
    end
    Egrid = exp.(Egrid)

    function createUtilityFunctions(σ_in)
        if σ_in == 1.0
            u = c -> c <= 0.0 ? -1.0e12 : log(c)
        else
            u = c -> c <= 0.0 ? -1.0e12 : (c^(1 - σ_in) - 1.0) / (1.0 - σ_in)
        end
        u_prime(c)       = c  <= 0.0    ? 1.0e12 :   c^(-σ_in)
        u_prime_inv(u_p) = u_p > 9.9e11 ? 0.0    : u_p^(-1/σ_in)

        return u, u_prime, u_prime_inv
    end
end

using .DefineConstantsGridsEtc