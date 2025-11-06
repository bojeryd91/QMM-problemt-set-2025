#=
    Help functions
=#

#=
    Source: https://discourse.julialang.org/t/stationary-distribution-with-sparse-transition-matrix/40301/3
=#
function inv_dist(Π)
	#Π is a Stochastic Matrix
    x = [1; (I - Π'[2:end,2:end]) \ Vector(Π'[2:end,1])]
    return  x./sum(x) #normalize so that vector sums up to 1.
end

#=  
    Construct an (nA⋅nE)×(nA⋅nE) transition matrix using the policy function
    k_dec.
    Dₜ₊₁ = Λₜ' Dₜ

    where each element is defined as
    Λ = [P((k⁻₁, e₁)  → (k⁻₁, e₁))  P((k⁻₁, e₁)  → (k⁻₁, e₂)) ⋯  P((k⁻₁, e₁)  → (k⁻ₙₐ, eₙₑ));
         P((k⁻₁, e₂)  → (k⁻₁, e₁))  P((k⁻₁, e₂)  → (k⁻₁, e₂)) ⋯  P((k⁻₁, e₂)  → (k⁻ₙₐ, eₙₑ));
         ⋮                          ⋮                          ⋱ ⋮
         P((k⁻ₙₐ, eₙₑ) → (k⁻₁, e₁)) P((k⁻ₙₐ, eₙₑ) → (k⁻₁, e₂))  ⋯ P((k⁻ₙₐ, eₙₑ) → (k⁻ₙₐ, eₙₑ))]
        (each column maps to same state)
    D = [P(k⁻₁, e₁)
         P(k⁻₁, e₂)
         ⋮
         P(k⁻ₙₐ, eₙₑ)] 
=#
function getTransitionMatrixFromPolicy(k_dec)
    Λ = zeros(nA*nE, nA*nE)

    for idx_e in eachindex(Egrid), idx_k⁻ in eachindex(Kgrid)
        k = k_dec[idx_k⁻, idx_e]

        # Find the index of the asset grid point just below k_next
        idx_k_below = findlast(Kgrid .<= k)
        idx_k_above = idx_k_below + 1
        
        k_below = Kgrid[idx_k_below]
        if idx_k_above > nA
            k_above = kmax
            w_above = 1.0
            idx_k_above = nA
        else
            k_above = Kgrid[idx_k_above]
            w_above = (k - k_below) / (k_above - k_below)
        end
        w_below = 1.0 - w_above

        row_idx = (idx_k⁻ - 1)*nE + idx_e
        # Loop over the columns that this state (k⁻, e) can transition to
        for idx_e_prime in 1:nE
            col_idx_below = (idx_k_below - 1)*nE + idx_e_prime
            col_idx_above = (idx_k_above - 1)*nE + idx_e_prime

            Λ[row_idx, col_idx_below] += w_below * Pₑ[idx_e, idx_e_prime]
            Λ[row_idx, col_idx_above] += w_above * Pₑ[idx_e, idx_e_prime]
        end
    end

    return sparse(Λ)
end

#=
    Transform value → (y_min, y_max) using the logistic function
=#
function logistic(y_in, y_min=0.0, y_max=1.0)
    return y_min + (y_max - y_min) / (1.0 + exp(-y_in))
end
#=
    Inverse of the logistic function
=#
function logistic_inv(y_out, y_min=0.0, y_max=1.0)
    return -log((y_max - y_out) / (y_out - y_min))
end