"""
    spectral_bounds(H)

Conservative spectral enclosure. For DFTK's standard DFT Hamiltonian use extrema
of the kinetic and local potentials, plus a norm bound on `P D P†`.
Dense matrices use Gershgorin bounds. No full Hamiltonian diagonalization is used.
For other matrix-free operators, supply an enclosure explicitly.
"""
function nonlocal_radius(op)
    # Nonzero eigenvalues of PDP† are those of sqrt(G) D sqrt(G), G=P†P.
    # Only diagonalize in projector space, never in the plane-wave space.
    P, D = op.P, op.D
    iszero(size(P, 2)) && return 0.0
    gram = eigen(Hermitian(P' * P))
    root = gram.vectors * Diagonal(sqrt.(max.(gram.values, 0.0))) * gram.vectors'
    maximum(abs, eigvals(Hermitian(root * D * root)); init=0.0)
end

function spectral_bounds(H::DFTK.DftHamiltonianBlock)
    isnothing(H.divAgrad_op) || throw(ArgumentError("divAgrad operators are not supported"))
    kinetic = H.fourier_op.multiplier
    potential = H.local_op.potential
    radius = if isnothing(H.nonlocal_op)
        0.0
    else
        nonlocal_radius(H.nonlocal_op)
    end
    widen_bounds(minimum(kinetic) + minimum(potential) - radius,
                 maximum(kinetic) + maximum(potential) + radius)
end

function spectral_bounds(H::DFTK.GenericHamiltonianBlock)
    lo, hi = 0.0, 0.0
    for op in H.optimized_operators
        if op isa DFTK.FourierMultiplication
            lo += minimum(op.multiplier)
            hi += maximum(op.multiplier)
        elseif op isa DFTK.RealSpaceMultiplication
            lo += minimum(op.potential)
            hi += maximum(op.potential)
        elseif op isa DFTK.NonlocalOperator
            radius = nonlocal_radius(op)
            lo -= radius
            hi += radius
        elseif !(op isa DFTK.NoopOperator)
            throw(ArgumentError("no automatic enclosure for $(typeof(op)); supply spectral_bounds"))
        end
    end
    widen_bounds(lo, hi)
end

function spectral_bounds(H::AbstractMatrix)
    size(H, 1) == size(H, 2) > 0 || throw(ArgumentError("H must be a nonempty square matrix"))
    ishermitian(H) || throw(ArgumentError("H must be Hermitian"))
    d = real.(diag(H))
    r = vec(sum(abs, H; dims=2)) - abs.(diag(H))
    widen_bounds(minimum(d - r), maximum(d + r))
end

spectral_bounds(H) = throw(ArgumentError("no automatic spectral enclosure for $(typeof(H)); supply spectral_bounds"))

function widen_bounds(lo, hi)
    padding = 1e-10 * max(1.0, abs(lo), abs(hi))
    checked_bounds((lo - padding, hi + padding))
end

struct ChebyshevCache{H}
    ham::H
    probes::Matrix{ComplexF64}
    bounds::Tuple{Float64,Float64}
    nodes::Vector{Float64}
    transform::Matrix{Float64}
    moments::Matrix{Float64}
    degree::Int
    ham_applications::Int
end

function chebyshev_transform(order, quadrature_order, bounds)
    lo, hi = bounds
    θ = ((1:quadrature_order) .- 0.5) .* (π / quadrature_order)
    nodes = (lo + hi) / 2 .+ ((hi - lo) / 2) .* cos.(θ)
    transform = [((k == 0 ? 1.0 : 2.0) / quadrature_order) * cos(k * θj)
                 for k in 0:order, θj in θ]
    (; nodes, transform)
end

function scaled_hamiltonian!(y, H, x, bounds)
    lo, hi = bounds
    mul!(y, H, x)
    @. y = (y - ((lo + hi) / 2) * x) / ((hi - lo) / 2)
    y
end

function prepare_cache(backend::ChebyshevFOE, H, probes, supplied_bounds)
    H isa AbstractMatrix && !ishermitian(H) && throw(ArgumentError("H must be Hermitian"))
    bounds = isnothing(supplied_bounds) ? spectral_bounds(H) : checked_bounds(supplied_bounds)
    dim, n_samples = size(probes)
    order = 2backend.degree
    moments = zeros(order + 1, n_samples)
    t0, t1, t2 = (zeros(ComplexF64, dim) for _ in 1:3)
    for j in 1:n_samples
        z = @view probes[:, j]
        copyto!(t0, z)
        moments[1, j] = real(dot(z, t0))
        scaled_hamiltonian!(t1, H, t0, bounds)
        moments[2, j] = real(dot(z, t1))
        for k in 2:order
            scaled_hamiltonian!(t2, H, t1, bounds)
            @. t2 = 2t2 - t0
            moments[k + 1, j] = real(dot(z, t2))
            # On [-1,1], ||T_k(H)z|| ≤ ||z||. Check wrong supplied bounds early.
            norm(t2) <= (1 + 1e-7) * norm(z) ||
                throw(ArgumentError("Chebyshev recurrence escaped the spectral enclosure; widen spectral_bounds"))
            t0, t1, t2 = t1, t2, t0
        end
    end
    all(isfinite, moments) || throw(ArgumentError("non-finite Chebyshev moments"))
    (; nodes, transform) = chebyshev_transform(order, backend.quadrature_order, bounds)
    ChebyshevCache(H, probes, bounds, nodes, transform, moments,
                   backend.degree, order * n_samples)
end

root_coefficients(cache::ChebyshevCache, μ, temperature, occmax) =
    (@view cache.transform[1:cache.degree+1, :]) *
    sqrt.(fermi_occupation.(cache.nodes, μ, temperature, occmax))

function square_chebyshev_coefficients(c)
    # T_i T_j = (T_{i+j} + T_{|i-j|}) / 2, also valid for i or j = 0.
    result = zeros(Float64, 2length(c) - 1)
    for i in eachindex(c), j in eachindex(c)
        product = c[i] * c[j] / 2
        result[i + j - 1] += product
        result[abs(i - j) + 1] += product
    end
    result
end

function count_samples(cache::ChebyshevCache, μ, temperature, occmax)
    c = root_coefficients(cache, μ, temperature, occmax)
    transpose(cache.moments) * square_chebyshev_coefficients(c)
end

function entropy_samples(cache::ChebyshevCache, μ, temperature, occmax)
    c = cache.transform * entropy_integrand.(cache.nodes, μ, temperature, occmax)
    transpose(cache.moments) * c
end

function chebyshev_action(H, z, c, bounds)
    t0 = ComplexF64.(z)
    t1, t2 = similar(t0), similar(t0)
    y = c[1] .* t0
    length(c) == 1 && return y
    scaled_hamiltonian!(t1, H, t0, bounds)
    y .+= c[2] .* t1
    for k in 2:length(c)-1
        scaled_hamiltonian!(t2, H, t1, bounds)
        @. t2 = 2t2 - t0
        y .+= c[k + 1] .* t2
        t0, t1, t2 = t1, t2, t0
    end
    all(isfinite, y) || throw(ArgumentError("non-finite filtered orbitals"))
    y
end

function filtered_orbitals(cache::ChebyshevCache, μ, temperature, occmax)
    c = root_coefficients(cache, μ, temperature, occmax)
    ψ = similar(cache.probes)
    for j in axes(ψ, 2)
        ψ[:, j] = chebyshev_action(cache.ham, @view(cache.probes[:, j]), c, cache.bounds)
    end
    ψ
end

application_count(cache::ChebyshevCache) =
    cache.ham_applications + cache.degree * size(cache.probes, 2)

"""
    fermi_operator_action(H, χ, μ, temperature;
                          backend=ChebyshevFOE(), occmax=2, spectral_bounds=nothing)

Return `sqrt(f(H; μ, temperature))χ` without diagonalizing `H`.
`χ` must use the same normalization as plane-wave coefficients in DFTK.
"""
function fermi_operator_action(H, χ, μ, temperature; backend::AbstractFOE=ChebyshevFOE(),
                               occmax=2.0, spectral_bounds=nothing)
    isfinite(temperature) && temperature > 0 || throw(ArgumentError("temperature must be positive"))
    isfinite(μ) && isfinite(occmax) && occmax > 0 || throw(ArgumentError("invalid μ or occmax"))
    size(H) == (length(χ), length(χ)) || throw(DimensionMismatch("H and χ have incompatible dimensions"))
    all(isfinite, χ) || throw(ArgumentError("χ must be finite"))
    cache = prepare_cache(backend, H, reshape(ComplexF64.(χ), :, 1), spectral_bounds)
    vec(filtered_orbitals(cache, μ, temperature, occmax))
end
