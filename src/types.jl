abstract type AbstractFOE end

"""
    ChebyshevFOE(; degree=80, quadrature_order=max(256, 8(degree+1)))

Approximate `sqrt(f(H))` with a degree-`degree` Chebyshev polynomial.
Electron counts use the square of this same polynomial. Energies are in Hartree.
Increase `degree` to check expansion convergence, especially at low temperatures.
"""
struct ChebyshevFOE <: AbstractFOE
    degree::Int
    quadrature_order::Int
    function ChebyshevFOE(; degree::Int=80,
                           quadrature_order::Int=max(256, 8(degree + 1)))
        degree > 0 || throw(ArgumentError("degree must be positive"))
        quadrature_order > 2degree ||
            throw(ArgumentError("quadrature_order must exceed 2degree"))
        new(degree, quadrature_order)
    end
end

"""
    StochasticOrbitals(basis; n_samples=32, seed=0, sampling=:phase)

Fixed probes with `E[χχ†] = I`, generated without touching Julia's global RNG.
`sampling=:phase` uses uniform complex phases, `:rademacher` uses ±1 or ±i.
`:complete` uses the identity basis with unit weights, for small-system validation.
Probes belong to this exact `basis` object and are reused across SCF iterations.
"""
struct StochasticOrbitals{B}
    basis::B
    vectors::Vector{Matrix{ComplexF64}}
    weights::Vector{Float64}
    sampling::Symbol
    seed::Int
end

function StochasticOrbitals(basis; n_samples::Int=32, seed::Int=0,
                            sampling::Symbol=:phase)
    validate_basis(basis)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    sampling in (:phase, :rademacher, :complete) ||
        throw(ArgumentError("sampling must be :phase, :rademacher or :complete"))
    rng = MersenneTwister(seed)
    vectors = Matrix{ComplexF64}[]
    weights = Float64[]
    for kpt in basis.kpoints
        dim = length(DFTK.G_vectors(basis, kpt))
        dim > 0 || throw(ArgumentError("empty plane-wave basis; increase Ecut"))
        if sampling == :complete
            push!(vectors, Matrix{ComplexF64}(I, dim, dim))
            push!(weights, 1.0)
        else
            χ = Matrix{ComplexF64}(undef, dim, n_samples)
            for i in eachindex(χ)
                χ[i] = sampling == :phase ? cis(2π * rand(rng)) : rand(rng, (1, -1, im, -im))
            end
            push!(vectors, χ)
            push!(weights, 1.0 / n_samples)
        end
    end
    StochasticOrbitals(basis, vectors, weights, sampling, seed)
end

function validate_basis(basis)
    eltype(basis) == Float64 || throw(ArgumentError("only Float64 bases are supported"))
    basis.architecture isa DFTK.CPU || throw(ArgumentError("only CPU bases are supported"))
    DFTK.mpi_nprocs(basis.comm_kpts) == 1 ||
        throw(ArgumentError("distributed MPI bases are not supported yet"))
    isfinite(basis.model.temperature) && basis.model.temperature > 0 ||
        throw(ArgumentError("stochastic DFT requires a finite, positive temperature (in Ha)"))
    basis.model.smearing isa DFTK.Smearing.FermiDirac ||
        throw(ArgumentError("stochastic DFT requires FermiDirac smearing"))
    isnothing(basis.model.εF) ||
        throw(ArgumentError("use a fixed-electron-count DFTK model; pass μ to stochastic_density for fixed-H work"))
    basis.model.n_electrons > 0 || throw(ArgumentError("n_electrons must be positive"))
    # Orbital-dependent functionals need a separate SCF formulation.
    allowed = (:Kinetic, :AtomicLocal, :AtomicNonlocal, :Ewald, :PspCorrection,
               :Hartree, :Xc, :Entropy, :ExternalFromReal, :ExternalFromFourier,
               :ExternalFromValues, :LocalNonlinearity)
    for term in basis.model.term_types
        nameof(typeof(term)) in allowed ||
            throw(ArgumentError("unsupported model term $(nameof(typeof(term)))"))
    end
    for term in basis.terms
        if term isa DFTK.TermXc && DFTK.needs_τ(term)
            throw(ArgumentError("meta-GGA requiring kinetic energy density is not supported"))
        end
    end
    nothing
end

function validate_probes(basis, probes)
    probes.basis === basis || throw(ArgumentError("probes were generated for a different basis"))
    length(probes.vectors) == length(probes.weights) == length(basis.kpoints) ||
        throw(ArgumentError("one probe matrix and weight per k-point is required"))
    for (ik, kpt) in enumerate(basis.kpoints)
        χ = probes.vectors[ik]
        size(χ, 1) == length(DFTK.G_vectors(basis, kpt)) && size(χ, 2) > 0 ||
            throw(ArgumentError("invalid probe dimensions at k-point $ik"))
        all(isfinite, χ) && isfinite(probes.weights[ik]) && probes.weights[ik] > 0 ||
            throw(ArgumentError("probes and weights must be finite; weights must be positive"))
    end
    nothing
end

fermi_occupation(e, μ, temperature, occmax) =
    occmax * DFTK.Smearing.occupation(DFTK.Smearing.FermiDirac(), (e - μ) / temperature)

entropy_integrand(e, μ, temperature, occmax) =
    -temperature * occmax * DFTK.Smearing.entropy(DFTK.Smearing.FermiDirac(), (e - μ) / temperature)

function checked_bounds(bounds)
    length(bounds) == 2 || throw(ArgumentError("spectral bounds must be (emin, emax)"))
    lo, hi = Float64.(bounds)
    isfinite(lo) && isfinite(hi) && lo < hi ||
        throw(ArgumentError("spectral bounds must be finite and strictly increasing"))
    (lo, hi)
end
