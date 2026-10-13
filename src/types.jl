abstract type AbstractFOE end

"""
    ChebyshevFOE(; degree=80, quadrature_order=max(256, 8(degree+1)), batch_size=32)

Approximate `sqrt(f(H))` with a degree-`degree` Chebyshev polynomial.
Electron counts use the square of this same polynomial. Energies are in Hartree.
Increase `degree` to check expansion convergence, especially at low temperatures.
`batch_size` controls how many probes share each Hamiltonian matrix application.
"""
struct ChebyshevFOE <: AbstractFOE
    degree::Int
    quadrature_order::Int
    batch_size::Int
    function ChebyshevFOE(; degree::Int=80,
                           quadrature_order::Int=max(256, 8(degree + 1)),
                           batch_size::Int=32)
        degree > 0 || throw(ArgumentError("degree must be positive"))
        quadrature_order > 2degree ||
            throw(ArgumentError("quadrature_order must exceed 2degree"))
        batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
        new(degree, quadrature_order, batch_size)
    end
end

"""
    StochasticOrbitals(basis; n_samples=32, seed=0, sampling=:phase, probe_comm=MPI.COMM_SELF)

Fixed probes with `E[χχ†] = I`, generated without touching Julia's global RNG.
`sampling=:phase` uses uniform complex phases, `:rademacher` uses ±1 or ±i.
`:complete` uses the identity basis with unit weights, for small-system validation.
Probes belong to this exact `basis` object and are reused across SCF iterations.
Seeds depend on the physical k-point and global probe index, so changing MPI
partitioning preserves the samples. Probe MPI requires a replicated basis
with `comm_kpts=MPI.COMM_SELF`; `n_samples` is the global sample count.
"""
struct StochasticOrbitals{B,M<:AbstractMatrix{ComplexF64}}
    basis::B
    vectors::Vector{M}
    weights::Vector{Float64}
    sampling::Symbol
    seed::Int
    probe_comm::MPI.Comm
    global_counts::Vector{Int}
    indices::Vector{Vector{Int}}
end

function StochasticOrbitals(basis; n_samples::Int=32, seed::Int=0,
                            sampling::Symbol=:phase, probe_comm=MPI.COMM_SELF)
    validate_basis(basis)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    sampling in (:phase, :rademacher, :complete) ||
        throw(ArgumentError("sampling must be :phase, :rademacher or :complete"))
    validate_communicators(basis, probe_comm)
    vectors = Matrix{ComplexF64}[]
    weights = Float64[]
    global_counts = Int[]
    indices = Vector{Int}[]
    for kpt in basis.kpoints
        dim = length(DFTK.G_vectors(basis, kpt))
        dim > 0 || throw(ArgumentError("empty plane-wave basis; increase Ecut"))
        total = sampling == :complete ? dim : n_samples
        selected = local_probe_indices(total, probe_comm)
        χ = zeros(ComplexF64, dim, length(selected))
        for (column, index) in enumerate(selected)
            if sampling == :complete
                χ[index, column] = 1
            else
                rng = Xoshiro(probe_seed(seed, kpt, index))
                for i in axes(χ, 1)
                    χ[i, column] = sampling == :phase ? cis(2π * rand(rng)) : rand(rng, (1, -1, im, -im))
                end
            end
        end
        push!(vectors, χ)
        push!(weights, sampling == :complete ? 1.0 : 1.0 / n_samples)
        push!(global_counts, total)
        push!(indices, selected)
    end
    device_vectors = [DFTK.to_device(basis.architecture, χ) for χ in vectors]
    StochasticOrbitals(basis, device_vectors, weights, sampling, seed,
                       probe_comm, global_counts, indices)
end

function validate_basis(basis)
    eltype(basis) == Float64 || throw(ArgumentError("only Float64 bases are supported"))
    basis.architecture isa Union{DFTK.CPU,DFTK.GPU} ||
        throw(ArgumentError("only DFTK CPU and GPU bases are supported"))
    isfinite(basis.model.temperature) && basis.model.temperature > 0 ||
        throw(ArgumentError("stochastic DFT requires a finite, positive temperature (in Ha)"))
    basis.model.smearing isa DFTK.Smearing.FermiDirac ||
        throw(ArgumentError("stochastic DFT requires FermiDirac smearing"))
    isnothing(basis.model.εF) ||
        throw(ArgumentError("use a fixed-electron-count DFTK model; pass μ to stochastic_density for fixed-H work"))
    basis.model.n_electrons > 0 || throw(ArgumentError("n_electrons must be positive"))
    allowed = (:Kinetic, :AtomicLocal, :AtomicNonlocal, :Ewald, :PspCorrection,
               :Hartree, :Xc, :Entropy, :ExternalFromReal, :ExternalFromFourier,
               :ExternalFromValues, :LocalNonlinearity)
    for term in basis.model.term_types
        nameof(typeof(term)) in allowed ||
            throw(ArgumentError("unsupported model term $(nameof(typeof(term)))"))
    end
    nothing
end

needs_tau(basis) = any(DFTK.needs_τ, basis.terms)

function validate_probes(basis, probes)
    probes.basis === basis || throw(ArgumentError("probes were generated for a different basis"))
    validate_communicators(basis, probes.probe_comm)
    length(probes.vectors) == length(probes.weights) == length(basis.kpoints) ||
        throw(ArgumentError("one probe matrix and weight per k-point is required"))
    for (ik, kpt) in enumerate(basis.kpoints)
        χ = probes.vectors[ik]
        size(χ, 1) == length(DFTK.G_vectors(basis, kpt)) && probes.global_counts[ik] > 0 &&
            size(χ, 2) == length(probes.indices[ik]) ||
            throw(ArgumentError("invalid probe dimensions at k-point $ik"))
        all(isfinite, χ) && isfinite(probes.weights[ik]) && probes.weights[ik] > 0 ||
            throw(ArgumentError("probes and weights must be finite; weights must be positive"))
        collective(size(χ, 2), +, probes.probe_comm) == probes.global_counts[ik] ||
            throw(ArgumentError("MPI probe partitions do not cover the global sample count"))
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
