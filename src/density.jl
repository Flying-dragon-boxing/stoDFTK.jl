function kpoint_bounds(supplied, ik, n_kpoints)
    isnothing(supplied) && return nothing
    if supplied isa Tuple && length(supplied) == 2 && all(x -> x isa Real, supplied)
        return checked_bounds(supplied)
    end
    supplied isa AbstractVector && length(supplied) == n_kpoints ||
        throw(ArgumentError("spectral_bounds must be one tuple or a vector of tuples, one per k-point"))
    checked_bounds(supplied[ik])
end

function estimate_count(basis, caches, probes, μ)
    temperature = basis.model.temperature
    occmax = DFTK.filled_occupation(basis.model)
    local_count = sum(basis.kweights[ik] * probes.weights[ik] *
        sum(count_samples(cache, μ, temperature, occmax)) for (ik, cache) in enumerate(caches))
    collective(local_count, +, trace_comm(basis, probes))
end

function solve_chemical_potential(basis, caches, probes; tol_n_elec, maxiter=100)
    target = basis.model.n_electrons
    occmax = DFTK.filled_occupation(basis.model)
    local_capacity = occmax * sum(basis.kweights[ik] * probes.weights[ik] * sum(abs2, χ)
                            for (ik, χ) in enumerate(probes.vectors))
    comm = trace_comm(basis, probes)
    capacity = collective(local_capacity, +, comm)
    0 < target < capacity || throw(ArgumentError(
        "electron target $target must lie strictly between 0 and basis capacity $capacity; increase Ecut"))
    temperature = basis.model.temperature
    lo = collective(minimum(cache -> cache.bounds[1], caches), min, comm) - 8temperature
    hi = collective(maximum(cache -> cache.bounds[2], caches), max, comm) + 8temperature
    count(μ) = estimate_count(basis, caches, probes, μ)
    nlo, nhi = count(lo), count(hi)
    # Bound expansion is finite even for invalid probes or an inadequate basis.
    for _ in 1:40
        nlo <= target <= nhi && break
        width = max(hi - lo, temperature, 1.0)
        if nlo > target
            lo -= width
            nlo = count(lo)
        end
        if nhi < target
            hi += width
            nhi = count(hi)
        end
    end
    isfinite(nlo) && isfinite(nhi) && nlo <= target <= nhi ||
        error("could not bracket chemical potential; check spectral bounds and expansion order")
    for n_iter in 1:maxiter
        μ = (lo + hi) / 2
        n = count(μ)
        isfinite(n) || error("non-finite electron count during chemical-potential solve")
        abs(n - target) <= tol_n_elec && return (; μ, n_electrons=n, n_iter)
        if n < target
            lo = μ
        else
            hi = μ
        end
    end
    error("chemical-potential solve did not meet electron-count tolerance $tol_n_elec")
end

function stochastic_energy(basis, ψ, occupation, ρ, entropy_energy; probe_comm=MPI.COMM_SELF)
    names, values = String[], Float64[]
    for (term_type, term) in zip(basis.model.term_types, basis.terms)
        name = string(nameof(typeof(term_type)))
        value = name == "Entropy" ? entropy_energy : DFTK.energy(term, basis, ψ, occupation; ρ)
        if name in ("Kinetic", "AtomicNonlocal")
            value = collective(value, +, probe_comm)
        end
        push!(names, name)
        push!(values, value)
    end
    if !("Entropy" in names)
        push!(names, "Entropy")
        push!(values, entropy_energy)
    end
    all(isfinite, values) || error("non-finite stochastic energy")
    DFTK.Energies(names, values)
end

"""
    stochastic_density(basis, ham; backend=ChebyshevFOE(), probes=nothing,
                       n_samples=32, seed=0, sampling=:phase,
                       μ=nothing, spectral_bounds=nothing, tol_n_elec=1e-8,
                       probe_comm=MPI.COMM_SELF)

Construct `ρ = Σ_k w_k ⟨|sqrt(f(H_k))χ_k|²⟩` using DFTK's FFT normalization,
spin channels, k-point weights and density symmetrization. Unless `μ` is given,
solve the electron count with the *same probes and same squared filter*.
No rescaling of the output density is performed.

Returns `ρ`, `εF`, `ψ` (filtered stochastic orbitals), `occupation` (estimator
weights, not KS occupations), `n_electrons`, `energies`, `entropy_energy`,
`bounds`, `probes`, and `timings`. `energies.total` is the Mermin free energy;
entropy is a separate trace approximation. `ψ` are not KS eigenstates.
"""
function stochastic_density(basis, ham; backend::AbstractFOE=ChebyshevFOE(),
                             probes=nothing, n_samples::Int=32, seed::Int=0,
                             sampling::Symbol=:phase, μ=nothing,
                             spectral_bounds=nothing, tol_n_elec::Real=1e-8,
                             probe_comm=MPI.COMM_SELF)
    validate_basis(basis)
    isfinite(tol_n_elec) && tol_n_elec > 0 || throw(ArgumentError("tol_n_elec must be positive"))
    ham.basis === basis || throw(ArgumentError("Hamiltonian belongs to a different basis"))
    probes = isnothing(probes) ? StochasticOrbitals(basis; n_samples, seed, sampling, probe_comm) : probes
    validate_probes(basis, probes)
    start = time_ns()
    caches = [prepare_cache(backend, ham[ik], probes.vectors[ik],
                            kpoint_bounds(spectral_bounds, ik, length(basis.kpoints)))
              for ik in eachindex(basis.kpoints)]
    preparation_seconds = (time_ns() - start) * 1e-9
    tμ = time_ns()
    chemical = if isnothing(μ)
        solve_chemical_potential(basis, caches, probes; tol_n_elec)
    else
        isfinite(μ) || throw(ArgumentError("μ must be finite"))
        (; μ=Float64(μ), n_electrons=estimate_count(basis, caches, probes, Float64(μ)), n_iter=0)
    end
    chemical_potential_seconds = (time_ns() - tμ) * 1e-9
    εF = chemical.μ
    temperature, occmax = basis.model.temperature, DFTK.filled_occupation(basis.model)
    tρ = time_ns()
    ψ = [filtered_orbitals(cache, εF, temperature, occmax) for cache in caches]
    occupation = [DFTK.to_device(basis.architecture, fill(probes.weights[ik], size(ψk, 2)))
                  for (ik, ψk) in enumerate(ψ)]
    ρ = DFTK.compute_density(basis, ψ, occupation)
    reduce_probe_density!(ρ, probes.probe_comm)
    density_seconds = (time_ns() - tρ) * 1e-9
    n_electrons = sum(ρ) * basis.dvol
    abs(n_electrons - chemical.n_electrons) <= max(1e-7, 1e-10 * n_electrons) ||
        error("density integral disagrees with the cached electron count; increase expansion accuracy")
    tE = time_ns()
    local_entropy = sum(basis.kweights[ik] * probes.weights[ik] *
                         sum(entropy_samples(cache, εF, temperature, occmax))
                         for (ik, cache) in enumerate(caches))
    entropy_energy = collective(local_entropy, +, trace_comm(basis, probes))
    energies = stochastic_energy(basis, ψ, occupation, ρ, entropy_energy;
                                 probe_comm=probes.probe_comm)
    energy_seconds = (time_ns() - tE) * 1e-9
    timings = (; preparation_seconds, chemical_potential_seconds, density_seconds, energy_seconds,
                total_seconds=(time_ns() - start) * 1e-9,
                ham_applications=collective(sum(application_count, caches), +, trace_comm(basis, probes)),
                ham_batch_calls=collective(sum(batch_application_count, caches), +, trace_comm(basis, probes)))
    (; ρ, εF, ψ, occupation, n_electrons, energies, entropy_energy,
       internal_energy=energies.total - entropy_energy,
       electron_count_error=n_electrons - basis.model.n_electrons,
       chemical_potential_iterations=chemical.n_iter,
       bounds=[cache.bounds for cache in caches], probes, timings)
end

const sdft_density = stochastic_density
