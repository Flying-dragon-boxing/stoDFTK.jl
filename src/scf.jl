function default_density(basis)
    # A Fourier-truncated atomic guess can undershoot below zero on a coarse grid.
    # Project only the automatic initial guess; explicit user input is validated below.
    ρ = max.(DFTK.guess_density(basis), 0.0)
    # Atom-free model problems have no atomic charge from which to form a guess.
    if iszero(sum(ρ)) && !isnothing(basis.model.n_electrons)
        fill!(ρ, basis.model.n_electrons / (basis.model.unit_cell_volume * basis.model.n_spin_components))
    end
    ρ
end

function physical_density(basis, ρ)
    size(ρ) == (basis.fft_size..., basis.model.n_spin_components) ||
        throw(DimensionMismatch("density must have shape (fft_size..., n_spin_components)"))
    all(isfinite, ρ) || throw(ArgumentError("density must be finite"))
    minimum(ρ) >= -1e-10 || throw(ArgumentError("initial density must be nonnegative"))
    result = max.(Float64.(ρ), 0.0)
    charge = sum(result) * basis.dvol
    charge > 0 || throw(ArgumentError("density must have positive total charge"))
    result .*= basis.model.n_electrons / charge
    DFTK.symmetrize_ρ(basis, result; do_lowpass=false)
end

"""
    stochastic_scf(basis; backend=ChebyshevFOE(), ρ=default_density(basis),
                   n_samples=32, seed=0, sampling=:phase, probes=nothing,
                   maxiter=80, tol=1e-5, damping=0.3,
                   mixing=DFTK.SimpleMixing(), callback=identity, kwargs...)

Self-consistent stochastic finite-temperature DFT. Reuse a fixed probe set across
iterations, update `H[ρin]`, solve the chemical potential, filter the orbitals,
and mix `ρout - ρin`. Convergence uses the *unmixed* residual
`norm(ρout - ρin) * sqrt(basis.dvol)` and the electron-count error.

`mixing` supports DFTK SimpleMixing, KerkerMixing and DielectricMixing. `kwargs`
are passed to `stochastic_density` (e.g. `tol_n_elec` and `spectral_bounds`).
The callback receives `stage=:iterate` snapshots and a final `stage=:finalize`.

Return `ρ=ρout`, `ham=H[ρout]`, `ψ`, `occupation`, `εF`, `energies`,
`converged`, `n_iter`, and `history`. `ρin` and `ham_input=H[ρin]` record the
last filter input; the two densities differ by the reported residual. A small
residual means convergence of the sampled map, not convergence in sample count.
"""
function stochastic_scf(basis; backend::AbstractFOE=ChebyshevFOE(),
                         ρ=default_density(basis), n_samples::Int=32,
                         seed::Int=0, sampling::Symbol=:phase, probes=nothing,
                         maxiter::Int=80, tol::Real=1e-5, damping::Real=0.3,
                         mixing=DFTK.SimpleMixing(), callback=identity,
                         tol_n_elec::Real=1e-8, kwargs...)
    validate_basis(basis)
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be positive"))
    isfinite(damping) && 0 < damping <= 1 || throw(ArgumentError("damping must lie in (0,1]"))
    mixing isa Union{DFTK.SimpleMixing,DFTK.KerkerMixing,DFTK.DielectricMixing} ||
        throw(ArgumentError("mixing must be SimpleMixing, KerkerMixing or DielectricMixing"))
    haskey(kwargs, :μ) && throw(ArgumentError("SCF solves μ at fixed electron count; do not supply μ"))
    probes = isnothing(probes) ? StochasticOrbitals(basis; n_samples, seed, sampling) : probes
    validate_probes(basis, probes)
    ρin = physical_density(basis, ρ)
    history = NamedTuple[]
    converged = false
    last, ham = nothing, nothing
    start = time_ns()
    for n_iter in 1:maxiter
        ham = DFTK.Hamiltonian(basis; ρ=ρin)
        last = stochastic_density(basis, ham; backend, probes, tol_n_elec, kwargs...)
        Δρ = last.ρ - ρin
        residual_norm = norm(Δρ) * sqrt(basis.dvol)
        push!(history, (; n_iter, residual_norm, εF=last.εF,
                        free_energy=last.energies.total, internal_energy=last.internal_energy,
                        n_electrons=last.n_electrons, electron_count_error=last.electron_count_error,
                        timings=last.timings))
        converged = residual_norm <= tol && abs(last.electron_count_error) <= max(tol_n_elec, 1e-7)
        info = (; basis, ham, ρin=copy(ρin), ρout=copy(last.ρ), Δρ, n_iter,
                  residual_norm, εF=last.εF, energies=last.energies, converged,
                  history=copy(history), stage=:iterate, algorithm="stochastic-SCF")
        callback(info)
        (converged || n_iter == maxiter) && break
        δρ = DFTK.mix_density(mixing, basis, Δρ; info...)
        # Only the mixed input is projected to a physical density. Never rescale ρout.
        ρnext = ρin .+ damping .* δρ
        all(isfinite, ρnext) || error("density mixing produced a non-finite density")
        ρin = physical_density(basis, max.(ρnext, 0.0))
    end
    ham_out = DFTK.Hamiltonian(basis; ρ=last.ρ)
    result = merge(last, (; basis, ham=ham_out, ham_input=ham, ρin, backend, converged,
                          n_iter=length(history), history,
                          residual_norm=history[end].residual_norm,
                          history_Δρ=[h.residual_norm for h in history],
                          history_εF=[h.εF for h in history],
                          runtime_seconds=(time_ns() - start) * 1e-9,
                          algorithm="stochastic-SCF"))
    callback(merge(result, (; stage=:finalize)))
    result
end

const sdft_scf = stochastic_scf
