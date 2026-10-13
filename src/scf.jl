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

function physical_tau(basis, ρ, τ)
    if !needs_tau(basis)
        isnothing(τ) || throw(ArgumentError("τ is only accepted for models requiring kinetic energy density"))
        return nothing
    end
    τw = DFTK.von_weizsaecker_kinetic_energy_density(basis, ρ)
    result = if isnothing(τ)
        # Positive excess kinetic density also gives a useful uniform-gas guess.
        prefactor = (3 / 10) * (6π^2 / DFTK.filled_occupation(basis.model))^(2 / 3)
        τw .+ prefactor .* max.(ρ, 0.0) .^ (5 / 3)
    else
        size(τ) == size(ρ) || throw(DimensionMismatch("τ must have the same shape as ρ"))
        all(isfinite, τ) || throw(ArgumentError("τ must be finite"))
        minimum(τ) >= -1e-10 || throw(ArgumentError("initial τ must be nonnegative"))
        DFTK.to_device(basis.architecture, Float64.(τ))
    end
    # Project SCF inputs only. Never change the orbital-derived output estimator.
    DFTK.symmetrize_ρ(basis, max.(result, τw, 0.0); do_lowpass=false)
end

function physical_density(basis, ρ)
    size(ρ) == (basis.fft_size..., basis.model.n_spin_components) ||
        throw(DimensionMismatch("density must have shape (fft_size..., n_spin_components)"))
    all(isfinite, ρ) || throw(ArgumentError("density must be finite"))
    minimum(ρ) >= -1e-10 || throw(ArgumentError("initial density must be nonnegative"))
    result = DFTK.to_device(basis.architecture, max.(Float64.(ρ), 0.0))
    charge = sum(result) * basis.dvol
    charge > 0 || throw(ArgumentError("density must have positive total charge"))
    result .*= basis.model.n_electrons / charge
    DFTK.symmetrize_ρ(basis, result; do_lowpass=false)
end

"""
    stochastic_scf(basis; backend=ChebyshevFOE(), ρ=default_density(basis), τ=nothing,
                   n_samples=32, seed=0, sampling=:phase, probes=nothing,
                   maxiter=80, tol=1e-5, tol_tau=tol, damping=0.3,
                   mixing=DFTK.SimpleMixing(), callback=identity, kwargs...)

Self-consistent stochastic finite-temperature DFT. Reuse a fixed probe set across
iterations, update `H[ρin]`, solve the chemical potential, filter the orbitals,
and mix `ρout - ρin`. Convergence uses the *unmixed* residual
`norm(ρout - ρin) * sqrt(basis.dvol)` and the electron-count error. Meta-GGAs
also update kinetic density `τ` from the same orbitals and require its unmixed
residual `norm(τout-τin)*sqrt(basis.dvol)` to meet `tol_tau`.

`mixing` supports DFTK SimpleMixing, KerkerMixing and DielectricMixing. `kwargs`
are passed to `stochastic_density` (e.g. `tol_n_elec` and `spectral_bounds`).
The callback receives `stage=:iterate` snapshots and a final `stage=:finalize`.

Return `ρ=ρout`, `τ=τout` (or `nothing`), `ham=H[ρout,τout]`, `ψ`, `occupation`,
`εF`, `energies`, `converged`, `n_iter`, and `history`. `ρin`, `τin` and
`ham_input=H[ρin,τin]` record the last filter input. Mixed inputs are projected
to `ρ≥0`, fixed charge and `τ≥τW(ρ)`; orbital-derived outputs are unchanged. A small
residual means convergence of the sampled map, not convergence in sample count.
"""
function stochastic_scf(basis; backend::AbstractFOE=ChebyshevFOE(),
                         ρ=default_density(basis), τ=nothing, n_samples::Int=32,
                         seed::Int=0, sampling::Symbol=:phase, probes=nothing,
                         maxiter::Int=80, tol::Real=1e-5, tol_tau::Real=tol, damping::Real=0.3,
                         mixing=DFTK.SimpleMixing(), callback=identity,
                         tol_n_elec::Real=1e-8, probe_comm=MPI.COMM_SELF, kwargs...)
    validate_basis(basis)
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be positive"))
    isfinite(tol_tau) && tol_tau > 0 || throw(ArgumentError("tol_tau must be positive"))
    isfinite(damping) && 0 < damping <= 1 || throw(ArgumentError("damping must lie in (0,1]"))
    mixing isa Union{DFTK.SimpleMixing,DFTK.KerkerMixing,DFTK.DielectricMixing} ||
        throw(ArgumentError("mixing must be SimpleMixing, KerkerMixing or DielectricMixing"))
    haskey(kwargs, :μ) && throw(ArgumentError("SCF solves μ at fixed electron count; do not supply μ"))
    probes = isnothing(probes) ? StochasticOrbitals(basis; n_samples, seed, sampling, probe_comm) : probes
    validate_probes(basis, probes)
    ρin = physical_density(basis, ρ)
    τin = physical_tau(basis, ρin, τ)
    history = NamedTuple[]
    converged = false
    last, ham = nothing, nothing
    start = time_ns()
    for n_iter in 1:maxiter
        ham = DFTK.Hamiltonian(basis; ρ=ρin, τ=τin)
        last = stochastic_density(basis, ham; backend, probes, tol_n_elec, kwargs...)
        Δρ = last.ρ - ρin
        residual_norm = norm(Δρ) * sqrt(basis.dvol)
        Δτ = isnothing(τin) ? nothing : last.τ - τin
        residual_tau = isnothing(Δτ) ? 0.0 : norm(Δτ) * sqrt(basis.dvol)
        push!(history, (; n_iter, residual_norm, residual_tau, εF=last.εF,
                        free_energy=last.energies.total, internal_energy=last.internal_energy,
                        n_electrons=last.n_electrons, electron_count_error=last.electron_count_error,
                        timings=last.timings))
        converged = residual_norm <= tol && residual_tau <= tol_tau &&
                    abs(last.electron_count_error) <= max(tol_n_elec, 1e-7)
        info = (; basis, ham, ρin=copy(ρin), ρout=copy(last.ρ),
                  τin=isnothing(τin) ? nothing : copy(τin),
                  τout=isnothing(last.τ) ? nothing : copy(last.τ), Δρ, Δτ, n_iter,
                  residual_norm, residual_tau, εF=last.εF, energies=last.energies, converged,
                  history=copy(history), stage=:iterate, algorithm="stochastic-SCF")
        callback(info)
        (converged || n_iter == maxiter) && break
        δρ = DFTK.mix_density(mixing, basis, Δρ; info...)
        # Only the mixed input is projected to a physical density. Never rescale ρout.
        ρnext = ρin .+ damping .* δρ
        all(isfinite, ρnext) || error("density mixing produced a non-finite density")
        ρin = physical_density(basis, max.(ρnext, 0.0))
        if !isnothing(τin)
            τnext = τin .+ damping .* Δτ
            all(isfinite, τnext) || error("τ mixing produced a non-finite kinetic energy density")
            τin = physical_tau(basis, ρin, max.(τnext, 0.0))
        end
    end
    ham_out = DFTK.Hamiltonian(basis; ρ=last.ρ, τ=last.τ)
    result = merge(last, (; basis, ham=ham_out, ham_input=ham, ρin, τin, backend, converged,
                          n_iter=length(history), history,
                          residual_norm=history[end].residual_norm,
                          residual_tau=history[end].residual_tau,
                          history_Δρ=[h.residual_norm for h in history],
                          history_Δτ=[h.residual_tau for h in history],
                          history_εF=[h.εF for h in history],
                          runtime_seconds=(time_ns() - start) * 1e-9,
                          algorithm="stochastic-SCF"))
    callback(merge(result, (; stage=:finalize)))
    result
end

const sdft_scf = stochastic_scf
