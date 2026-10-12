# stoDFTK.jl

Stochastic wavefunction density-functional theory built on [DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl).

stoDFTK approximates the square root of the finite-temperature Fermi operator,
constructs the density from filtered stochastic orbitals, and performs a full
self-consistent density iteration. The public `main` branch implements the
Chebyshev backend. Experimental Lanczos development is maintained separately.

[中文说明](README.zh-CN.md) · [Algorithm](docs/method.md) · [Validation](docs/validation.md)

## Installation

Requires Julia 1.10 or newer and DFTK 0.7.25–0.7.x. This package is not yet in
the Julia General registry; install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Flying-dragon-boxing/stoDFTK.jl", rev="main")
using DFTK, stoDFTK
```

For local development:

```sh
git clone https://github.com/Flying-dragon-boxing/stoDFTK.jl
cd stoDFTK.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=. examples/nonlinear_scf.jl
julia --project=. examples/helium_scf.jl
```

Tests and examples use analytic potentials and an all-electron Coulomb atom.
No pseudopotential data files are bundled or downloaded by these examples.

## Self-consistent stochastic DFT

Construct a DFTK `PlaneWaveBasis` with positive temperature in Hartree and
Fermi–Dirac smearing, then call:

```julia
result = stochastic_scf(basis;
    backend=ChebyshevFOE(degree=100),
    n_samples=64, seed=2026,
    maxiter=80, tol=1e-6, damping=0.3,
    mixing=DFTK.KerkerMixing(),
)

result.converged
result.ρ                     # Real-space density, including spin channels
result.εF                    # Chemical potential in Hartree
result.energies.total         # Finite-temperature free energy in Hartree
result.internal_energy       # Energy excluding the -TS contribution
result.history               # Residuals, electron counts, energies and timings
```

Each iteration constructs `H[ρin]`, solves the electron-count constraint,
filters the orbitals, builds `ρout`, and mixes the density. A fixed set of
probes is reused throughout the SCF cycle. Chemical-potential solving and
density construction use the same probes and squared Chebyshev filter; the
output density is not rescaled to impose the electron count.

Convergence is based on the unmixed residual
`norm(ρout - ρin) * sqrt(basis.dvol)`. The result reports `converged=false`
when `maxiter` is reached. Callbacks receive `stage=:iterate` and
`stage=:finalize` events.

## Fixed-Hamiltonian calculations

```julia
# ρin is a nonnegative trial density in DFTK real-space layout.
ham = DFTK.Hamiltonian(basis; ρ=ρin)
probes = StochasticOrbitals(basis; n_samples=64, seed=2026, sampling=:phase)
out = stochastic_density(basis, ham;
    backend=ChebyshevFOE(degree=100), probes,
)
```

Sampling supports complex random phases (`:phase`), complex Rademacher
vectors (`:rademacher`), and the complete identity basis (`:complete`). Use
the complete basis only for small validation systems. Supply `μ` to hold the
chemical potential fixed, or `spectral_bounds=(emin, emax)` to provide a
spectral enclosure explicitly. Automatic bounds use the kinetic and local
potential extrema and a projector-space bound for nonlocal operators.

## Scope and validation

The implementation supports CPU / Float64, a single MPI process, multiple
k-points, spinless / unpolarized / collinear spin models, LDA / GGA,
norm-conserving pseudopotential Hamiltonians supplied by the caller, and
Simple / Kerker / Dielectric density mixing.

Zero temperature, distributed MPI, GPU, deterministic-subspace mixing,
hybrid functionals, meta-GGA requiring kinetic energy density, forces and
molecular dynamics are not implemented.

`result.ψ` contains filtered stochastic orbitals and `result.occupation`
contains estimator weights. They can reconstruct the density through
`DFTK.compute_density`, but they are not Kohn–Sham eigenstates or occupation
numbers. `ham=H[ρout]` and `ham_input=H[ρin]` retain the last SCF output and
filter input, with their difference measured by the reported residual.

SCF residual convergence must be distinguished from expansion-order and
sample-count convergence. See the [validation record](docs/validation.md)
for small-system comparisons against full diagonalization. The all-electron
He example deliberately uses a low cutoff for an inexpensive smoke test.

## License

stoDFTK's source code and documentation are licensed under the [MIT License](LICENSE).
DFTK and other dependencies retain their own licenses; see
[third-party notices](THIRD_PARTY_NOTICES.md).
