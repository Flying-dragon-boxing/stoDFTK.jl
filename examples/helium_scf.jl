using DFTK
using LinearAlgebra
using stoDFTK

# Analytic Coulomb atom: no pseudopotential files or downloads are required.
# Ecut=1.5 Ha is deliberately small for a smoke test, not a converged physical cutoff.
lattice = 6.0 * Matrix{Float64}(I, 3, 3)
model = model_DFT(lattice, [ElementCoulomb(:He)], [[0.0, 0.0, 0.0]];
                  functionals=PBE(), temperature=0.75)
basis = PlaneWaveBasis(model; Ecut=1.5, kgrid=(1, 1, 1))
println("He: PW dimensions=$([length(DFTK.G_vectors(basis, k)) for k in basis.kpoints])")
sampling = "--complete" in ARGS ? :complete : :phase
result = stochastic_scf(basis; backend=ChebyshevFOE(; degree=100), sampling,
                         n_samples=64, seed=2026, damping=0.5, maxiter=80, tol=1e-6,
                         callback=info -> begin
                             info.stage == :iterate || return
                             println("iter=$(info.n_iter) residual=$(info.residual_norm) F=$(info.energies.total) Ha")
                         end)
println("converged=$(result.converged), N=$(result.n_electrons), μ=$(result.εF) Ha")
