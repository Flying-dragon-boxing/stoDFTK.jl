using DFTK
using stoDFTK

# Self-contained finite-temperature model; no pseudopotential downloads needed.
lattice = [6.0 0 0; 0 0 0; 0 0 0]
terms = [Kinetic(), ExternalFromReal(r -> 0.3cos(2π * r[1] / 6)),
         LocalNonlinearity(ρ -> 0.5ρ^2), Entropy()]
model = Model(lattice; terms, n_electrons=1, temperature=0.4,
              spin_polarization=:spinless, disable_electrostatics_check=true)
basis = PlaneWaveBasis(model; Ecut=4.0, kgrid=(3, 1, 1))

result = stochastic_scf(basis; backend=ChebyshevFOE(; degree=80), n_samples=64,
                         seed=2026, damping=0.6, maxiter=100, tol=1e-7,
                         callback=info -> begin
                             info.stage == :iterate || return
                             println("iter=$(info.n_iter) residual=$(info.residual_norm) F=$(info.energies.total) Ha")
                         end)
println("converged=$(result.converged), N=$(result.n_electrons), μ=$(result.εF) Ha")
