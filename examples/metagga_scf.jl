using DFTK, LinearAlgebra, stoDFTK

# Small all-electron He smoke test, without pseudopotential data.
complete = "--complete" in ARGS
lattice = 6.0 * Matrix{Float64}(I,3,3)
model = model_DFT(lattice, [ElementCoulomb(:He)], [[0.0,0.0,0.0]];
                  functionals=[:mgga_x_r2scan,:mgga_c_r2scan], temperature=0.75)
basis = PlaneWaveBasis(model; Ecut=1.5, kgrid=(1,1,1))
result = stochastic_scf(basis; backend=ChebyshevFOE(; degree=150),
                        sampling=complete ? :complete : :phase,
                        n_samples=32, seed=2026, damping=0.5,
                        tol=1e-6, tol_tau=1e-6, maxiter=100)
println("He r²SCAN: converged=$(result.converged), iterations=$(result.n_iter), " *
        "Δρ=$(result.residual_norm), Δτ=$(result.residual_tau), F=$(result.energies.total) Ha")
result.converged || error("joint ρ/τ SCF did not converge")
