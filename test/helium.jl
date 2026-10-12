@testset "Small all-electron He PBE integration" begin
    basis = helium_basis()
    @test maximum(length(DFTK.G_vectors(basis, k)) for k in basis.kpoints) < 40
    backend = ChebyshevFOE(; degree=100)
    ham = DFTK.Hamiltonian(basis; ρ=stoDFTK.default_density(basis))
    ref = dense_reference(basis, ham)
    bounds = spectral_bounds(ham[1])
    @test bounds[1] < minimum(ref.eigenvalues[1]) < maximum(ref.eigenvalues[1]) < bounds[2]
    fixed = stochastic_density(basis, ham; backend, sampling=:complete)
    @test density_distance(basis, fixed.ρ, ref.ρ) < 1e-8
    @test fixed.energies.total ≈ ref.energies.total atol=1e-7
    result = stochastic_scf(basis; backend, sampling=:complete,
                             damping=0.5, maxiter=80, tol=1e-7)
    @test result.converged
    exact = dense_reference(basis, result.ham)
    @test density_distance(basis, result.ρ, exact.ρ) < 1e-7
    @test result.energies.total ≈ exact.energies.total atol=1e-6
    @test result.n_electrons ≈ 2 atol=1e-8
    println("He validation: dim=$(size(ham[1], 1)), iterations=$(result.n_iter), " *
            "residual=$(result.residual_norm), ΔF=$(abs(result.energies.total - exact.energies.total)) Ha")
end
