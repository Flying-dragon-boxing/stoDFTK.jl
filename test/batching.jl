@testset "Batched Chebyshev agrees with scalar probes" begin
    @test_throws ArgumentError ChebyshevFOE(; batch_size=0)
    basis = toy_basis()
    ham = DFTK.Hamiltonian(basis; ρ=stoDFTK.default_density(basis))
    for sampling in (:phase, :rademacher, :complete)
        probes = StochasticOrbitals(basis; sampling, n_samples=17, seed=72)
        scalar = stochastic_density(basis, ham; probes, backend=ChebyshevFOE(; degree=60, batch_size=1))
        for batch_size in (4, 32)
            batched = stochastic_density(basis, ham; probes, backend=ChebyshevFOE(; degree=60, batch_size))
            @test density_distance(basis, scalar.ρ, batched.ρ) < 1e-8
            @test batched.εF ≈ scalar.εF atol=1e-8
            @test batched.energies.total ≈ scalar.energies.total atol=1e-8
            @test batched.n_electrons ≈ 1 atol=1e-8
            @test batched.timings.ham_applications == scalar.timings.ham_applications
            @test batched.timings.ham_batch_calls == sum(180cld(size(z, 2), batch_size) for z in probes.vectors)
        end
    end
    probes = StochasticOrbitals(basis; n_samples=17, seed=79)
    scalar = stochastic_scf(basis; probes, backend=ChebyshevFOE(; degree=60, batch_size=1), damping=0.6, tol=1e-7)
    batched = stochastic_scf(basis; probes, backend=ChebyshevFOE(; degree=60, batch_size=8), damping=0.6, tol=1e-7)
    @test scalar.converged && batched.converged
    @test density_distance(basis, scalar.ρ, batched.ρ) < 1e-7
    @test scalar.energies.total ≈ batched.energies.total atol=1e-7
end
