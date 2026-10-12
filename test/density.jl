@testset "Probe covariance and reproducibility" begin
    basis = toy_basis()
    a = StochasticOrbitals(basis; n_samples=256, seed=123)
    b = StochasticOrbitals(basis; n_samples=256, seed=123)
    @test a.vectors == b.vectors
    @test all(z -> all(x -> abs(x) ≈ 1, z), a.vectors)
    Random.seed!(11)
    expected = rand()
    Random.seed!(11)
    StochasticOrbitals(basis; seed=44)
    @test rand() == expected
    for sampling in (:phase, :rademacher)
        probes = StochasticOrbitals(basis; sampling, n_samples=10000, seed=9)
        χ = probes.vectors[1]
        @test opnorm(χ * χ' / size(χ, 2) - I) < 0.1
    end
    @test_throws ArgumentError StochasticOrbitals(basis; n_samples=0)
    @test_throws ArgumentError StochasticOrbitals(basis; sampling=:unknown)
end

@testset "DFTK fixed-H density, chemical potential and free energy" begin
    for spin in (:spinless, :none, :collinear)
        basis = toy_basis(; spin)
        ham = DFTK.Hamiltonian(basis; ρ=DFTK.guess_density(basis))
        ref = dense_reference(basis, ham)
        backend = ChebyshevFOE(; degree=100)
        exact = stochastic_density(basis, ham; backend, sampling=:complete, tol_n_elec=1e-10)
        @test length(basis.kpoints) >= 2
        @test density_distance(basis, exact.ρ, ref.ρ) < 1e-9
        @test abs(exact.εF - ref.εF) < 1e-8
        @test exact.n_electrons ≈ basis.model.n_electrons atol=1e-10
        @test exact.energies.total ≈ ref.energies.total atol=1e-8
        @test exact.internal_energy ≈ exact.energies.total - exact.entropy_energy
        @test exact.ρ ≈ DFTK.compute_density(basis, exact.ψ, exact.occupation)
        @test exact.timings.ham_applications == sum(300size(χ, 2) for χ in exact.probes.vectors)
        fixed = stochastic_density(basis, ham; backend, sampling=:complete, μ=ref.εF)
        @test density_distance(basis, fixed.ρ, ref.ρ) < 1e-10
        @test fixed.chemical_potential_iterations == 0

        stochastic = stochastic_density(basis, ham; backend, n_samples=32, seed=7, tol_n_elec=1e-10)
        repeat = stochastic_density(basis, ham; backend, probes=stochastic.probes, tol_n_elec=1e-10)
        @test stochastic.ρ == repeat.ρ
        @test stochastic.εF == repeat.εF
        @test stochastic.n_electrons ≈ 1 atol=1e-10
        @test sum(stochastic.ρ) * basis.dvol ≈ stochastic.n_electrons
        @test_throws ArgumentError stochastic_density(basis, ham; probes=StochasticOrbitals(toy_basis()))
        @test_throws ArgumentError stochastic_density(basis, ham; tol_n_elec=-1)
        @test_throws ArgumentError stochastic_density(basis, ham; spectral_bounds=[(0, 1)])
    end
    @test_throws ArgumentError StochasticOrbitals(toy_basis(; temperature=0.0))
end
