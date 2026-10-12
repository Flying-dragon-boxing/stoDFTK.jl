@testset "Self-consistent nonlinear DFTK model" begin
    basis = toy_basis()
    stages = Symbol[]
    backend = ChebyshevFOE(; degree=100)
    result = stochastic_scf(basis; backend, sampling=:complete, maxiter=100, tol=1e-9,
                             damping=0.6, tol_n_elec=1e-10,
                             callback=info -> push!(stages, info.stage))
    @test result.converged
    @test 1 < result.n_iter < 100
    @test result.residual_norm <= 1e-9
    @test result.history[1].residual_norm > 100result.residual_norm
    @test result.history_Δρ == [h.residual_norm for h in result.history]
    @test stages == [fill(:iterate, result.n_iter); :finalize]
    @test result.n_electrons ≈ 1 atol=1e-10
    @test result.residual_norm ≈ density_distance(basis, result.ρ, result.ρin)
    # Validate the fixed point independently using full diagonalization.
    ref = dense_reference(basis, result.ham)
    @test density_distance(basis, result.ρ, ref.ρ) < 1e-8
    @test abs(result.energies.total - ref.energies.total) < 1e-8
    @test abs(result.εF - ref.εF) < 1e-8

    for mixing in (SimpleMixing(), KerkerMixing(), DielectricMixing())
        stochastic = stochastic_scf(basis; backend, n_samples=24, seed=36,
                                     maxiter=100, damping=0.6, tol=1e-7, mixing)
        again = stochastic_scf(basis; backend, probes=stochastic.probes,
                                maxiter=100, damping=0.6, tol=1e-7, mixing)
        @test stochastic.converged
        @test stochastic.ρ == again.ρ
        @test stochastic.history_Δρ == again.history_Δρ
        @test stochastic.n_electrons ≈ 1 atol=1e-8
    end
    limited = stochastic_scf(basis; backend, n_samples=4, maxiter=1, tol=1e-14)
    @test !limited.converged
    @test limited.n_iter == 1
    @test limited.residual_norm > 1e-14
    @test limited.residual_norm ≈ density_distance(basis, limited.ρ, limited.ρin)
    @test_throws ArgumentError stochastic_scf(basis; maxiter=0)
    @test_throws ArgumentError stochastic_scf(basis; damping=0)
    @test_throws ArgumentError stochastic_scf(basis; tol=0)
    @test_throws ArgumentError stochastic_scf(basis; ρ=zero(DFTK.guess_density(basis)))
    @test_throws ArgumentError stochastic_scf(basis; μ=0.1)
    @test_throws ArgumentError stochastic_scf(basis; mixing=DFTK.KerkerDosMixing())
end

@testset "Free electrons without a local-potential operator" begin
    model = Model([6.0 0 0; 0 0 0; 0 0 0]; terms=[Kinetic(), Entropy()],
                  n_electrons=1, temperature=0.4, spin_polarization=:spinless,
                  disable_electrostatics_check=true)
    basis = PlaneWaveBasis(model; Ecut=4.0, kgrid=(1, 1, 1))
    result = stochastic_scf(basis; sampling=:complete, tol=1e-7)
    @test result.converged
    @test result.n_electrons ≈ 1 atol=1e-8
    ref = dense_reference(basis, result.ham)
    @test density_distance(basis, result.ρ, ref.ρ) < 1e-8
    @test result.energies.total ≈ ref.energies.total atol=1e-8
end
