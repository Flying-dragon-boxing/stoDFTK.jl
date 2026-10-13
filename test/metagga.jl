@testset "DivAgrad spectral enclosures" begin
    basis = toy_basis()
    for kpt in basis.kpoints, mode in (:positive, :negative, :mixed)
        A = [mode == :positive ? 0.7 : mode == :negative ? -0.7 :
             0.7cos(2π*r[1]/6) for r in DFTK.r_vectors_cart(basis)]
        op = DFTK.DivAgradOperator(basis, kpt, A)
        H = DFTK.HamiltonianBlock(basis, kpt, [op])
        dim = size(H, 1)
        matrix = H * Matrix{ComplexF64}(I, dim, dim)
        @test matrix ≈ matrix' atol=1e-12
        lo, hi = spectral_bounds(H)
        values = eigvals(Hermitian(matrix))
        @test lo < minimum(values) <= maximum(values) < hi
    end
end

@testset "Small all-electron He r²SCAN SCF" begin
    basis = helium_basis(; functionals=[:mgga_x_r2scan,:mgga_c_r2scan])
    result = stochastic_scf(basis; sampling=:complete,
                            backend=ChebyshevFOE(; degree=150), damping=0.5,
                            tol=1e-7, tol_tau=1e-7, maxiter=100)
    @test size(result.ham[1],1) == 19
    @test result.converged
    @test result.residual_tau < 1e-7
    reference = dense_reference(basis, result.ham)
    @test density_distance(basis, result.ρ, reference.ρ) < 1e-7
    @test density_distance(basis, result.τ, reference.τ) < 1e-7
    @test result.energies.total ≈ reference.energies.total atol=1e-6
    println("He r²SCAN: iterations=$(result.n_iter), Δρ=$(result.residual_norm), " *
            "Δτ=$(result.residual_tau), ΔF=$(abs(result.energies.total-reference.energies.total)) Ha")
end

@testset "Stochastic meta-GGA density and kinetic density" begin
    for spin in (:none, :collinear)
        basis = metagga_basis(; spin)
        ρin = stoDFTK.physical_density(basis, stoDFTK.default_density(basis))
        τin = stoDFTK.physical_tau(basis, ρin, nothing)
        ham = DFTK.Hamiltonian(basis; ρ=ρin, τ=τin)
        reference = dense_reference(basis, ham)
        backend = ChebyshevFOE(; degree=120)
        result = stochastic_density(basis, ham; backend, sampling=:complete)
        @test maximum(size(H,1) for H in ham.blocks) < 20
        @test density_distance(basis, result.ρ, reference.ρ) < 1e-8
        @test density_distance(basis, result.τ, reference.τ) < 1e-8
        @test result.energies.total ≈ reference.energies.total atol=1e-7
        @test result.εF ≈ reference.εF atol=1e-8
        @test result.τ ≈ DFTK.compute_kinetic_energy_density(basis, result.ψ, result.occupation)
        @test sum(result.τ)*basis.dvol ≈ result.energies["Kinetic"] atol=1e-9
        random = stochastic_density(basis, ham; backend, n_samples=17, seed=31)
        repeated = stochastic_density(basis, ham; backend, probes=random.probes)
        @test random.τ == repeated.τ
        @test sum(random.τ)*basis.dvol ≈ random.energies["Kinetic"] atol=1e-9
        scalar = stochastic_density(basis, ham; probes=random.probes,
                                     backend=ChebyshevFOE(; degree=120, batch_size=1))
        @test density_distance(basis, random.τ, scalar.τ) < 1e-9
        @test_throws ArgumentError stochastic_scf(basis; tol_tau=0)
        @test_throws DimensionMismatch stochastic_scf(basis; τ=zeros(2))
        @test_throws ArgumentError stochastic_scf(basis; τ=fill(-1.0,size(ρin)))
        @test_throws ArgumentError stochastic_scf(basis; τ=fill(NaN,size(ρin)))
        unconverged = stochastic_scf(basis; backend, maxiter=1, tol=1e3, tol_tau=1e-14)
        @test !unconverged.converged
        @test unconverged.residual_tau > 1e-14
    end
end

@testset "Meta-GGA joint self-consistency" begin
    basis = metagga_basis()
    backend = ChebyshevFOE(; degree=120)
    for sampling in (:complete, :phase), mixing in (SimpleMixing(), KerkerMixing(), DielectricMixing())
        result = stochastic_scf(basis; backend, sampling, n_samples=24, seed=42,
                                damping=0.4, tol=1e-7, tol_tau=1e-7, maxiter=100, mixing)
        @test result.converged
        @test result.residual_norm <= 1e-7
        @test result.residual_tau <= 1e-7
        @test result.n_electrons ≈ 1 atol=1e-8
        @test result.history_Δτ[end] == result.residual_tau
        if sampling == :complete
            reference = dense_reference(basis, result.ham)
            @test density_distance(basis, result.ρ, reference.ρ) < 2e-7
            @test density_distance(basis, result.τ, reference.τ) < 2e-7
            @test result.energies.total ≈ reference.energies.total atol=1e-6
        end
    end
end
