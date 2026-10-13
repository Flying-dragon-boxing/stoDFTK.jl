using Test, DFTK, LinearAlgebra, Random, MPI, stoDFTK
include("helpers.jl")
MPI.Initialized() || MPI.Init()
MPI.Comm_size(MPI.COMM_WORLD) >= 2 || error("run with at least two MPI processes")
DFTK.disable_threading()

@testset "Meta-GGA MPI density, kinetic density and SCF" begin
    serial_basis = metagga_basis(; comm_kpts=MPI.COMM_SELF)
    backend = ChebyshevFOE(; degree=120)
    for mode in (:kpoints, :probes)
        basis = metagga_basis(; comm_kpts=mode == :kpoints ? MPI.COMM_WORLD : MPI.COMM_SELF)
        pc = mode == :probes ? MPI.COMM_WORLD : MPI.COMM_SELF
        ρ = stoDFTK.physical_density(basis, stoDFTK.default_density(basis))
        τ = stoDFTK.physical_tau(basis, ρ, nothing)
        ham = DFTK.Hamiltonian(basis; ρ, τ)
        serial_ham = DFTK.Hamiltonian(serial_basis; ρ, τ)
        for sampling in (:complete, :phase), n_samples in (1,7)
            serial = stochastic_density(serial_basis, serial_ham; backend, sampling, n_samples, seed=67)
            result = stochastic_density(basis, ham; backend, sampling, n_samples, seed=67, probe_comm=pc)
            @test density_distance(serial_basis, serial.ρ, result.ρ) < 1e-8
            @test density_distance(serial_basis, serial.τ, result.τ) < 1e-8
            @test result.energies.total ≈ serial.energies.total atol=1e-8
            @test sum(result.τ)*basis.dvol ≈ result.energies["Kinetic"] atol=1e-8
        end
        serial = stochastic_scf(serial_basis; backend, n_samples=7, seed=67, damping=0.4,
                                tol=1e-7, tol_tau=1e-7, maxiter=100)
        result = stochastic_scf(basis; backend, n_samples=7, seed=67, damping=0.4,
                                tol=1e-7, tol_tau=1e-7, maxiter=100, probe_comm=pc)
        @test serial.converged && result.converged
        @test density_distance(serial_basis, serial.τ, result.τ) < 1e-7
        @test result.energies.total ≈ serial.energies.total atol=1e-7
    end
end
