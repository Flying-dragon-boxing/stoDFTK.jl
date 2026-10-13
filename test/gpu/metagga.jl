using Test, DFTK, LinearAlgebra, Random, MPI, CUDA, stoDFTK
include("../helpers.jl")
CUDA.functional() || error("CUDA is not functional")
DFTK.Libxc.has_cuda() || error("native GPU Libxc is required for this test")
CUDA.allowscalar(false)
DFTK.disable_threading()

@testset "Native CUDA meta-GGA" begin
    cpu = metagga_basis(; comm_kpts=MPI.COMM_SELF)
    gpu = PlaneWaveBasis(cpu.model; Ecut=cpu.Ecut, kgrid=cpu.kgrid,
                         comm_kpts=MPI.COMM_SELF, architecture=DFTK.GPU(CuArray))
    backend = ChebyshevFOE(; degree=120)
    ρ = stoDFTK.physical_density(cpu, stoDFTK.default_density(cpu))
    τ = stoDFTK.physical_tau(cpu, ρ, nothing)
    cpu_ham = DFTK.Hamiltonian(cpu; ρ, τ)
    gpu_ham = DFTK.Hamiltonian(gpu; ρ=CuArray(ρ), τ=CuArray(τ))
    for sampling in (:complete, :phase)
        reference = stochastic_density(cpu, cpu_ham; backend, sampling, n_samples=17, seed=73)
        result = stochastic_density(gpu, gpu_ham; backend, sampling, n_samples=17, seed=73)
        @test result.τ isa CuArray
        @test density_distance(cpu, reference.ρ, Array(result.ρ)) < 1e-8
        @test density_distance(cpu, reference.τ, Array(result.τ)) < 1e-8
        @test result.energies.total ≈ reference.energies.total atol=1e-7
    end
    for sampling in (:complete, :phase)
        reference = stochastic_scf(cpu; backend, sampling, n_samples=17, seed=73,
                                   damping=0.4, tol=1e-7, tol_tau=1e-7, maxiter=100)
        result = stochastic_scf(gpu; backend, sampling, n_samples=17, seed=73,
                                damping=0.4, tol=1e-7, tol_tau=1e-7, maxiter=100, ρ, τ)
        @test reference.converged && result.converged
        @test result.τ isa CuArray
        @test density_distance(cpu, reference.τ, Array(result.τ)) < 1e-7
        @test result.energies.total ≈ reference.energies.total atol=1e-7
    end
end
