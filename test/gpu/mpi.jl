using Test, DFTK, LinearAlgebra, Random, MPI, CUDA, stoDFTK
include("../helpers.jl")
MPI.Initialized() || MPI.Init()
MPI.Comm_size(MPI.COMM_WORLD) >= 2 || error("run with at least two MPI processes")
CUDA.functional() || error("CUDA is not functional")
CUDA.allowscalar(false)
DFTK.disable_threading()

@testset "CUDA probe MPI with host staging" begin
    cpu = toy_basis(; kgrid=(1,1,1), comm_kpts=MPI.COMM_SELF)
    gpu = PlaneWaveBasis(cpu.model; Ecut=cpu.Ecut, kgrid=cpu.kgrid,
                         architecture=DFTK.GPU(CuArray), comm_kpts=MPI.COMM_SELF)
    cpu_ham = DFTK.Hamiltonian(cpu; ρ=stoDFTK.default_density(cpu))
    gpu_ham = DFTK.Hamiltonian(gpu; ρ=stoDFTK.default_density(gpu))
    backend = ChebyshevFOE(; degree=60, batch_size=8)
    for n_samples in (1, 7)
        reference = stochastic_density(cpu, cpu_ham; backend, n_samples, seed=92)
        result = stochastic_density(gpu, gpu_ham; backend, n_samples, seed=92,
                                     probe_comm=MPI.COMM_WORLD)
        @test result.ρ isa CuArray
        @test density_distance(cpu, reference.ρ, Array(result.ρ)) < 1e-7
        @test reference.energies.total ≈ result.energies.total atol=1e-7
        @test result.n_electrons ≈ 1 atol=1e-8
    end
    reference = stochastic_scf(cpu; backend, n_samples=7, seed=92, damping=0.6, tol=1e-7)
    result = stochastic_scf(gpu; backend, n_samples=7, seed=92, damping=0.6, tol=1e-7,
                            probe_comm=MPI.COMM_WORLD)
    @test reference.converged && result.converged
    @test density_distance(cpu, reference.ρ, Array(result.ρ)) < 1e-6
    @test reference.energies.total ≈ result.energies.total atol=1e-6
end

@testset "CUDA meta-GGA probe MPI" begin
    cpu = metagga_basis(; kgrid=(1,1,1), comm_kpts=MPI.COMM_SELF)
    gpu = PlaneWaveBasis(cpu.model; Ecut=cpu.Ecut, kgrid=cpu.kgrid,
                         architecture=DFTK.GPU(CuArray), comm_kpts=MPI.COMM_SELF)
    backend = ChebyshevFOE(; degree=120)
    ρ = stoDFTK.physical_density(cpu, stoDFTK.default_density(cpu))
    τ = stoDFTK.physical_tau(cpu, ρ, nothing)
    cpu_ham = DFTK.Hamiltonian(cpu; ρ, τ)
    gpu_ham = DFTK.Hamiltonian(gpu; ρ=CuArray(ρ), τ=CuArray(τ))
    for n_samples in (1,7)
        reference = stochastic_density(cpu, cpu_ham; backend, n_samples, seed=92)
        result = stochastic_density(gpu, gpu_ham; backend, n_samples, seed=92,
                                     probe_comm=MPI.COMM_WORLD)
        @test result.τ isa CuArray
        @test density_distance(cpu, reference.τ, Array(result.τ)) < 1e-8
        @test result.energies.total ≈ reference.energies.total atol=1e-7
    end
    reference = stochastic_scf(cpu; backend, n_samples=7, seed=92, damping=0.4,
                               tol=1e-7, tol_tau=1e-7, maxiter=100)
    result = stochastic_scf(gpu; backend, n_samples=7, seed=92, damping=0.4,
                            tol=1e-7, tol_tau=1e-7, maxiter=100, probe_comm=MPI.COMM_WORLD)
    @test reference.converged && result.converged
    @test density_distance(cpu, reference.τ, Array(result.τ)) < 1e-7
    @test result.energies.total ≈ reference.energies.total atol=1e-7
end
