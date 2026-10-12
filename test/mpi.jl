using Test, DFTK, LinearAlgebra, Random, MPI, stoDFTK
include("helpers.jl")

MPI.Initialized() || MPI.Init()
MPI.Comm_size(MPI.COMM_WORLD) >= 2 || error("run with at least two MPI processes")
DFTK.disable_threading()
rank = MPI.Comm_rank(MPI.COMM_WORLD)

@testset "MPI k-points and probes" begin
    for spin in (:spinless, :collinear), kgrid in ((1, 1, 1), (3, 1, 1))
        serial_basis = toy_basis(; spin, kgrid, comm_kpts=MPI.COMM_SELF)
        distributed_basis = toy_basis(; spin, kgrid)
        serial_ham = DFTK.Hamiltonian(serial_basis; ρ=stoDFTK.default_density(serial_basis))
        distributed_ham = DFTK.Hamiltonian(distributed_basis; ρ=stoDFTK.default_density(distributed_basis))
        for sampling in (:phase, :complete)
            serial = stochastic_density(serial_basis, serial_ham; sampling, n_samples=7, seed=54)
            distributed = stochastic_density(distributed_basis, distributed_ham; sampling, n_samples=7, seed=54)
            @test density_distance(serial_basis, serial.ρ, distributed.ρ) < 1e-7
            @test serial.εF ≈ distributed.εF atol=1e-7
            @test serial.energies.total ≈ distributed.energies.total atol=1e-7
            @test distributed.n_electrons ≈ 1 atol=1e-8
            # Probe MPI replicates the basis and divides the global sample set.
            probe_distributed = stochastic_density(serial_basis, serial_ham; sampling, n_samples=7,
                                                    seed=54, probe_comm=MPI.COMM_WORLD)
            @test density_distance(serial_basis, serial.ρ, probe_distributed.ρ) < 1e-7
            @test serial.εF ≈ probe_distributed.εF atol=1e-7
            @test serial.energies.total ≈ probe_distributed.energies.total atol=1e-7
            @test probe_distributed.n_electrons ≈ 1 atol=1e-8
        end
        @test_throws ArgumentError StochasticOrbitals(distributed_basis; probe_comm=MPI.COMM_WORLD)
    end
    # More ranks than probes: some ranks must contribute zero orbitals safely.
    basis = toy_basis(; kgrid=(1, 1, 1), comm_kpts=MPI.COMM_SELF)
    ham = DFTK.Hamiltonian(basis; ρ=stoDFTK.default_density(basis))
    serial = stochastic_density(basis, ham; n_samples=1, seed=54)
    probes = StochasticOrbitals(basis; n_samples=1, seed=54, probe_comm=MPI.COMM_WORLD)
    distributed = stochastic_density(basis, ham; probes)
    @test density_distance(basis, serial.ρ, distributed.ρ) < 1e-7
    @test serial.energies.total ≈ distributed.energies.total atol=1e-7
    @test MPI.Allreduce(Int(isempty(probes.indices[1])), +, MPI.COMM_WORLD) >= 1
    for mode in (:kpoints, :probes)
        distributed_basis = toy_basis(; comm_kpts=mode == :kpoints ? MPI.COMM_WORLD : MPI.COMM_SELF)
        pc = mode == :probes ? MPI.COMM_WORLD : MPI.COMM_SELF
        reference_basis = toy_basis(; comm_kpts=MPI.COMM_SELF)
        reference = stochastic_scf(reference_basis; seed=71, n_samples=7, damping=0.6, tol=1e-7)
        result = stochastic_scf(distributed_basis; seed=71, n_samples=7, damping=0.6, tol=1e-7, probe_comm=pc)
        @test result.converged && reference.converged
        @test density_distance(reference_basis, result.ρ, reference.ρ) < 1e-7
        @test result.energies.total ≈ reference.energies.total atol=1e-7
    end
    rank == 0 && println("MPI validation passed on $(MPI.Comm_size(MPI.COMM_WORLD)) ranks")
end
