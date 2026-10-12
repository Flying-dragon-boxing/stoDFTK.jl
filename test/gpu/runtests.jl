using Test, DFTK, DftFunctionals, LinearAlgebra, Random, MPI, CUDA, stoDFTK

if !CUDA.functional()
    println("GPU validation unavailable: CUDA.functional() == false (no working NVIDIA device/driver)")
    exit(2)
end
CUDA.allowscalar(false)
CUDA.versioninfo()
MPI.Initialized() || MPI.Init()
println("MPI.has_cuda() = ", MPI.has_cuda())
DFTK.setup_threading(; n_fft=1, n_blas=1, n_DFTK=1)
include("../helpers.jl")

@testset "Experimental CUDA Chebyshev and SCF" begin
    # A tiny dense reference independently checks the square-root filter.
    H = ComplexF64[0.2 0.1im; -0.1im 0.7]
    z = ComplexF64[1, im]
    cpu_action = fermi_operator_action(H, z, 0.3, 0.4)
    gpu_action = fermi_operator_action(CuArray(H), CuArray(z), 0.3, 0.4)
    @test gpu_action isa CuArray
    @test Array(gpu_action) ≈ cpu_action atol=1e-10

    cpu = toy_basis(; comm_kpts=MPI.COMM_SELF)
    gpu = PlaneWaveBasis(cpu.model; Ecut=cpu.Ecut, kgrid=cpu.kgrid,
                          architecture=DFTK.GPU(CuArray), comm_kpts=MPI.COMM_SELF)
    backend = ChebyshevFOE(; degree=60, batch_size=8)
    cpu_ham = DFTK.Hamiltonian(cpu; ρ=stoDFTK.default_density(cpu))
    gpu_ham = DFTK.Hamiltonian(gpu; ρ=stoDFTK.default_density(gpu))
    for sampling in (:phase, :rademacher, :complete)
        reference = stochastic_density(cpu, cpu_ham; backend, sampling, n_samples=17, seed=77)
        result = stochastic_density(gpu, gpu_ham; backend, sampling, n_samples=17, seed=77)
        @test result.ρ isa CuArray
        @test all(ψ -> ψ isa CuArray, result.ψ)
        @test all(occ -> occ isa CuArray, result.occupation)
        @test density_distance(cpu, reference.ρ, Array(result.ρ)) < 1e-7
        @test reference.εF ≈ result.εF atol=1e-7
        @test reference.energies.total ≈ result.energies.total atol=1e-7
        println("CUDA $sampling: Δρ=$(density_distance(cpu, reference.ρ, Array(result.ρ))), " *
                "ΔF=$(abs(reference.energies.total-result.energies.total)) Ha")
    end
    reference = stochastic_scf(cpu; backend, n_samples=17, seed=77, tol=1e-7, damping=0.6)
    result = stochastic_scf(gpu; backend, n_samples=17, seed=77, tol=1e-7, damping=0.6,
                            ρ=stoDFTK.default_density(cpu))
    @test result.converged
    @test density_distance(cpu, reference.ρ, Array(result.ρ)) < 1e-6
    @test reference.energies.total ≈ result.energies.total atol=1e-6
    println("CUDA nonlinear SCF: iterations=$(result.n_iter), residual=$(result.residual_norm), " *
            "ΔF=$(abs(reference.energies.total-result.energies.total)) Ha")

    # Full Hartree/PBE integration without distributing pseudopotential files.
    native_xc = DFTK.Libxc.has_cuda()
    functionals = native_xc ? PBE() : DftFunctional.([:gga_x_pbe, :gga_c_pbe])
    println("He XC evaluation: ", native_xc ? "native Libxc CUDA" :
            "DFTK built-in PBE on CPU; native Libxc CUDA unavailable")
    cpu_he = helium_basis(; comm_kpts=MPI.COMM_SELF, functionals)
    gpu_he = PlaneWaveBasis(cpu_he.model; Ecut=cpu_he.Ecut, kgrid=cpu_he.kgrid,
                           architecture=DFTK.GPU(CuArray), comm_kpts=MPI.COMM_SELF)
    he_backend = ChebyshevFOE(; degree=100, batch_size=32)
    reference_he = stochastic_scf(cpu_he; backend=he_backend, sampling=:complete,
                                  damping=0.5, tol=1e-7)
    result_he = stochastic_scf(gpu_he; backend=he_backend, sampling=:complete,
                               damping=0.5, tol=1e-7)
    @test reference_he.converged && result_he.converged
    @test result_he.ρ isa CuArray
    @test density_distance(cpu_he, reference_he.ρ, Array(result_he.ρ)) < 1e-6
    @test reference_he.energies.total ≈ result_he.energies.total atol=1e-6
    if !native_xc
        libxc_cpu = helium_basis(; comm_kpts=MPI.COMM_SELF)
        libxc_reference = stochastic_scf(libxc_cpu; backend=he_backend, sampling=:complete,
                                         damping=0.5, tol=1e-7)
        @test density_distance(cpu_he, reference_he.ρ, libxc_reference.ρ) < 1e-6
        @test reference_he.energies.total ≈ libxc_reference.energies.total atol=1e-6
    end
    println("CUDA He PBE SCF: dim=$(size(result_he.ham[1],1)), iterations=$(result_he.n_iter), " *
            "residual=$(result_he.residual_norm), ΔF=$(abs(reference_he.energies.total-result_he.energies.total)) Ha")
end
