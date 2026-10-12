get(ENV, "STODFTK_USE_MKL", "false") == "true" && (@eval using MKL)
using DFTK, LinearAlgebra, stoDFTK
device = get(ENV, "STODFTK_DEVICE", "cpu")
device in ("cpu", "cuda") || error("STODFTK_DEVICE must be cpu or cuda")
if device == "cuda"
    @eval using CUDA
    CUDA.functional() || error("CUDA is not functional")
    CUDA.allowscalar(false)
end
architecture = device == "cuda" ? DFTK.GPU(CUDA.CuArray) : DFTK.CPU()

# Moderate plane-wave basis, random probes only (no complete-basis expansion).
# Compare equal degree, sample set and chemical potential after warming up.
nthreads = Threads.nthreads()
nfft = parse(Int, get(ENV, "STODFTK_FFT_THREADS", "1"))
nblas = parse(Int, get(ENV, "STODFTK_BLAS_THREADS", "1"))
DFTK.setup_threading(; n_fft=nfft, n_blas=nblas, n_DFTK=nthreads)
lattice = 8.0 * Matrix{Float64}(I, 3, 3)
terms = [Kinetic(), ExternalFromReal(r -> 0.3cos(2π*r[1]/8)), Entropy()]
model = Model(lattice; terms, n_electrons=2, temperature=1.0,
              symmetries=false, disable_electrostatics_check=true)
basis = PlaneWaveBasis(model; Ecut=5.0, kgrid=(1, 1, 1), architecture)
ham = DFTK.Hamiltonian(basis; ρ=stoDFTK.default_density(basis))
probes = StochasticOrbitals(basis; n_samples=64, seed=2026)
reference_basis = device == "cpu" ? basis : PlaneWaveBasis(model; Ecut=5.0, kgrid=(1,1,1))
reference_ham = device == "cpu" ? ham : DFTK.Hamiltonian(reference_basis; ρ=stoDFTK.default_density(reference_basis))
reference_probes = device == "cpu" ? probes : StochasticOrbitals(reference_basis; n_samples=64, seed=2026)
reference = stochastic_density(reference_basis, reference_ham; probes=reference_probes,
                                 backend=ChebyshevFOE(; degree=30, batch_size=1))
println("device,threads,fft_threads,blas_threads,blas,dim,batch_size,seconds,density_error,energy_error,ham_batch_calls")
for batch_size in (device == "cpu" ? (1, 8, 32, 64) : (32, 64))
    backend = ChebyshevFOE(; degree=30, batch_size)
    function run_case()
        result = stochastic_density(basis, ham; probes, backend, μ=reference.εF)
        DFTK.synchronize_device(basis.architecture)
        result
    end
    run_case()
    trials = Float64[]
    for _ in 1:3
        GC.gc()
        push!(trials, @elapsed run_case())
    end
    result = run_case()
    blas_library = basename(first(BLAS.get_config().loaded_libs).libname)
    println("$device,$nthreads,$nfft,$nblas,$blas_library,$(size(ham[1],1)),$batch_size," *
            "$(sort(trials)[2]),$(norm(DFTK.to_cpu(result.ρ)-reference.ρ)*sqrt(basis.dvol))," *
            "$(abs(result.energies.total-reference.energies.total)),$(result.timings.ham_batch_calls)")
end
