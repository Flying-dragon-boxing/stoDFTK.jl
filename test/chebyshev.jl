@testset "Chebyshev operator action" begin
    rng = MersenneTwister(28)
    A = randn(rng, ComplexF64, 9, 9)
    H = Hermitian((A + A') / 4)
    χ = randn(rng, ComplexF64, 9)
    μ, temperature = 0.1, 0.7
    e = eigen(H)
    ref = e.vectors * (sqrt.(stoDFTK.fermi_occupation.(e.values, μ, temperature, 2)) .*
                       (e.vectors' * χ))
    y = fermi_operator_action(H, χ, μ, temperature; backend=ChebyshevFOE(; degree=100))
    @test norm(y - ref) / norm(ref) < 1e-10
    bounds = spectral_bounds(H)
    @test bounds[1] <= minimum(e.values) <= maximum(e.values) <= bounds[2]
    @test fermi_operator_action(H, zero(χ), μ, temperature) == zero(χ)
    C = Diagonal(fill(0.4, 5))
    @test fermi_operator_action(C, ones(5), μ, temperature) ≈
          fill(sqrt(stoDFTK.fermi_occupation(0.4, μ, temperature, 2)), 5)
    @test_throws ArgumentError ChebyshevFOE(; degree=0)
    @test_throws ArgumentError ChebyshevFOE(; degree=10, quadrature_order=10)
    @test_throws ArgumentError fermi_operator_action(H, χ, μ, 0.0)
    @test_throws ArgumentError fermi_operator_action(H, χ, μ, temperature; spectral_bounds=(1.0, -1.0))
    @test_throws ArgumentError fermi_operator_action(H, χ, μ, temperature; spectral_bounds=(-0.01, 0.01))
    @test_throws DimensionMismatch fermi_operator_action(H, ones(2), μ, temperature)

    cache = stoDFTK.prepare_cache(ChebyshevFOE(; degree=100), H, reshape(χ, :, 1), nothing)
    count = only(stoDFTK.count_samples(cache, μ, temperature, 2.0))
    @test count ≈ sum(abs2, y) atol=1e-11
    # Solving μ must not apply H again: cached moments alone give the count.
    @test cache.ham_applications == 200

    # Test the nonlocal projector norm with synthetic matrices; no PSP data is needed.
    P = randn(rng, ComplexF64, 9, 3)
    D = Diagonal([-0.8, 0.4, 1.2])
    radius = stoDFTK.nonlocal_radius((; P, D))
    @test radius ≈ maximum(abs, eigvals(Hermitian(P * D * P'))) rtol=1e-12
    @test stoDFTK.nonlocal_radius((; P=zeros(ComplexF64, 9, 0), D=zeros(0, 0))) == 0.0
end
