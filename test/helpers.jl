function toy_basis(; spin=:spinless, temperature=0.4, kgrid=(3, 1, 1), comm_kpts=MPI.COMM_WORLD)
    lattice = [6.0 0 0; 0 0 0; 0 0 0]
    terms = [Kinetic(), ExternalFromReal(r -> 0.3cos(2π * r[1] / 6) +
                                            0.1sin(4π * r[1] / 6)),
             LocalNonlinearity(ρ -> 0.5ρ^2), Entropy()]
    model = Model(lattice; n_electrons=1, terms, temperature,
                  spin_polarization=spin, symmetries=false,
                  disable_electrostatics_check=true)
    PlaneWaveBasis(model; Ecut=4.0, kgrid, comm_kpts)
end

function dense_reference(basis, ham)
    eigensystems = map(eachindex(basis.kpoints)) do ik
        dim = size(ham[ik], 1)
        eigen(Hermitian(ham[ik] * Matrix{ComplexF64}(I, dim, dim)))
    end
    eigenvalues = [e.values for e in eigensystems]
    ψ = [e.vectors for e in eigensystems]
    # Use exact eigenvalues and DFTK occupations for an independent reference.
    # DFTK's ±1 Ha bracket can be too narrow for a small, hot, dilute model.
    lo = minimum(minimum, eigenvalues) - 64basis.model.temperature
    hi = maximum(maximum, eigenvalues) + 64basis.model.temperature
    εF = (lo + hi) / 2
    for _ in 1:100
        εF = (lo + hi) / 2
        occupations = DFTK.compute_occupation(basis, eigenvalues, εF).occupation
        count = sum(basis.kweights[ik] * sum(occ) for (ik, occ) in enumerate(occupations))
        abs(count - basis.model.n_electrons) <= 1e-12 && break
        if count < basis.model.n_electrons
            lo = εF
        else
            hi = εF
        end
    end
    occupation = DFTK.compute_occupation(basis, eigenvalues, εF).occupation
    ρ = DFTK.compute_density(basis, ψ, occupation)
    energies = DFTK.energy(basis, ψ, occupation; ρ, eigenvalues, εF).energies
    (; ψ, occupation, εF, ρ, energies, eigenvalues)
end

density_distance(basis, a, b) = norm(a - b) * sqrt(basis.dvol)

function helium_basis(; comm_kpts=MPI.COMM_WORLD, functionals=PBE())
    lattice = 6.0 * Matrix{Float64}(I, 3, 3)
    model = model_DFT(lattice, [ElementCoulomb(:He)], [[0.0, 0.0, 0.0]];
                      functionals, temperature=0.75)
    PlaneWaveBasis(model; Ecut=1.5, kgrid=(1, 1, 1), comm_kpts)
end
