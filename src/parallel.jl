# k-point distribution is owned by DFTK. Probe distribution uses a separate
# communicator with a replicated basis (comm_kpts=COMM_SELF).
function validate_communicators(basis, probe_comm)
    nk = MPI.Comm_size(basis.comm_kpts)
    np = MPI.Comm_size(probe_comm)
    nk > 1 && basis.architecture isa DFTK.GPU && throw(ArgumentError(
        "GPU k-point MPI is not supported; use a replicated GPU basis with comm_kpts=MPI.COMM_SELF and probe_comm for host-staged probe MPI"))
    nk > 1 && np > 1 && throw(ArgumentError(
        "use either k-point MPI or probe MPI, not both; probe MPI requires comm_kpts=MPI.COMM_SELF"))
    nothing
end

trace_comm(basis, probes) = MPI.Comm_size(probes.probe_comm) > 1 ?
    probes.probe_comm : basis.comm_kpts

collective(value, op, comm) = MPI.Comm_size(comm) == 1 ? value : MPI.Allreduce(value, op, comm)

function reduce_probe_density!(ρ, comm)
    MPI.Comm_size(comm) == 1 && return ρ
    host = DFTK.to_cpu(ρ)
    MPI.Allreduce!(host, +, comm)
    host === ρ || copyto!(ρ, host)
    ρ
end

function local_probe_indices(total, comm)
    rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    collect(fld(total * rank, nranks) + 1:fld(total * (rank + 1), nranks))
end

# Deterministic physical-k-point and global-probe seeds. Avoid process-local
# enumeration: MPI may split or duplicate k-points and split probe columns.
function mix_seed(x::UInt64)
    x += 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    x ⊻ (x >> 31)
end

function probe_seed(seed, kpt, index)
    value = mix_seed(reinterpret(UInt64, Int64(seed)) ⊻ UInt64(kpt.spin))
    for coordinate in kpt.coordinate
        # Canonicalize signed zero for the same physical k-point.
        c = iszero(coordinate) ? 0.0 : Float64(coordinate)
        value = mix_seed(value ⊻ reinterpret(UInt64, c))
    end
    mix_seed(value ⊻ UInt64(index))
end

function column_dots(X::StridedMatrix, Y::StridedMatrix)
    [real(dot(@view(X[:, j]), @view(Y[:, j]))) for j in axes(X, 2)]
end
column_dots(X, Y) = DFTK.to_cpu(vec(sum(real.(conj.(X) .* Y); dims=1)))

column_norms2(X::StridedMatrix) = [sum(abs2, @view(X[:, j])) for j in axes(X, 2)]
column_norms2(X) = DFTK.to_cpu(vec(sum(abs2, X; dims=1)))

matrix_ishermitian(H::StridedMatrix) = ishermitian(H)
matrix_ishermitian(H) = all(H .== adjoint(H))

# Vector-only backends count every application as an individual call.
batch_application_count(cache) = application_count(cache)
