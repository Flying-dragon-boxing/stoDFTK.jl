using MPI
script = length(ARGS) >= 2 ? abspath(ARGS[2]) : joinpath(@__DIR__, "mpi.jl")
nranks = isempty(ARGS) ? 2 : parse(Int, first(ARGS))
active_project = dirname(Base.active_project())
command = `$(MPI.mpiexec()) -n $nranks $(Base.julia_cmd()) --project=$active_project $script`
run(command)
