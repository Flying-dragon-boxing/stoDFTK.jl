using Test
get(ENV, "STODFTK_USE_MKL", "false") == "true" && (@eval using MKL)
using DFTK
using LinearAlgebra
using Random
using MPI
using stoDFTK

include("helpers.jl")
include("chebyshev.jl")
include("density.jl")
include("scf.jl")
include("helium.jl")
include("batching.jl")
include("metagga.jl")
