# Third-party notices

## DFTK.jl

stoDFTK depends on [DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl), which is
licensed under the [MIT License](https://github.com/JuliaMolSim/DFTK.jl/blob/master/LICENSE).

DFTK's copyright notice is:

> Copyright (c) 2019 Michael F. Herbst and Antoine Levitt

DFTK is installed as a Julia dependency rather than vendored in this repository.
Its own distribution retains its license and copyright notices.

## Data and other dependencies

No pseudopotential data files are distributed in this repository. Tests and
examples use analytic model potentials and an all-electron Coulomb atom.

Other Julia packages and binary libraries are installed by Julia's package
manager as dependencies and retain their own licenses. They are not vendored or
relicensed by stoDFTK's MIT License.
