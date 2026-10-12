# Contributing

Use a local Julia development environment:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The `main` branch contains the Chebyshev implementation. Changes should keep
chemical-potential solving and filtered-density reconstruction consistent,
reuse fixed probes across SCF iterations, and test numerical changes against
small independent diagonalization references.

Keep complete-basis tests small. Use analytic test potentials and synthetic
operator matrices instead of adding pseudopotential data files or local
calculation outputs.

Open an issue or pull request with the numerical problem, the resulting
behavior, and relevant validation. Contributions are licensed under the
project's MIT License.
