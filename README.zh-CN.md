# stoDFTK.jl

基于 DFTK.jl 的有限温度随机波函数密度泛函理论包。`main` 实现 Chebyshev 费米算符展开和完整自洽迭代；`feature/lanczos` 在共同框架上增加 Lanczos 后端。公开版本不包含赝势数据文件，测试和示例无需下载赝势。

## 安装与运行

需要 Julia ≥ 1.10、DFTK 0.7.25 / 0.7.26。实际验证版本见 [验证记录](docs/validation.md)。

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. examples/nonlinear_scf.jl
julia --project=. examples/helium_scf.jl
```

从其他 Julia 项目使用本地包：

```julia
using Pkg
Pkg.add(name="DFTK", version="0.7")
Pkg.develop(path="/path/to/stoDFTK.jl")
using DFTK, stoDFTK
```

## 自洽计算

```julia
# basis 是 DFTK.PlaneWaveBasis，模型需采用正温度的 FermiDirac 分布。
result = stochastic_scf(basis;
    backend=ChebyshevFOE(degree=100),
    n_samples=64, seed=2026,
    maxiter=80, tol=1e-6, damping=0.3,
    mixing=DFTK.KerkerMixing(),
)

result.converged
result.ρ                     # DFTK 实空间密度，含自旋通道
result.εF                    # 化学势，Hartree
result.energies.total         # 有限温度自由能，Hartree
result.internal_energy       # 不含 -TS 的能量
result.history               # 每轮未混合密度残差、自由能、电子数、耗时
```

每轮执行 `ρin → H[ρin] → μ → sqrt(f(H))χ → ρout → 密度混合`。默认固定随机样本，使迭代求解同一个随机映射。化学势与密度使用同一批样本和同一个展开算符，输出密度不做事后电子数归一化。

`tol` 对应 `norm(ρout - ρin) * sqrt(basis.dvol)`。达到 `maxiter` 会返回 `converged=false`，不会伪装成收敛。`callback` 接收每轮 `stage=:iterate` 和结束时 `stage=:finalize`。

## 固定 Hamiltonian 与小体系验证

```julia
# ρin 是 DFTK 实空间网格上的非负试探密度。
ham = DFTK.Hamiltonian(basis; ρ=ρin)
probes = StochasticOrbitals(basis; n_samples=64, seed=2026, sampling=:phase)
out = stochastic_density(basis, ham; backend=ChebyshevFOE(degree=100), probes)

# 固定化学势可用于与对角化结果比较。
out_fixed_mu = stochastic_density(basis, ham; probes, μ=out.εF)

# 完整基只用于很小的体系，消除随机采样误差。
out_complete = stochastic_density(basis, ham; sampling=:complete)
```

`sampling` 支持 `:phase`、`:rademacher`、`:complete`。随机相位探针满足 `E[χχ†]=I`，没有除以平面波维数；统计平均权重为 `1/n_samples`。完整基的每个单位向量权重为 1。

自动谱界使用动能与局域势极值以及非局域赝势在投影子空间中的范数界，不对完整 Hamiltonian 对角化。可用 `spectral_bounds=(emin, emax)` 或每个 k 点一个元组的向量指定谱包围区间。错误谱界可能导致展开发散，包会检查递推是否越界。扩大量级和样本数需要分别做收敛测试。

低成本示例：

```sh
julia --project=. examples/helium_scf.jl --complete
```

该全电子 He 示例使用解析库仑原子、Γ 点和 `Ecut=1.5 Ha`，旨在验证 PBE 和 SCF 接口；该截断**不代表物理量收敛**。

## 当前范围

支持 Float64 / ComplexF64、CPU 线程与探针批处理、CPU 的 k 点或探针 MPI、CUDA 与经 CPU 暂存的 GPU 探针 MPI。也支持多个 k 点、非自旋极化 / spinless / 共线自旋、LDA / GGA、调用方提供的模守恒赝势，以及 Simple / Kerker / Dielectric 混合。本次使用本地 CUDA 13.4 的 `local=true` 配置禁用了 Libxc GPU 预编译包，原生 GPU 交换关联尚未验证；DFTK 内置泛函可在 GPU basis 上经 CPU 计算交换关联。GPU 的 k 点 MPI、k 点 × 探针二维 MPI、AMD 路径暂不支持。配置和测试记录见 [并行说明](docs/parallelization.md)。

暂不支持零温、混合确定性子空间、杂化泛函、需要动能密度的 meta-GGA、力或分子动力学。

`ψ` 是经过 `sqrt(f(H))` 滤波的随机波函数，`occupation` 是估计器权重，二者可传给 `DFTK.compute_density`；探针 MPI 下还需对探针通信器求和。它们不是 KS 本征态和占据数，不能直接用于依赖本征值的 DFTK 后处理。SCF 返回 `ham=H[ρout]`，同时保存用于最后一次滤波的 `ham_input=H[ρin]`；两者的密度差由 `residual_norm` 表示。

随机 SCF 的残差收敛不等于样本数收敛；自由能含随机及展开误差。熵通过单独的算符迹估计，细节见 [算法说明](docs/method.md)。

## 许可

代码采用 [MIT](LICENSE) 许可证；依赖的许可见 [第三方说明](THIRD_PARTY_NOTICES.md)。
