# 数值验证

环境：Julia 1.13.1，DFTK 0.7.26，Float64 / ComplexF64。CPU 使用 Julia 1 / 4 线程、OpenBLAS / MKL，以及 2 / 3 个 MPICH 进程；GPU 为 RTX 4060 Laptop（8 GiB），CUDA.jl 6.3.1，驱动 615.78.08。先使用本地 CUDA 13.4 验证基础算符和 CPU XC 混合路径，随后通过无代理镜像下载 CUDA 13.3 预编译包，验证原生 GPU Libxc。GPU 测试禁用标量回退。

公开版本不包含赝势数据文件。测试使用一维解析周期模型、合成非局域投影矩阵，以及全电子 He 原子。He 仅用 Γ 点、`Ecut=1.5 Ha`，平面波维数为 19；完整基只用于这些小体系。

## 覆盖范围

- Hermitian 算符的平方根费米函数作用与完整对角化比较。
- 合成非局域投影矩阵的范数，以及空投影子空间。
- 复随机相位与 Rademacher 探针的协方差、随机种子和全局 RNG 隔离。
- 多 k 点和 spinless / 非自旋极化 / 共线自旋密度，化学势及自由能。
- 同一探针、同一平方滤波的电子数与密度积分一致性。
- 非线性自洽循环及 Simple / Kerker / Dielectric 混合。
- 无局域势的自由电子 Hamiltonian。
- 小全电子 He 的 PBE、自动谱界、初始密度和完整基 SCF。
- 非收敛返回、错误参数、跨 basis 探针及错误谱界。

最终 SCF 用 `H[ρout]` 的完整对角化独立核对密度、化学势和自由能。参考占据由 DFTK 根据完整本征值计算，参考化学势使用足够宽的独立区间求解，避免小高温体系中固定 ±1 Ha 包围区间不足。

自动原子密度猜测在粗 Fourier 网格上可能出现负值；SCF 将自动猜测投影为非负初始密度。显式用户输入仍校验非负性，输出随机密度不做电子数重标定。

## Chebyshev 主分支

CPU 单线程和四线程，以及 MKL 四线程环境均全部通过，共 158 项检查。主分支只实现 Chebyshev 后端。

| 测试组 | 检查数 |
|---|---:|
| 算符作用及合成投影矩阵 | 14 |
| 探针协方差和可复现性 | 7 |
| 固定 Hamiltonian 密度 / 化学势 / 自由能 | 52 |
| 非线性自洽与混合 | 33 |
| 自由电子模型 | 4 |
| 小全电子 He PBE 集成 | 8 |
| 批量与逐探针对照、批量 SCF | 40 |
| 合计 | 158 |

He：平面波维数 19，Chebyshev `degree=100`，完整基，阻尼 0.5，密度容差 `1e-7`；独立验证中 24 轮收敛，残差 `7.4346884e-8`。最终自由能与输出 Hamiltonian 的完整对角化参考之差为 `7.4374116e-9 Ha`。

`examples/helium_scf.jl` 的 64 个随机相位样本在 21 轮收敛，电子数 `2.000000002468197`，密度残差 `6.0547205e-7`，满足示例的 `1e-6` 容差。探针种子规则已改为物理 k 点和全局探针编号，以保持不同 MPI 分区下的样本一致；固定 seed 的样本与初始串行版本不同。

## MPI 和 CUDA

| 验证 | 检查数 | 结果 |
|---|---:|---|
| CPU MPI，2 / 3 进程 | 每进程 77 | k 点分配、探针分配、SCF 均通过 |
| CUDA 13.4，CPU XC | 29 | 算符、三种采样、密度、非线性 SCF、He/PBE 混合路径通过 |
| CUDA 13.3，原生 GPU Libxc | 27 | 同类检查及原生 GPU He/PBE SCF，通过 |
| GPU 探针 MPI，2 进程 | 每进程 11 | 包括零探针进程、全局密度/能量、SCF，通过 |
| AMDGPU | 可用性检查 | `functional()=false`，缺少 HIP、rocBLAS、rocSOLVER、rocFFT |

MPI 与串行比较的密度、化学势及自由能容差为 `1e-7`，电子数为 `1e-8`。包含 spinless / 共线自旋、Γ 点数少于进程数时 DFTK 的重复 k 点，以及随机样本数少于进程数时的空探针分区。GPU 探针 MPI 经 CPU 暂存规约，测试使用的 MPI 库 `has_cuda()=false`。

CUDA 三种采样的密度与 CPU 的 L² 差约 `1e-16`，自由能差不超过 `5.6e-17 Ha`。非线性随机 SCF 在 13 轮收敛，残差 `8.1066894e-8`，CPU/GPU 自由能差 `2.2e-16 Ha`。

**最初本地 CUDA 13.4 的 Libxc GPU 加载失败。** 直接原因是 `local="true"` 配置触发 Libxc_GPU_jll 的平台选择规则，未选择 GPU artifact；`Libxc.has_cuda()` 为 false，普通 `functionals=PBE()` 的 He GPU SCF 在构建 Hamiltonian 时失败。CUDA 13.4 的警告在包不可用后发出，尚未实际加载 GPU Libxc 来验证其与 CUDA 13.4 的二进制兼容性。另行显式选择 DftFunctionals 内置 PBE 后，交换关联在 CPU 求值，波函数、Hartree、FFT 和密度使用 GPU。这条混合路径的 19 维 He 完整基 SCF 在 24 轮收敛，残差 `7.4346884e-8`，与对应 CPU SCF 自由能差 `1.8e-15 Ha`；CPU 内置 PBE 与 CPU Libxc PBE 也在 `1e-6` 容差内一致。

**清除下载代理后，原生 GPU Libxc PBE 验证通过。** 北大镜像直连完成 CUDA artifact 下载并校验内容哈希；改用 `local="false"`、CUDA runtime 13.3.0 / compiler 13.3.73 后，`Libxc.has_cuda()` 为 true。原生 GPU He/PBE 的 19 维完整基 SCF 在 24 轮收敛，残差 `7.434688411825148e-8`，与 CPU 自由能差 `3.552713678800501e-15 Ha`。

DFTK 自带 CUDA 预编译示例在 GPU 非局域投影算符中混用 CPU 矩阵和 GPU 数组而报错。本次仅禁用预编译演练，实际 JIT、GPU Libxc 和小体系计算正常执行；GPU 赝势没有完成数值验证。没有修改依赖包代码或替换 GPU XC 为 CPU 求值来通过原生 GPU 检查。

配置、可选依赖和启动入口见 [并行说明](parallelization.md)。GPU 的 k 点 MPI 和 k 点 × 探针二维 MPI 尚不支持。

## 小体系计时记录

这些计时只记录已验证路径的运行情况，**速度不是本轮验收标准**，没有据此优化或判断大 FFT 下的 GPU 性能。模型为 `8×8×8 bohr` 晶胞、解析局域势、Γ 点、`Ecut=5 Ha`、257 个平面波、64 个随机探针、`degree=30`；各路径使用同一探针和化学势，不使用完整基。预热后测三次取中位数，CUDA 显式同步设备。

| 路径 | DFTK / FFT / BLAS 线程 | 每批探针 | 秒 |
|---|---|---:|---:|
| CPU 逐探针 OpenBLAS | 1 / 1 / 1 | 1 | 0.248 |
| CPU 分批 OpenBLAS | 1 / 1 / 1 | 32 | 0.167 |
| CPU 分批 OpenBLAS | 4 / 1 / 1 | 32 | 0.071 |
| CPU FFT 多线程 | 1 / 4 / 1 | 32 | 0.188 |
| CPU BLAS 多线程 | 1 / 1 / 4 | 32 | 0.241 |
| CPU 分批 MKL | 4 / 1 / 1 | 32 | 0.069 |
| CUDA 分批 | 1 / 1 / 1 | 64 | 0.312 |

CPU 各批次与逐探针对照的密度和能量差在本次记录中为零；CUDA 密度 L² 差 `3.6e-17`、自由能差 `3.6e-15 Ha`。本例每批 64 列时 Hamiltonian 矩阵调用从 5760 次减少为 90 次，逻辑轨道作用次数不变。小数组存在启动、同步和传输开销；更大 FFT 及多节点吞吐量留待以后测量。所有配置的原始计时见 [CSV](acceleration_benchmark.csv)。

## 运行方式

```sh
JULIA_CUDA_USE_COMPAT=false OPENBLAS_NUM_THREADS=1 \
  julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

测试时设置 `JULIA_CUDA_USE_COMPAT=false` 跳过 CPU 验证不需要的 CUDA 兼容驱动探测。DFTK 的 MPI 库在单进程加载时也会初始化本机通信，本次在允许本机通信的环境中运行。

完整基消除了采样误差，保留了展开误差；随机样本测试另外核对可复现性和自洽收敛。这些测试证明实现与相同离散化下的对角化参考一致，不代表低截断 He 的物理量已经收敛，也不证明大体系可扩展性。
