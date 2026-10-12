# 并行与设备

Chebyshev 递推按探针列分批调用 DFTK 的 Hamiltonian。`batch_size=32` 是默认值；`batch_size=1` 用于逐探针对照。分批不改变样本数、展开次数或滤波函数。

## CPU 线程与 BLAS

在建立 basis 之前设置 DFTK 线程，避免重复分配给 FFT、BLAS 和探针列：

```julia
using DFTK, stoDFTK
DFTK.setup_threading(n_DFTK=Threads.nthreads(), n_fft=1, n_blas=1)
# 然后建立 basis。
result = stochastic_scf(basis; backend=ChebyshevFOE(degree=100, batch_size=32))
```

用 `julia --threads=4 --project=. ...` 启动。DFTK 在线程间处理一批中的轨道；不要在同一 Hamiltonian 上再并发运行多个递推，因为 DFTK 的 Hamiltonian 使用共享工作缓冲区。FFT/BLAS 线程可以单独设置，是否更快需要按体系测量。MKL 是调用项目的可选依赖，需在加载 DFTK 前 `using MKL`；主包不绑定特定 BLAS 厂商。

可从仓库根目录运行隔离的 MKL 验证：

```sh
julia --project=test/backends -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
STODFTK_USE_MKL=true julia --threads=4 --project=test/backends test/runtests.jl
```

## MPI：k 点或探针

两种分配方式分别使用，暂不组合成 k 点 × 探针二维通信网格。

```julia
using DFTK, MPI, stoDFTK
# 方式一：DFTK 分配 k 点；用于 CPU。
basis = PlaneWaveBasis(model; Ecut, kgrid, comm_kpts=MPI.COMM_WORLD)
result = stochastic_scf(basis; n_samples=64, seed=2026)

# 方式二：每个进程保留完整 basis，stoDFTK 分配全局探针。
basis = PlaneWaveBasis(model; Ecut, kgrid, comm_kpts=MPI.COMM_SELF)
result = stochastic_scf(basis; n_samples=64, seed=2026, probe_comm=MPI.COMM_WORLD)
```

所有进程均调用 SCF，并使用相同模型、参数和种子。第二种方式的 `n_samples=64` 表示**所有进程合计 64 个样本**。探针按全局编号生成，改变进程数不会改变样本。进程可以分到零个探针。

化学势、谱界、熵和轨道能量需要相应的全局规约。密度依赖项只计算一次全局密度的能量。`result.ρ` 是全局密度；在探针 MPI 下，`result.ψ` 和 `occupation` 是本进程的列，单独调用 `DFTK.compute_density` 后仍需对探针通信器求和。

使用 `MPI.mpiexec()` 选择与 MPI.jl 实际加载库匹配的启动器，避免系统 OpenMPI 与 Julia MPICH 混用。验证入口：

```sh
julia --project=. test/run_mpi.jl 2
julia --project=. test/run_mpi.jl 3
```

## NVIDIA CUDA

CUDA 是调用项目的可选依赖，不加入 stoDFTK 主包依赖。先在该项目安装 CUDA，再选 GPU basis：

```julia
using CUDA, DFTK, stoDFTK
CUDA.allowscalar(false)
basis = PlaneWaveBasis(model; Ecut, kgrid,
                       architecture=DFTK.GPU(CuArray), comm_kpts=MPI.COMM_SELF)
result = stochastic_scf(basis; n_samples=64, backend=ChebyshevFOE(batch_size=32))
```

大波函数、递推工作数组和密度保留在设备上；Chebyshev 系数、迹矩和化学势求解在 CPU 上。探针先按可复现种子在 CPU 生成，再传到设备。当前仅支持 Float64 / ComplexF64。

原生 Libxc GPU 泛函需要可用的 Libxc GPU artifact。本次为避免下载超时而设置 `local="true"`，使用系统 CUDA 13.4；当前 Libxc_GPU_jll 的平台选择逻辑在这个设置下直接拒绝所有 GPU artifact，导致 `Libxc.has_cuda()` 为 false，普通 `functionals=PBE()` 的 He GPU SCF 在 DFTK 的可用性断言处失败。LibxcCudaExt 随后因 GPU 包不可用且运行时高于 13.3 发出版本警告；警告本身不是加载后的 ABI 或内核失败。安装的预编译包最高标记 CUDA 13.3，但同一 CUDA 主版本内的选择逻辑允许使用较早的构建。CUDA 13.4 下实际加载并运行 GPU Libxc 尚未验证，不能据此认定其必然不兼容。

可显式使用 DFTK 支持的内置 PBE 泛函：

```julia
using DftFunctionals
functionals = DftFunctional.([:gga_x_pbe, :gga_c_pbe])
model = model_DFT(lattice, atoms, positions; functionals, temperature=0.75)
```

DFTK 将这类泛函的交换关联求值送到 CPU，其他 GPU basis 运算仍在设备上。测试单独标记这条混合路径，并核对 CPU 内置 PBE 与 CPU Libxc PBE 的结果；这不等于原生 GPU Libxc 的验证。

GPU 的 k 点 MPI 暂不支持。探针 MPI 使用复制 GPU basis，并经 CPU 暂存规约密度，不要求 MPI 库支持 CUDA 设备指针。

从仓库根目录准备和运行单 GPU 验证：

```sh
julia --project=test/gpu -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=test/gpu test/gpu/runtests.jl
julia --project=test/gpu test/run_mpi.jl 2 test/gpu/mpi.jl
```

本次下载通过北大 Julia 镜像直连完成。下载子进程清除大写和小写代理变量，其他进程的代理设置不变：

```sh
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    JULIA_PKG_SERVER=https://mirrors.pku.edu.cn/julia \
    julia --project=test/gpu -e 'using Pkg; Pkg.instantiate()'
```

`JULIA_PKG_SERVER` 指定包和 artifact 的镜像服务。Git URL 安装等请求仍可能访问原始站点，见 [Pkg 的下载说明](https://pkgdocs.julialang.org/v1.10/managing-packages/#Pkg-client/server)。本次镜像的注册表接口和 CUDA artifact 均已实测可用；约 988 MiB 的续传在取消代理后约 11 秒完成，完整内容哈希也已核对。

原生 GPU Libxc 的下载和加载检查使用 CUDA 13.3 预编译包，`Libxc.has_cuda()` 为 true。在 **test/gpu/LocalPreferences.toml** 中配置后重启 Julia；此文件不会提交到仓库：

```toml
[CUDA_Runtime_jll]
version = "13.3"
local = "false"
[CUDA_Compiler_jll]
version = "13.3"
local = "false"
[PrecompileTools]
precompile_workloads = false
```

本机 DFTK 0.7.26 / CUDA.jl 6.3.1 的上游 CUDA 预编译示例在非局域投影矩阵的 CPU/GPU 混合乘法处失败。本测试环境禁用预编译演练来绕过该示例，实际 JIT 编译和小体系计算仍执行。GPU 赝势路径尚未验证，当前 GPU 数值测试使用解析势与全电子 He。

也可使用已安装的本地 CUDA；此设置在当前 Libxc_GPU_jll 中会禁用 GPU Libxc artifact：

```toml
[CUDA_Runtime_jll]
local = "true"
[CUDA_Compiler_jll]
local = "true"
```

这是 [CUDA.jl 的本地工具包配置](https://cuda.juliagpu.org/stable/installation/overview/#Using-a-local-CUDA)。默认不设置时，CUDA.jl 下载匹配驱动的工具包。

## AMD 与测量范围

DFTK 的 AMD 路径使用 `DFTK.GPU(ROCArray)` 和 AMDGPU.jl，要求安装 ROCm。本机有 AMD 设备节点，但实际 `AMDGPU.functional()` 为 false，缺少 HIP、rocBLAS、rocSOLVER 和 rocFFT；本次不能验证 AMD 数值结果，也不宣称 AMD 已受支持。

`examples/benchmark_acceleration.jl` 固定 degree、随机探针和化学势，预热后测三次取中位数，不使用完整基。可用 `STODFTK_FFT_THREADS`、`STODFTK_BLAS_THREADS`、`STODFTK_USE_MKL=true` 和 `STODFTK_DEVICE=cuda` 选择后端。CUDA 测量同步设备。小体系结果及环境记录在 [验证记录](validation.md)，不代表大体系性能。
