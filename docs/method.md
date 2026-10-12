# Chebyshev 随机 DFT 与 SCF

DFTK 提供模型、平面波基、动能、Hartree / XC 势、赝势和 FFT。包负责有限温度费米算符展开、随机迹估计、电子数约束和自洽迭代。所有标量能量和温度 `kBT` 使用 Hartree。

## 估计器与电子数一致性

令 `g = DFTK.filled_occupation(model)`，非自旋极化时为 2，spinless / 共线自旋每个通道为 1。每个 k 点计算

```math
f_\mu(H) = \frac{g}{1+\exp((H-\mu)/k_BT)},\qquad
\eta_{ks}=\sqrt{f_\mu(H_k)}\chi_{ks},
```

```math
\rho_\sigma(r)=\sum_{k:\mathrm{spin}(k)=\sigma}
  w_k\,\frac{1}{N_\chi}\sum_s |\eta_{ks}(r)|^2.
```

探针在平面波系数空间满足 `E[χχ†]=I`；随机相位和复 Rademacher 均有单位模长。DFTK 的 `compute_density` 处理 FFT 的体积归一化、k 点权重和密度对称化。

在完整谱包围区间 `[a,b]` 上用 Chebyshev 多项式 `pμ(H)` 逼近 `sqrt(fμ(H))`。化学势必须满足

```math
N_e=\sum_k w_k\,\frac{1}{N_\chi}\sum_s\|p_\mu(H_k)\chi_{ks}\|^2,
```

而不是用另一组样本估计 `Tr fμ(H)`，也不是用独立的 `fμ` 多项式后归一化密度。展开平方采用恒等式 `Ti Tj = (Ti+j + T|i-j|)/2`，因此每个样本只需缓存至 `2degree` 的 Chebyshev 矩。二分求 μ 时复用这些矩，不再调用 Hamiltonian；最终生成波函数时再递推至 `degree`。滤波与电子数一致，但低阶多项式仍有物理近似误差。平方多项式关于 μ 的严格单调性没有保证，数值求解会检查包围和电子数容差，失败时报告错误。

每个 k 点的主要存储为平面波探针矩阵和 `(2degree+1) × n_samples` 的矩；不保存整个 Chebyshev 向量序列。每次密度构造总计约 `3degree × n_samples` 次逐向量 Hamiltonian 应用。完整基将 `n_samples` 替换为平面波维数，因此只用于小体系。

## 谱界

标准 DFTK Hamiltonian 可写为动能对角算符加局域势和非局域赝势 `P D P†`。利用前两项的极值和后一项的范数得到保守包围区间。非零赝势本征值可从投影子空间矩阵 `sqrt(P†P) D sqrt(P†P)` 得到；该矩阵通常远小于平面波 Hamiltonian。自动谱界不使用极端 Ritz 值来冒充保证的包围区间。特殊矩阵自由算符需显式提供谱界。

## 能量与熵

动能和非局域能量用过滤波函数及其样本权重传给 DFTK；局域势、Hartree 和 XC 项用生成的密度计算。熵项独立估计

```math
-TS = k_BT\,g\,\mathrm{Tr}[n\log n+(1-n)\log(1-n)],
\qquad n=f_\mu/g.
```

它使用同一 Hamiltonian 和样本的 Chebyshev 矩，但直接展开熵函数，阶数为 `2degree`。`energies.total` 是有限温度自由能；`internal_energy` 去掉熵项。熵与有限阶平方滤波的密度矩阵之间存在展开误差，应分别做阶数收敛验证。

## 自洽迭代

初始密度可由 DFTK 原子猜测得到；无原子的模型采用均匀密度。每轮更新 Hamiltonian、缓存矩、求 μ、生成密度，检查未混合残差，再执行 DFTK 密度预条件和阻尼混合。仅混合后的下一轮输入投影为非负、给定电荷的密度；输出不做电荷重标定。

随机探针在整个 SCF 过程中固定。对采样映射的收敛须与随机样本数、种子和展开阶数收敛区分。结束时返回输出密度的 Hamiltonian 和最后一次滤波的输入 Hamiltonian，便于核查残差而不隐藏有限步差异。当前模型验证会拒绝未实现的 orbital-dependent 项与需要动能密度的 meta-GGA。
