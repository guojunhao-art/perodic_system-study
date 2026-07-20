# 周期平面波 DFT toy code

这是一个以“看清公式如何落到代码”为目标的周期平面波 Kohn–Sham DFT
教学程序。当前只使用 Gamma 点、无自旋轨道、非自旋极化轨道，并使用
Hartree 原子单位。代码保留了许多生产级程序会封装起来的中间量，方便逐项
检查能量、Hamiltonian、占据数和 Hellmann–Feynman 力。

当前实现包括：

- Gamma-point 平面波基组和 FFT-based $H\psi$；
- LibXC 非自旋极化 LDA exchange + Perdew–Zunger correlation SCF；
- fixed、简并感知零温和 Fermi–Dirac 占据；
- 线性/自适应辅助函数和 Pulay density mixing；
- Gaussian 平滑局域赝势、短程修正和离子–离子能；
- $s$-like、$p$-like Gaussian 非局域 projector；
- 总能量、非局域能和三部分解析离子力；
- 径向 Fourier–Bessel 变换；
- 严格 NC-UPF v2 reader；
- `PP_LOCAL` 的 Gaussian-screened Fourier–Bessel 变换、周期势和解析局域力；
- Gaussian-split point-ion Ewald 能量和解析力；
- 无 projector H UPF 的 H₂ 一维键长优化驱动；
- `upf_info` 文件及局域势检查工具。

> 当前代码已经能从真实 NC-UPF 构造局域势，但默认的 `fft` 示例仍使用原来的
> toy Gaussian 赝势；`PP_BETA.*` 和 `PP_DIJ` 也尚未接入 Hamiltonian。
> 因此，无 projector 的 H UPF 已经可以完成 H₂ SCF 和一维键长优化；Si 等有
> projector 的赝势还不能做完整 SCF。

## 1. 构建与运行

依赖：C++17、Eigen3、FFTW3 和 LibXC（已用 7.0.0 验证）。例如在
Ubuntu/Debian 上安装开发包：

```bash
sudo apt update
sudo apt install g++ make cmake pkg-config libeigen3-dev libfftw3-dev libxc-dev
```

使用 Makefile：

```bash
make -j
./fft
make test
```

或使用 CMake：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### 1.1 下载并检查 NC-UPF

第一阶段建议使用 Quantum ESPRESSO 官方库中的两个文件：

- `H.pz-vbc.UPF`：只有局域势，适合最小 parser 测试；
- `Si.pz-vbc.UPF`：有一个 $s$ projector 和一个 $p$ projector，适合后续
  非局域算符开发。

```bash
mkdir -p pseudopotentials

curl -L \
  https://pseudopotentials.quantum-espresso.org/upf_files/H.pz-vbc.UPF \
  -o pseudopotentials/H.pz-vbc.UPF

curl -L \
  https://pseudopotentials.quantum-espresso.org/upf_files/Si.pz-vbc.UPF \
  -o pseudopotentials/Si.pz-vbc.UPF

make upf_info
./upf_info pseudopotentials/Si.pz-vbc.UPF
./upf_info pseudopotentials/Si.pz-vbc.UPF 0.70
```

`upf_info` 会打印实际读取到的元素、泛函、径向网格、projector 的 $l$ 和
`PP_DIJ`，并同时显示 Ry 与 Ha 单位的矩阵。给出第二个参数时，它还会用指定的
Gaussian 宽度（Bohr）构造局域势，打印短程尾部和若干 $G$ 点的径向变换。

### 1.2 H₂ 局域 UPF 键长优化

`H.pz-vbc.UPF` 没有非局域 projector，因此可以在实现 `PP_BETA` 前作为真实
局域赝势的端到端测试：

```bash
make h2_opt
./h2_opt pseudopotentials/H.pz-vbc.UPF

# 显式指定 cutoff、晶胞边长和 FFT 网格
./h2_opt pseudopotentials/H.pz-vbc.UPF 10.0 12.0 36
```

程序固定 H₂ 质心，只优化一个 H–H 距离。每个几何点都会重新收敛 SCF，随后用
解析局域力与 point-ion Ewald 力构造键向力；最终点还会做一次中心有限差分。
这还不是通用三维几何优化器。

为了和 Quantum ESPRESSO 使用相同 UPF、晶胞和 cutoff 交叉验证，可以运行
[`examples/h2_qe.in`](examples/h2_qe.in)：

```bash
mkdir -p tmp
pw.x -in examples/h2_qe.in > h2_qe.out
```

## 2. 单位和 Fourier 约定

程序内部使用 Hartree 原子单位：

$$
\hbar=m_e=e=4\pi\epsilon_0=1,
\qquad E\ [\mathrm{Ha}],\quad r\ [a_0].
$$

UPF 使用 Rydberg 原子单位，长度仍为 Bohr，但能量满足

$$
1\ \mathrm{Ry}=\frac12\ \mathrm{Ha}.
$$

因此读入后接入 Hamiltonian 时，至少有

$$
V_{\mathrm{loc}}^{\mathrm{Ha}}(r)
=\frac12 V_{\mathrm{loc}}^{\mathrm{Ry}}(r),
\qquad
D_{ij}^{\mathrm{Ha}}=\frac12 D_{ij}^{\mathrm{Ry}}.
$$

周期函数采用 Fourier 级数

$$
f(\mathbf G)=\frac{1}{\Omega}\int_\Omega
f(\mathbf r)e^{-i\mathbf G\cdot\mathbf r}\,d\mathbf r,
\qquad
f(\mathbf r)=\sum_{\mathbf G}f(\mathbf G)
e^{i\mathbf G\cdot\mathbf r}.
$$

平面波归一化为

$$
\langle\mathbf r|\mathbf G\rangle
=\frac{1}{\sqrt\Omega}e^{i\mathbf G\cdot\mathbf r},
\qquad
\psi_n(\mathbf r)=\frac{1}{\sqrt\Omega}
\sum_{\mathbf G}c_{n\mathbf G}e^{i\mathbf G\cdot\mathbf r}.
$$

Gamma 点截断条件是

$$
\frac{|\mathbf G|^2}{2}\le E_{\mathrm{cut}}.
$$

FFTW 的 forward transform 没有 $1/N$ 归一化，所以代码构造 Fourier 系数
时显式除以实空间网格点数；backward transform 则直接实现 Fourier 求和。

## 3. Kohn–Sham 方程和 FFT-based $H\psi$

当前单粒子方程为

$$
\hat H\psi_n
=\left[-\frac12\nabla^2+V_{\mathrm{loc}}(\mathbf r)
+V_H(\mathbf r)+V_{\mathrm{xc}}(\mathbf r)
+\hat V_{\mathrm{NL}}\right]\psi_n
=\varepsilon_n\psi_n.
$$

动能在倒空间是对角的：

$$
(T\psi_n)_{\mathbf G}=\frac{|\mathbf G|^2}{2}c_{n\mathbf G}.
$$

局域势部分通过

$$
c_{n\mathbf G}
\xrightarrow{\mathrm{IFFT}}\psi_n(\mathbf r)
\xrightarrow{\times V_{\mathrm{eff}}(\mathbf r)}
V_{\mathrm{eff}}(\mathbf r)\psi_n(\mathbf r)
\xrightarrow{\mathrm{FFT}}(V_{\mathrm{eff}}\psi_n)_{\mathbf G}
$$

计算，因此不需要显式建立稠密矩阵
$V_{\mathbf G\mathbf G'}=V(\mathbf G-\mathbf G')$。

电子数密度为

$$
n(\mathbf r)=\sum_n f_n|\psi_n(\mathbf r)|^2,
\qquad
\int_\Omega n(\mathbf r)\,d\mathbf r=N_e.
$$

## 4. Hartree 与 LibXC 交换-关联接口

对 $\mathbf G\ne0$，Hartree 势为

$$
V_H(\mathbf G)=\frac{4\pi}{G^2}n(\mathbf G),
\qquad V_H(\mathbf 0)=0,
$$

对应能量

$$
E_H=\frac12\int_\Omega n(\mathbf r)V_H(\mathbf r)\,d\mathbf r.
$$

保留用于解析验证的非自旋极化 LDA exchange 是

$$
E_x[n]= -\frac34\left(\frac3\pi\right)^{1/3}
\int n(\mathbf r)^{4/3}\,d\mathbf r,
$$

$$
V_x(\mathbf r)=\frac{\delta E_x}{\delta n(\mathbf r)}
=-\left(\frac3\pi\right)^{1/3}n(\mathbf r)^{1/3}.
$$

SCF 用 LibXC 统一处理交换和关联。对 LDA，LibXC 返回每粒子能量
$\varepsilon_{\mathrm{xc}}(n)$ 和能量密度的一阶导数，因此网格上的转换是

$$
E_{\mathrm{xc}}
=\int n(\mathbf r)\varepsilon_{\mathrm{xc}}(n)\,d\mathbf r
\simeq \Delta V\sum_p n_p\varepsilon_{\mathrm{xc},p},
$$

$$
V_{\mathrm{xc},p}
=\frac{\partial[n\varepsilon_{\mathrm{xc}}(n)]}{\partial n}
\bigg|_{n=n_p}.
$$

为了匹配 `*.pz-*.UPF`，当前选择：

- `XC_LDA_X`：Slater/Dirac LDA exchange；
- `XC_LDA_C_PZ`：Perdew–Zunger LDA correlation。

初始接口保持 `XC_UNPOLARIZED`。LibXC 要求输入密度非负，因此进入库前只应
清理数值噪声导致的微小负值，而不能用 density floor 改变正常低密度区域。
上述解析 exchange 仍保留为单元测试 oracle；它不再是 SCF 的生产路径。
`SCFOptions::lda_functional` 默认选择 `PerdewZunger`，也可选择
`ExchangeOnly` 做回归比较。LibXC functional 对象在一次 SCF 开始前初始化，
随后跨迭代复用，而不是对每个网格点重复初始化。

LibXC 的 `exc` 是每粒子能量，而 `vrho` 是单位体积能量对密度的一阶导数；
不要把 `exc` 直接当成势。参考
[LibXC C API manual](https://libxc.gitlab.io/manual/libxc-5.1.x/)。

## 5. 总能量和有限温占据

当前零温总能量写成

$$
E_{\mathrm{tot}}=T_s+E_H+E_x+E_c+E_{\mathrm{ext,loc}}
+E_{\mathrm{NL}}+E_{\mathrm{II}}^{\mathrm{smooth}}.
$$

各电子项为

$$
T_s=\sum_n f_n\langle\psi_n|-\tfrac12\nabla^2|\psi_n\rangle,
$$

$$
E_{\mathrm{ext,loc}}=\int n(\mathbf r)V_{\mathrm{loc}}(\mathbf r)\,d\mathbf r.
$$

Fermi–Dirac 占据采用最大占据数 2：

$$
f_n=\frac{2}{\exp[(\varepsilon_n-\mu)/\sigma]+1},
\qquad \sum_n f_n=N_e,
$$

其中 $\sigma=k_BT$。令 $p_n=f_n/2$，无量纲电子熵是

$$
S=-2\sum_n\left[p_n\ln p_n+(1-p_n)\ln(1-p_n)\right].
$$

有限温变分量为 Mermin free energy

$$
F=E-\sigma S,
$$

代码还输出常用的零 smearing 外推估计

$$
E_{\sigma\to0}\approx\frac12(E+F).
$$

## 6. SCF 和 Pulay mixing

一次 SCF 映射可写为

$$
n_{\mathrm{in}}^{(k)}
\xrightarrow{V_H+V_{\mathrm{xc}}}
\hat H^{(k)}
\xrightarrow{\mathrm{diagonalize}}
\{\psi_n,f_n\}
\xrightarrow{}n_{\mathrm{out}}^{(k)}.
$$

密度残差为

$$
R^{(k)}=n_{\mathrm{out}}^{(k)}-n_{\mathrm{in}}^{(k)},
\qquad
\|R^{(k)}\|=\left[\int|R^{(k)}(\mathbf r)|^2d\mathbf r\right]^{1/2}.
$$

线性 trial density 是

$$
n_{\mathrm{trial}}^{(k)}=n_{\mathrm{in}}^{(k)}+\alpha R^{(k)}.
$$

Pulay 系数通过最小化历史残差组合求得：

$$
\min_{\{c_i\}}\left\|\sum_i c_iR^{(i)}\right\|^2,
\qquad \sum_i c_i=1,
$$

随后组合历史 trial densities。混合后代码裁掉微小负密度并重新归一化到
$N_e$。

## 7. 当前 Gaussian 离子模型

第 $I$ 个离子的平滑正电荷 Fourier 系数为

$$
\rho_I(\mathbf G)=\frac{Z_I}{\Omega}
\exp\left(-\frac12\sigma_I^2G^2\right)
e^{-i\mathbf G\cdot\mathbf R_I}.
$$

总离子密度是 $\rho_{\mathrm{ion}}=\sum_I\rho_I$。电子–离子 Coulomb
势为

$$
V_{I,\mathrm{coul}}(\mathbf G)
=-\frac{4\pi}{G^2}\rho_I(\mathbf G),\qquad \mathbf G\ne0.
$$

短程 Gaussian 修正

$$
V_{I,\mathrm{short}}(\mathbf r)
=A_I\exp\left[-\frac{|\mathbf r-\mathbf R_I|^2}{2r_{c,I}^2}\right]
$$

的 Fourier 系数为

$$
V_{I,\mathrm{short}}(\mathbf G)
=\frac{A_I(2\pi)^{3/2}r_{c,I}^3}{\Omega}
e^{-r_{c,I}^2G^2/2}e^{-i\mathbf G\cdot\mathbf R_I}.
$$

原始 toy 模型使用的平滑离子 Coulomb 能是

$$
E_{\mathrm{II}}^{\mathrm{smooth}}
=2\pi\Omega\sum_{\mathbf G\ne0}
\frac{|\rho_{\mathrm{ion}}(\mathbf G)|^2}{G^2}.
$$

其中包含与离子位置无关的 Gaussian self contribution；它不影响力，但在与
其他程序比较绝对总能量时必须统一处理。

真实赝势的离子核应当使用点电荷 Ewald 排斥，不能直接用上述平滑能量做几何
优化。取所有离子共用的辅助 Gaussian 宽度 $\sigma$，point-ion Ewald 能写成

$$
E_{\mathrm{II}}^{\mathrm{point}}
=E_{\mathrm{II}}^{\mathrm{smooth}}
+\frac12\sum_{IJ\mathbf L}'
\frac{Z_IZ_J}{r_{IJ\mathbf L}}
\operatorname{erfc}\left(\frac{r_{IJ\mathbf L}}{2\sigma}\right)
-\sum_I\frac{Z_I^2}{2\sqrt\pi\sigma}
-\frac{2\pi\sigma^2}{\Omega}\left(\sum_I Z_I\right)^2.
$$

四项依次是 Gaussian 电荷的倒空间能、point-minus-Gaussian 实空间修正、
Gaussian self energy 和离子子系统的均匀中和背景项。各项分别依赖 $\sigma$，
总和不依赖。`test_ewald` 同时检查宽度不变量、总力为零和解析力有限差分。

## 8. UPF 径向数据和 Fourier–Bessel 变换

NC-UPF v2 的第一版 reader 读取：

- `PP_HEADER`；
- `PP_R` 和 `PP_RAB`；
- `PP_LOCAL`；
- `PP_BETA.*`；
- `PP_DIJ`。

`PP_R` 是 Bohr 径向网格。常见 Quantum ESPRESSO 对数网格可写成

$$
r_i=\frac{e^{x_i}}{z_{\mathrm{mesh}}},
\qquad x_i=x_{\min}+i\Delta x.
$$

`PP_RAB` 存的是变量代换产生的原始网格因子；对上述网格，

$$
\mathrm{rab}_i=\frac{dr}{dx}\bigg|_{x_i}\Delta x=r_i\Delta x.
$$

它还不是完整 Simpson 权重。网格点数为奇数时，代码通过
`make_upf_simpson_weights` 构造

$$
w_i=\frac{c_i}{3}\,\mathrm{rab}_i,
\qquad c_i=1,4,2,4,\ldots,2,4,1,
$$

于是径向积分离散为

$$
\int f(r)\,dr\simeq\sum_i w_i f(r_i).
$$

球对称函数的三维 Fourier 变换归结为

$$
\widetilde f_l(G)=4\pi\int_0^\infty
r^2j_l(Gr)f_l(r)\,dr,
$$

其中 $j_l$ 是球 Bessel 函数。UPF 的 projector 数组存储的不是
$\beta_l(r)$，而是

$$
u_{\beta,l}(r)=r\beta_l(r).
$$

因此 projector 的径向变换应直接写成

$$
\widetilde\beta_l(G)=4\pi\int_0^\infty
rj_l(Gr)u_{\beta,l}(r)\,dr,
$$

避免在 $r\to0$ 时先除以 $r$。离散形式是

$$
\widetilde\beta_l(G)\simeq4\pi\sum_i
r_i j_l(Gr_i)u_{\beta,l}(r_i)\,w_i.
$$

对于位于 $\mathbf R_I$ 的角动量 projector，平面波矩阵元还包含

$$
\langle\mathbf G|\beta_{I,lm}\rangle
=\frac{4\pi}{\sqrt\Omega}(-i)^lY_{lm}(\widehat{\mathbf G})
\left[\int r^2j_l(Gr)\beta_l(r)\,dr\right]
e^{-i\mathbf G\cdot\mathbf R_I},
$$

具体相位需与所选复球谐或实球谐约定成套测试。

### 8.1 真实局域势为什么可以分解

NC 局域势在大 $r$ 处有 Coulomb tail。UPF 的 Ry 单位下

$$
V_{\mathrm{loc}}^{\mathrm{Ry}}(r)\to-\frac{2Z}{r},
$$

换成 Ha 后为 $-Z/r$。因此不能把整个 $V_{\mathrm{loc}}(r)$ 当普通短程函数
直接做数值 Fourier 积分：$G\to0$ 时它含有 $-4\pi Z/G^2$ 奇点，而且有限
径向网格会粗暴截断长程尾部。

选取任意正的辅助宽度 $\sigma$，定义一个单位 Gaussian 电荷产生的势

$$
\phi_\sigma(r)
=\frac{\operatorname{erf}[r/(\sqrt2\sigma)]}{r},
\qquad
\phi_\sigma(0)=\sqrt{\frac{2}{\pi}}\frac{1}{\sigma}.
$$

代码对同一个函数加一次、减一次：

$$
V_{\mathrm{loc}}^{\mathrm{Ha}}(r)
=-Z\phi_\sigma(r)+\Delta V_\sigma(r),
\qquad
\Delta V_\sigma(r)
=V_{\mathrm{loc}}^{\mathrm{Ha}}(r)+Z\phi_\sigma(r).
$$

这不是近似。它成立的根本原因只是等式右边的 $Z\phi_\sigma$ 精确相消，但它把
一个数值困难的问题拆成了两个容易的问题：

- 在原点，$\phi_\sigma(0)$ 有限，所以 $\Delta V_\sigma(r)$ 没有 $Z/r$ 奇点；
- 在远处，$\operatorname{erf}[r/(\sqrt2\sigma)]\to1$，所以
  $Z\phi_\sigma(r)$ 恰好抵消 $V_{\mathrm{loc}}^{\mathrm{Ha}}(r)\to-Z/r$
  的长程尾部，$\Delta V_\sigma$ 是短程函数；
- Gaussian Coulomb 势的 Fourier 变换有解析式，长程部分不必在有限径向网格上
  数值积分。

于是对 $G>0$ 有

$$
\widetilde V_{\mathrm{loc}}^{\mathrm{Ha}}(G)
=-\frac{4\pi Z}{G^2}e^{-\sigma^2G^2/2}
+4\pi\int_0^\infty r^2j_0(Gr)\Delta V_\sigma(r)\,dr.
$$

辅助宽度 $\sigma$ 可以改变两个右端项各自的数值，却不能改变它们的和。代码的
解析测试会用多个 $\sigma$ 检查这个不变量。实际计算时，$\sigma$ 只需要让
$\Delta V_\sigma$ 在 UPF 径向网格末端已经足够接近零。

对于位于 $\mathbf R_I$ 的离子，程序最终存入 FFT 网格的周期 Fourier 系数为

$$
V_I(\mathbf G)
=\frac{1}{\Omega}\widetilde V_{\mathrm{loc}}^{\mathrm{Ha}}(G)
e^{-i\mathbf G\cdot\mathbf R_I}.
$$

$G=0$ 必须单独约定。代码与平面波赝势程序常用的中性背景约定一致，去掉发散的
Coulomb 零模，只保留有限的 local-potential 常数

$$
\alpha
=4\pi\int_0^\infty r^2
\left[V_{\mathrm{loc}}^{\mathrm{Ha}}(r)+\frac{Z}{r}\right]dr
=4\pi\int_0^\infty r
\left[rV_{\mathrm{loc}}^{\mathrm{Ha}}(r)+Z\right]dr.
$$

这个 $\alpha$ 直接由原始 UPF 势计算，与辅助 $\sigma$ 无关。绝对总能量仍需和
离子–离子能、Gaussian self energy 及其他程序的零点约定一起比较；力不受
常数 $G=0$ 项影响，因为其位置导数含有 $\mathbf G=0$。

### 8.2 H₂ 端到端结果

使用 `H.pz-vbc.UPF`、LibXC `LDA_X + LDA_C_PZ` 和每个几何点完全重新收敛的
SCF，得到如下收敛趋势：

| $E_{\mathrm{cut}}$ (Ha) | cell (Bohr) | FFT | 优化 H–H (Å) |
|---:|---:|---:|---:|
| 2 | 10 | $24^3$ | 0.9727 |
| 4 | 10 | $28^3$ | 0.8386 |
| 6 | 10 | $32^3$ | 0.7984 |
| 6 | 12 | $32^3$ | 0.7963 |
| 10 | 10 | $32^3$ | 0.7947 |
| 10 | 12 | $36^3$ | 0.7887 |

在 12 Bohr、10 Ha 点，解析键向力与中心有限差分相差约
$1.5\times10^{-7}$ Ha/Bohr，说明能量和力实现彼此一致。
[NIST CCCBDB 给出的实验平衡键长](https://cccbdb.nist.gov/exp2x.asp?casno=1333740)
约为 $0.7414$ Å；当前模型结果仍偏长。因此现阶段只能断言“代码内部的能量–力一致”，
还不能断言“赝势结果已经复现实验”。下一项必要验证是用同一个
`H.pz-vbc.UPF` 和相同数值参数运行 Quantum ESPRESSO：若 QE 给出相近结果，
差异主要来自 LDA/赝势和有限尺寸；若 QE 明显更接近实验，则还应继续检查本代码
的 Fourier、$G=0$ 或总能量约定。

## 9. 非局域赝势

Kleinman–Bylander 形式写作

$$
\hat V_{\mathrm{NL}}
=\sum_I\sum_{ij}|\beta_i^I\rangle D_{ij}
\langle\beta_j^I|.
$$

作用在轨道上为

$$
\hat V_{\mathrm{NL}}|\psi_n\rangle
=\sum_I\sum_{ij}|\beta_i^I\rangle D_{ij}
\langle\beta_j^I|\psi_n\rangle,
$$

非局域能为

$$
E_{\mathrm{NL}}
=\sum_n f_n\sum_I\sum_{ij}
\langle\psi_n|\beta_i^I\rangle D_{ij}
\langle\beta_j^I|\psi_n\rangle.
$$

`PP_BETA` 与 `PP_DIJ` 的归一化是联合约定。若进行基变换

$$
|\beta_i'\rangle=s_i|\beta_i\rangle,
$$

为了保持算符不变必须同时变换

$$
D_{ij}'=\frac{D_{ij}}{s_i s_j^*}.
$$

所以真实 UPF projector 不能沿用 toy Gaussian 代码中“逐个归一化 beta、D 不变”
的做法。reader 原样保存 beta 和 $D$，不做隐藏归一化。

## 10. Hellmann–Feynman 离子力

当前解析力分为

$$
\mathbf F_I=\mathbf F_I^{\mathrm{loc}}
+\mathbf F_I^{\mathrm{II}}+\mathbf F_I^{\mathrm{NL}}.
$$

所有平移相关项都含有结构因子

$$
e^{-i\mathbf G\cdot\mathbf R_I},
$$

因此

$$
\frac{\partial}{\partial R_{I,a}}
e^{-i\mathbf G\cdot\mathbf R_I}
=-iG_a e^{-i\mathbf G\cdot\mathbf R_I}.
$$

这就是三类解析力中 $iG_a$ 因子的共同来源。

### 10.1 局域电子–离子力

$$
E_{\mathrm{loc},I}=\Omega\sum_{\mathbf G}
n(\mathbf G)^*V_I(\mathbf G),
$$

$$
F_{I,a}^{\mathrm{loc}}
=\Omega\,\mathrm{Re}\sum_{\mathbf G}
iG_a n(\mathbf G)^*V_I(\mathbf G).
$$

### 10.2 平滑离子–离子力

$$
F_{I,a}^{\mathrm{II}}
=4\pi\Omega\,\mathrm{Re}\sum_{\mathbf G\ne0}
\frac{iG_a\rho_{\mathrm{ion}}(\mathbf G)^*
\rho_I(\mathbf G)}{G^2}.
$$

### 10.3 非局域力

令

$$
b_{i,n}^I=\langle\beta_i^I|\psi_n\rangle.
$$

对 Hermitian $D$，非局域力可写为

$$
F_{I,a}^{\mathrm{NL}}
=-2\,\mathrm{Re}\sum_n f_n\sum_{ij}
(b_{i,n}^I)^*D_{ij}
\frac{\partial b_{j,n}^I}{\partial R_{I,a}}.
$$

projector 的平移相位给出倒空间导数；符号还要和代码中
`beta_G = <G|beta>` 以及复内积的约定一起检查。解析力最终通过每个位移点重新
收敛 SCF 的中心有限差分验证：

$$
F_{I,a}^{\mathrm{FD}}
=-\frac{E(\mathbf R_I+\delta\mathbf e_a)
-E(\mathbf R_I-\delta\mathbf e_a)}{2\delta}.
$$

两离子测试还应满足平移不变性给出的反作用关系

$$
\sum_I\mathbf F_I\approx\mathbf0.
$$

## 11. NC-UPF reader 的支持边界

第一版 reader 主动接受：

- UPF 2.x；
- `pseudo_type="NC"`；
- 稠密 `PP_DIJ`；
- 径向数组短于 mesh 时在尾部补零。

它主动拒绝：

- ultrasoft；
- PAW；
- nonlinear core correction；
- spin–orbit projectors；
- 数组长度、projector 数量或 header 相互矛盾的文件。

“拒绝未实现物理”比静默读入后给出看似合理的错误结果更安全。某些旧文件在
`number_of_proj="0"` 时仍在 `PP_DIJ` 中含有 converter 遗留的 dummy 数字；
reader 以 header 为准并忽略这种 payload。

UPF 格式和字段定义参考
[Quantum ESPRESSO UPF 2.0.1 specification](https://pseudopotentials.quantum-espresso.org/home/unified-pseudopotential-format)。

## 12. 下一步路线

1. 用相同 H₂ 输入与 Quantum ESPRESSO 交叉验证局域 UPF 的能量和键长；
2. 从 `PP_BETA.*` 构造带球谐简并度的倒空间 projector；
3. 用完整 $D_{ij}$ 重构 $H\psi$、$E_{\mathrm{NL}}$ 和非局域力；
4. 把含 projector 的真实 UPF 接入 SCF，并做原子、二原子参考计算。
