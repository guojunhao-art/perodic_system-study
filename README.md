# 周期平面波 DFT toy code

这是一个以“看清公式如何落到代码”为目标的周期平面波 Kohn–Sham DFT
教学程序。当前只使用 Gamma 点、无自旋轨道、非自旋极化轨道，并使用
Hartree 原子单位。代码保留了许多生产级程序会封装起来的中间量，方便逐项
检查能量、Hamiltonian、占据数和 Hellmann–Feynman 力。

当前实现包括：

- Gamma-point 平面波基组和 FFT-based \(H\psi\)；
- LibXC 非自旋极化 LDA exchange + Perdew–Zunger correlation SCF；
- fixed、简并感知零温和 Fermi–Dirac 占据；
- 线性/自适应辅助函数和 Pulay density mixing；
- Gaussian 平滑局域赝势、短程修正和离子–离子能；
- \(s\)-like、\(p\)-like Gaussian 非局域 projector；
- 总能量、非局域能和三部分解析离子力；
- 径向 Fourier–Bessel 变换；
- 第一版严格 NC-UPF v2 reader 和 `upf_info` 检查工具。

> 当前 UPF reader 只读取和检查数据，尚未用真实 UPF 代替 SCF 中的 Gaussian
> 赝势。这个边界是有意保留的：先单独验证文件、单位和径向数组，再改变
> Hamiltonian。

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
- `Si.pz-vbc.UPF`：有一个 \(s\) projector 和一个 \(p\) projector，适合后续
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
```

`upf_info` 会打印实际读取到的元素、泛函、径向网格、projector 的 \(l\) 和
`PP_DIJ`，并同时显示 Ry 与 Ha 单位的矩阵。

## 2. 单位和 Fourier 约定

程序内部使用 Hartree 原子单位：

\[
\hbar=m_e=e=4\pi\epsilon_0=1,
\qquad E\ [\mathrm{Ha}],\quad r\ [a_0].
\]

UPF 使用 Rydberg 原子单位，长度仍为 Bohr，但能量满足

\[
1\ \mathrm{Ry}=\frac12\ \mathrm{Ha}.
\]

因此读入后接入 Hamiltonian 时，至少有

\[
V_{\mathrm{loc}}^{\mathrm{Ha}}(r)
=\frac12 V_{\mathrm{loc}}^{\mathrm{Ry}}(r),
\qquad
D_{ij}^{\mathrm{Ha}}=\frac12 D_{ij}^{\mathrm{Ry}}.
\]

周期函数采用 Fourier 级数

\[
f(\mathbf G)=\frac{1}{\Omega}\int_\Omega
f(\mathbf r)e^{-i\mathbf G\cdot\mathbf r}\,d\mathbf r,
\qquad
f(\mathbf r)=\sum_{\mathbf G}f(\mathbf G)
e^{i\mathbf G\cdot\mathbf r}.
\]

平面波归一化为

\[
\langle\mathbf r|\mathbf G\rangle
=\frac{1}{\sqrt\Omega}e^{i\mathbf G\cdot\mathbf r},
\qquad
\psi_n(\mathbf r)=\frac{1}{\sqrt\Omega}
\sum_{\mathbf G}c_{n\mathbf G}e^{i\mathbf G\cdot\mathbf r}.
\]

Gamma 点截断条件是

\[
\frac{|\mathbf G|^2}{2}\le E_{\mathrm{cut}}.
\]

FFTW 的 forward transform 没有 \(1/N\) 归一化，所以代码构造 Fourier 系数
时显式除以实空间网格点数；backward transform 则直接实现 Fourier 求和。

## 3. Kohn–Sham 方程和 FFT-based \(H\psi\)

当前单粒子方程为

\[
\hat H\psi_n
=\left[-\frac12\nabla^2+V_{\mathrm{loc}}(\mathbf r)
+V_H(\mathbf r)+V_{\mathrm{xc}}(\mathbf r)
+\hat V_{\mathrm{NL}}\right]\psi_n
=\varepsilon_n\psi_n.
\]

动能在倒空间是对角的：

\[
(T\psi_n)_{\mathbf G}=\frac{|\mathbf G|^2}{2}c_{n\mathbf G}.
\]

局域势部分通过

\[
c_{n\mathbf G}
\xrightarrow{\mathrm{IFFT}}\psi_n(\mathbf r)
\xrightarrow{\times V_{\mathrm{eff}}(\mathbf r)}
V_{\mathrm{eff}}(\mathbf r)\psi_n(\mathbf r)
\xrightarrow{\mathrm{FFT}}(V_{\mathrm{eff}}\psi_n)_{\mathbf G}
\]

计算，因此不需要显式建立稠密矩阵
\(V_{\mathbf G\mathbf G'}=V(\mathbf G-\mathbf G')\)。

电子数密度为

\[
n(\mathbf r)=\sum_n f_n|\psi_n(\mathbf r)|^2,
\qquad
\int_\Omega n(\mathbf r)\,d\mathbf r=N_e.
\]

## 4. Hartree 与 LibXC 交换-关联接口

对 \(\mathbf G\ne0\)，Hartree 势为

\[
V_H(\mathbf G)=\frac{4\pi}{G^2}n(\mathbf G),
\qquad V_H(\mathbf 0)=0,
\]

对应能量

\[
E_H=\frac12\int_\Omega n(\mathbf r)V_H(\mathbf r)\,d\mathbf r.
\]

保留用于解析验证的非自旋极化 LDA exchange 是

\[
E_x[n]= -\frac34\left(\frac3\pi\right)^{1/3}
\int n(\mathbf r)^{4/3}\,d\mathbf r,
\]

\[
V_x(\mathbf r)=\frac{\delta E_x}{\delta n(\mathbf r)}
=-\left(\frac3\pi\right)^{1/3}n(\mathbf r)^{1/3}.
\]

SCF 用 LibXC 统一处理交换和关联。对 LDA，LibXC 返回每粒子能量
\(\varepsilon_{\mathrm{xc}}(n)\) 和能量密度的一阶导数，因此网格上的转换是

\[
E_{\mathrm{xc}}
=\int n(\mathbf r)\varepsilon_{\mathrm{xc}}(n)\,d\mathbf r
\simeq \Delta V\sum_p n_p\varepsilon_{\mathrm{xc},p},
\]

\[
V_{\mathrm{xc},p}
=\frac{\partial[n\varepsilon_{\mathrm{xc}}(n)]}{\partial n}
\bigg|_{n=n_p}.
\]

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

\[
E_{\mathrm{tot}}=T_s+E_H+E_x+E_c+E_{\mathrm{ext,loc}}
+E_{\mathrm{NL}}+E_{\mathrm{II}}^{\mathrm{smooth}}.
\]

各电子项为

\[
T_s=\sum_n f_n\langle\psi_n|-\tfrac12\nabla^2|\psi_n\rangle,
\]

\[
E_{\mathrm{ext,loc}}=\int n(\mathbf r)V_{\mathrm{loc}}(\mathbf r)\,d\mathbf r.
\]

Fermi–Dirac 占据采用最大占据数 2：

\[
f_n=\frac{2}{\exp[(\varepsilon_n-\mu)/\sigma]+1},
\qquad \sum_n f_n=N_e,
\]

其中 \(\sigma=k_BT\)。令 \(p_n=f_n/2\)，无量纲电子熵是

\[
S=-2\sum_n\left[p_n\ln p_n+(1-p_n)\ln(1-p_n)\right].
\]

有限温变分量为 Mermin free energy

\[
F=E-\sigma S,
\]

代码还输出常用的零 smearing 外推估计

\[
E_{\sigma\to0}\approx\frac12(E+F).
\]

## 6. SCF 和 Pulay mixing

一次 SCF 映射可写为

\[
n_{\mathrm{in}}^{(k)}
\xrightarrow{V_H+V_{\mathrm{xc}}}
\hat H^{(k)}
\xrightarrow{\mathrm{diagonalize}}
\{\psi_n,f_n\}
\xrightarrow{}n_{\mathrm{out}}^{(k)}.
\]

密度残差为

\[
R^{(k)}=n_{\mathrm{out}}^{(k)}-n_{\mathrm{in}}^{(k)},
\qquad
\|R^{(k)}\|=\left[\int|R^{(k)}(\mathbf r)|^2d\mathbf r\right]^{1/2}.
\]

线性 trial density 是

\[
n_{\mathrm{trial}}^{(k)}=n_{\mathrm{in}}^{(k)}+\alpha R^{(k)}.
\]

Pulay 系数通过最小化历史残差组合求得：

\[
\min_{\{c_i\}}\left\|\sum_i c_iR^{(i)}\right\|^2,
\qquad \sum_i c_i=1,
\]

随后组合历史 trial densities。混合后代码裁掉微小负密度并重新归一化到
\(N_e\)。

## 7. 当前 Gaussian 离子模型

第 \(I\) 个离子的平滑正电荷 Fourier 系数为

\[
\rho_I(\mathbf G)=\frac{Z_I}{\Omega}
\exp\left(-\frac12\sigma_I^2G^2\right)
e^{-i\mathbf G\cdot\mathbf R_I}.
\]

总离子密度是 \(\rho_{\mathrm{ion}}=\sum_I\rho_I\)。电子–离子 Coulomb
势为

\[
V_{I,\mathrm{coul}}(\mathbf G)
=-\frac{4\pi}{G^2}\rho_I(\mathbf G),\qquad \mathbf G\ne0.
\]

短程 Gaussian 修正

\[
V_{I,\mathrm{short}}(\mathbf r)
=A_I\exp\left[-\frac{|\mathbf r-\mathbf R_I|^2}{2r_{c,I}^2}\right]
\]

的 Fourier 系数为

\[
V_{I,\mathrm{short}}(\mathbf G)
=\frac{A_I(2\pi)^{3/2}r_{c,I}^3}{\Omega}
e^{-r_{c,I}^2G^2/2}e^{-i\mathbf G\cdot\mathbf R_I}.
\]

代码当前使用的平滑离子 Coulomb 能是

\[
E_{\mathrm{II}}^{\mathrm{smooth}}
=2\pi\Omega\sum_{\mathbf G\ne0}
\frac{|\rho_{\mathrm{ion}}(\mathbf G)|^2}{G^2}.
\]

其中包含与离子位置无关的 Gaussian self contribution；它不影响力，但在与
其他程序比较绝对总能量时必须统一处理。

## 8. UPF 径向数据和 Fourier–Bessel 变换

NC-UPF v2 的第一版 reader 读取：

- `PP_HEADER`；
- `PP_R` 和 `PP_RAB`；
- `PP_LOCAL`；
- `PP_BETA.*`；
- `PP_DIJ`。

`PP_R` 是 Bohr 径向网格。常见 Quantum ESPRESSO 对数网格可写成

\[
r_i=\frac{e^{x_i}}{z_{\mathrm{mesh}}},
\qquad x_i=x_{\min}+i\Delta x.
\]

`PP_RAB` 存的是变量代换产生的原始网格因子；对上述网格，

\[
\mathrm{rab}_i=\frac{dr}{dx}\bigg|_{x_i}\Delta x=r_i\Delta x.
\]

它还不是完整 Simpson 权重。网格点数为奇数时，代码通过
`make_upf_simpson_weights` 构造

\[
w_i=\frac{c_i}{3}\,\mathrm{rab}_i,
\qquad c_i=1,4,2,4,\ldots,2,4,1,
\]

于是径向积分离散为

\[
\int f(r)\,dr\simeq\sum_i w_i f(r_i).
\]

球对称函数的三维 Fourier 变换归结为

\[
\widetilde f_l(G)=4\pi\int_0^\infty
r^2j_l(Gr)f_l(r)\,dr,
\]

其中 \(j_l\) 是球 Bessel 函数。UPF 的 projector 数组存储的不是
\(\beta_l(r)\)，而是

\[
u_{\beta,l}(r)=r\beta_l(r).
\]

因此 projector 的径向变换应直接写成

\[
\widetilde\beta_l(G)=4\pi\int_0^\infty
rj_l(Gr)u_{\beta,l}(r)\,dr,
\]

避免在 \(r\to0\) 时先除以 \(r\)。离散形式是

\[
\widetilde\beta_l(G)\simeq4\pi\sum_i
r_i j_l(Gr_i)u_{\beta,l}(r_i)\,w_i.
\]

对于位于 \(\mathbf R_I\) 的角动量 projector，平面波矩阵元还包含

\[
\langle\mathbf G|\beta_{I,lm}\rangle
=\frac{4\pi}{\sqrt\Omega}(-i)^lY_{lm}(\widehat{\mathbf G})
\left[\int r^2j_l(Gr)\beta_l(r)\,dr\right]
e^{-i\mathbf G\cdot\mathbf R_I},
\]

具体相位需与所选复球谐或实球谐约定成套测试。

### 8.1 局域势的 Coulomb tail

NC 局域势在大 \(r\) 处有 Coulomb tail。UPF 的 Ry 单位下

\[
V_{\mathrm{loc}}^{\mathrm{Ry}}(r)\to-\frac{2Z}{r},
\]

换成 Ha 后为 \(-Z/r\)。因此不能把整个 \(V_{\mathrm{loc}}(r)\) 在
\(G=0\) 处当普通短程函数直接积分。后续接入 Hamiltonian 时应先分解

\[
V_{\mathrm{SR}}(r)=V_{\mathrm{loc}}^{\mathrm{Ha}}(r)+\frac{Z}{r},
\]

再对 \(G\ne0\) 写成

\[
\widetilde V_{\mathrm{loc}}^{\mathrm{Ha}}(G)
=-\frac{4\pi Z}{G^2}
+4\pi\int r^2j_0(Gr)V_{\mathrm{SR}}(r)\,dr.
\]

周期 Fourier 系数还要除以晶胞体积 \(\Omega\)。\(G=0\) 项必须和中性背景、
离子–离子 Ewald/smooth-Coulomb 约定统一，不能由 parser 自行猜测。

## 9. 非局域赝势

Kleinman–Bylander 形式写作

\[
\hat V_{\mathrm{NL}}
=\sum_I\sum_{ij}|\beta_i^I\rangle D_{ij}
\langle\beta_j^I|.
\]

作用在轨道上为

\[
\hat V_{\mathrm{NL}}|\psi_n\rangle
=\sum_I\sum_{ij}|\beta_i^I\rangle D_{ij}
\langle\beta_j^I|\psi_n\rangle,
\]

非局域能为

\[
E_{\mathrm{NL}}
=\sum_n f_n\sum_I\sum_{ij}
\langle\psi_n|\beta_i^I\rangle D_{ij}
\langle\beta_j^I|\psi_n\rangle.
\]

`PP_BETA` 与 `PP_DIJ` 的归一化是联合约定。若进行基变换

\[
|\beta_i'\rangle=s_i|\beta_i\rangle,
\]

为了保持算符不变必须同时变换

\[
D_{ij}'=\frac{D_{ij}}{s_i s_j^*}.
\]

所以真实 UPF projector 不能沿用 toy Gaussian 代码中“逐个归一化 beta、D 不变”
的做法。reader 原样保存 beta 和 \(D\)，不做隐藏归一化。

## 10. Hellmann–Feynman 离子力

当前解析力分为

\[
\mathbf F_I=\mathbf F_I^{\mathrm{loc}}
+\mathbf F_I^{\mathrm{II}}+\mathbf F_I^{\mathrm{NL}}.
\]

所有平移相关项都含有结构因子

\[
e^{-i\mathbf G\cdot\mathbf R_I},
\]

因此

\[
\frac{\partial}{\partial R_{I,a}}
e^{-i\mathbf G\cdot\mathbf R_I}
=-iG_a e^{-i\mathbf G\cdot\mathbf R_I}.
\]

这就是三类解析力中 \(iG_a\) 因子的共同来源。

### 10.1 局域电子–离子力

\[
E_{\mathrm{loc},I}=\Omega\sum_{\mathbf G}
n(\mathbf G)^*V_I(\mathbf G),
\]

\[
F_{I,a}^{\mathrm{loc}}
=\Omega\,\mathrm{Re}\sum_{\mathbf G}
iG_a n(\mathbf G)^*V_I(\mathbf G).
\]

### 10.2 平滑离子–离子力

\[
F_{I,a}^{\mathrm{II}}
=4\pi\Omega\,\mathrm{Re}\sum_{\mathbf G\ne0}
\frac{iG_a\rho_{\mathrm{ion}}(\mathbf G)^*
\rho_I(\mathbf G)}{G^2}.
\]

### 10.3 非局域力

令

\[
b_{i,n}^I=\langle\beta_i^I|\psi_n\rangle.
\]

对 Hermitian \(D\)，非局域力可写为

\[
F_{I,a}^{\mathrm{NL}}
=-2\,\mathrm{Re}\sum_n f_n\sum_{ij}
(b_{i,n}^I)^*D_{ij}
\frac{\partial b_{j,n}^I}{\partial R_{I,a}}.
\]

projector 的平移相位给出倒空间导数；符号还要和代码中
`beta_G = <G|beta>` 以及复内积的约定一起检查。解析力最终通过每个位移点重新
收敛 SCF 的中心有限差分验证：

\[
F_{I,a}^{\mathrm{FD}}
=-\frac{E(\mathbf R_I+\delta\mathbf e_a)
-E(\mathbf R_I-\delta\mathbf e_a)}{2\delta}.
\]

两离子测试还应满足平移不变性给出的反作用关系

\[
\sum_I\mathbf F_I\approx\mathbf0.
\]

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

1. 从 `PP_LOCAL` 构造去 Coulomb-tail 的径向变换，并明确 \(G=0\) 约定；
2. 从 `PP_BETA.*` 构造带球谐简并度的倒空间 projector；
3. 用完整 \(D_{ij}\) 重构 \(H\psi\)、\(E_{\mathrm{NL}}\) 和非局域力；
4. 用原子、二原子和 Quantum ESPRESSO 参考计算逐层验证。
