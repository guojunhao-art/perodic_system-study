# 周期平面波 DFT 学习程序

这是一个以“看清公式如何落到代码”为目标的周期平面波 Kohn–Sham DFT
教学程序。当前支持完整均匀多 k 点网格、非磁性和共线自旋极化计算（不含
spin--orbit coupling），并使用 Hartree 原子单位。代码保留了许多生产级程序
会封装起来的中间量，方便逐项
检查能量、Hamiltonian、占据数和 Hellmann–Feynman 力。

当前实现包括：

- Bloch 多 k 点平面波基组、FFT-based $H\psi$ 和可选的 k 点级 MPI 并行；
- Gamma-centered、Monkhorst–Pack 和显式带权 k 点输入；
- 所有 k 点和自旋通道共享化学势的零温/Fermi–Dirac 占据及带权密度、能量和力；
- LibXC 非磁性/共线自旋极化 LDA exchange + Perdew–Zunger correlation SCF；
- fixed、简并感知零温和 Fermi–Dirac 占据；
- 随外层密度残差自动收紧、并在退出前严格精修的 Davidson 容差；
- 线性/自适应辅助函数和 Pulay density mixing；
- Gaussian 平滑局域赝势、短程修正和离子–离子能；
- 保留用于解析回归测试的 $s$-like、$p$-like Gaussian projector；
- 真实 NC-UPF `PP_BETA.* + PP_DIJ` 非局域算符、能量和解析力；
- 总能量和局域、point-ion Ewald、非局域三部分解析离子力；
- 径向 Fourier–Bessel 变换；
- 严格 NC-UPF v2 reader；
- `PP_LOCAL` 的 Gaussian-screened Fourier–Bessel 变换、周期势和解析局域力；
- Gaussian-split point-ion Ewald 能量和解析力；
- 无 projector H UPF 的 H₂ 一维键长优化驱动；
- 真实 Si UPF 的非对称 Si₂ 分项/全 SCF 力验证驱动；
- POSCAR 结构解析、独立计算参数文件和通用多 k 点单点驱动；
- 固定晶胞 BFGS 离子弛豫、`Selective dynamics`、SCF 热启动和结构轨迹输出；
- `upf_info` 文件及局域势检查工具。

`pwdft` 是唯一的主 SCF 程序：它读取计算配置和 POSCAR，并从 NC-UPF 文件构造
局域势、projector、$D_{ij}$、价电子数和离子电荷。早期硬编码单原子的 `fft`
入口和未使用的 toy ionic potential 已删除。Gaussian 解析模型只保留在底层
有限差分回归中，用来独立检查能量和力的符号与归一化。

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
./pwdft examples/si_scf.in
make test
```

或使用 CMake：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

MPI 是可选依赖。安装 OpenMPI 后，可以让不同进程负责不同 k 点：

```bash
sudo apt install openmpi-bin libopenmpi-dev

# CMake：保留独立的 MPI 构建目录
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release -DPWDFT_ENABLE_MPI=ON
cmake --build build-mpi -j
ctest --test-dir build-mpi --output-on-failure
mpiexec -n 8 build-mpi/pwdft examples/si_kpoints_scf.in

# Makefile：生成独立的 pwdft_mpi，不覆盖串行程序
make -j pwdft_mpi
mpiexec -n 8 ./pwdft_mpi examples/si_kpoints_scf.in
make test-mpi
```

k 点采用循环分配：rank $r$ 处理 $r,r+N_{\mathrm{rank}},\ldots$。每轮 SCF
只全局汇总本征值、带权密度、动能和非局域能；轨道矩阵始终留在所属 rank，最终
非局域力也先局部计算再求和。有效进程数不超过 k 点数；Gamma-only 计算不会从
这一层并行中获得加速。未启用 MPI 时，同一套代码自动退化为原来的单进程路径。

单个 k 点内部可通过 threaded FFTW 和 OpenMP 并行。输入中省略 `fft_threads`
时保留单线程默认；写成

```text
fft_threads = auto
```

会读取 OpenMP 可用线程数，因而可以用 `OMP_NUM_THREADS` 控制：

```bash
OMP_NUM_THREADS=8 ./pwdft examples/o2_triplet_scf.in
```

也可以直接写 `fft_threads = 8`。该线程数同时用于 FFTW plan、Eigen 的线程上限、
平面波 scatter/gather、实空间势乘法、密度累加和 Hartree 网格循环。MPI 构建中每个
rank 各自使用该线程数，因此混合并行时应按每个 rank 分配的 CPU 核数设置
`OMP_NUM_THREADS`，避免过度订阅。CMake 可用
`PWDFT_ENABLE_FFTW_THREADS=OFF` 或 `PWDFT_ENABLE_OPENMP=OFF` 分别关闭两层线程
支持。

Davidson 的长复矩阵乘法可选择交给外部 BLAS。默认仍使用 Eigen 自带内核；例如
使用系统 OpenBLAS：

```bash
cmake -S . -B build-blas \
  -DCMAKE_BUILD_TYPE=Release \
  -DPWDFT_ENABLE_BLAS=ON \
  -DPWDFT_BLAS_VENDOR=OpenBLAS
```

若 Intel MKL 与 Intel MPI 安装在同一套 oneAPI 环境中，不需要执行会同时修改
MPI 环境的 `setvars.sh`。可以只把 MKL 根目录交给本项目：

```bash
cmake -S . -B build-mkl \
  -DCMAKE_BUILD_TYPE=Release \
  -DPWDFT_ENABLE_BLAS=ON \
  -DPWDFT_BLAS_ROOT=/opt/intel/oneapi/mkl/latest
```

也可以精确指定单个 BLAS runtime；该选项优先级最高：

```bash
cmake -S . -B build-mkl \
  -DCMAKE_BUILD_TYPE=Release \
  -DPWDFT_ENABLE_BLAS=ON \
  -DPWDFT_BLAS_LIBRARY=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so
```

这两个路径选项只用于定位 BLAS，不会写入 `CMAKE_PREFIX_PATH`，也不会改变
`MPI_CXX_COMPILER`。因此 MPI 构建仍可显式使用原来的编译器，例如
`-DMPI_CXX_COMPILER=/usr/bin/mpicxx`。GCC OpenMP 与 MKL runtime 混用时建议运行
前设置：

```bash
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export MKL_DYNAMIC=FALSE
export MKL_THREADING_LAYER=GNU
```

若使用 OpenBLAS，则以 `OPENBLAS_NUM_THREADS=10` 控制 BLAS 线程。外部 BLAS
只接管 Eigen 支持的动态稠密乘法；矩阵对象、小型本征值求解和其余 C++ 接口仍由
Eigen 管理。

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

主程序优先采用输入文件中的 `ecut_ha`；若设为 `auto`，则读取所有 UPF 的
`wfc_cutoff` 并取最大值（从 Ry 换算为 Ha），旧赝势没有有效建议值时默认使用
10 Ha。当前固定使用 LibXC PZ-LDA，并主动拒绝泛函标签中不含 `PZ` 的赝势，避免
电子泛函与生成赝势所用泛函静默混用。

### 1.2 POSCAR 与通用单点输入

`pwdft` 将结构和数值参数分成两个文件。POSCAR 负责晶胞、元素、坐标及可选的
`Selective dynamics` 标记；`key = value` 计算文件负责赝势映射、cutoff、占据和
收敛参数。FFT 网格默认自动生成，也可显式覆盖：

```bash
make pwdft
./pwdft examples/si_scf.in
./pwdft examples/si_kpoints_scf.in
```

示例 [`examples/POSCAR_Si`](examples/POSCAR_Si) 与
[`examples/si_scf.in`](examples/si_scf.in) 给出完整的 Gamma 点计算。配置文件中的
相对路径以配置文件所在目录为基准，而不是以启动程序时的工作目录为基准。多元素
体系为每种元素重复一行 `pseudo`：

```text
structure = POSCAR
pseudo = Si pseudopotentials/Si.pz-vbc.UPF
pseudo = H  pseudopotentials/H.pz-vbc.UPF
ecut_ha = 10.0
fft_grid = auto
fft_threads = auto
kpoints = gamma
```

省略 `fft_grid` 与写成 `fft_grid = auto` 等价。程序先为所有实际 k 点生成
截断平面波基组，再统计波函数频率及其两两频率差所需的最大整数频率。每个方向
采用严格避开 Nyquist 边界的最小尺寸，并向上舍入到仅含 2、3、5 质因子的偶数，
以兼顾无混叠和 FFT 效率。例如 12 Å 立方晶胞中的 10 Ha H 原子自动选择
$72^3$。仍可用 `fft_grid = 72 72 72` 显式固定网格；若尺寸不足，程序会报错。

外层 SCF 使用与 VASP `EDIFF` 类似的双能量判据：

$$
|dE|<\mathrm{EDIFF},\qquad |d\varepsilon|<\mathrm{EDIFF},
$$

其中 $dE$ 是变分总能变化，$d\varepsilon$ 是占据加权带能变化；二者共用
`energy_tolerance_ha`，默认值为 $10^{-6}$ Ha。`rms(c)` 只保留为密度残差诊断，
不再参与 SCF 退出判断。

Davidson 内层求解默认从较松的阈值开始，并随 `rms(c)` 逐渐收紧：

```text
density_tolerance = 1.0e-7
energy_tolerance_ha = 1.0e-6
eigensolver_initial_tolerance_ha = 1.0e-7
eigensolver_tolerance_ha = 2.0e-10
```

这里 `density_tolerance` 只是调节 Davidson 精度的参考尺度，不是外层收敛限；
`eigensolver_tolerance_ha` 始终是最终精度。即使 $dE$ 和 $d\varepsilon$ 已满足
外层条件，程序也会用该阈值再完成一次 Davidson 精修后才退出。把 initial 与
final 设成相同数值即可恢复固定容差。

POSCAR 按标准 VASP 单位解释：晶格和 Cartesian 坐标为 Å，读入后自动转换为 Bohr；
`Direct` 坐标不做长度转换。当前支持一个正缩放因子、负的目标体积缩放、VASP 5
元素行、`Direct/Cartesian` 和 `Selective dynamics`。三分量各向异性缩放和缺少
元素行的 VASP 4 格式会明确报错。选择性移动标记现在会被保存在结构对象中，供后续
几何优化使用，但单点 SCF 不使用它们。

单 Gamma 点写作：

```text
kpoints = gamma
```

完整均匀网格可以选择 Gamma-centered 或标准 Monkhorst–Pack 位移：

```text
kpoints = gamma 4 4 4
kpoints = monkhorst_pack 4 4 4
```

也可逐点给出倒格基矢下的分数坐标和正权重；程序会把权重统一归一化：

```text
kpoints = explicit
kpoint = 0.0 0.0 0.0  1
kpoint = 0.5 0.0 0.0  3
```

当前使用完整网格，不进行空间群或时间反演约化。因此均匀网格的每个点权重均为
$1/N_k$；显式输入则保留用户给定的相对权重。所有 k 点共享同一个全局费米能级，
不会分别填充电子。解析器和多 k 点回归可单独运行：

```bash
make test_input
./test_input
make test_kpoints
./test_kpoints
```

### 1.3 H₂ 局域 UPF 键长优化

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

### 1.4 Si₂ 真实 UPF 力验证

单原子高对称零力不能检查力的整体符号、projector 平移相位或离子编号。
`si2_force_check` 因此使用一个键轴不与笛卡尔方向重合的非对称 Si₂ 构型，依次验证：

1. 固定中心 SCF 密度，只移动局域势，比较解析 $F_{\mathrm{loc}}$ 与能量差分；
2. 对 point-ion Ewald 能量做差分，比较 $F_{\mathrm{II}}$；
3. 固定轨道和占据，只移动非局域 projector，比较 $F_{\mathrm{NL}}$；
4. 对每个位移点重新收敛 SCF，用 Mermin 自由能 `TOTEN` 验证总力；
5. 检查平移不变性要求的 $\max|\sum_I\mathbf F_I|$。

```bash
make si2_force_check
./si2_force_check pseudopotentials/Si.pz-vbc.UPF

# PSEUDO ECUT_HA CELL_BOHR FFT_N SMEARING_EV MIN_SCF_FD_STEP_BOHR
./si2_force_check pseudopotentials/Si.pz-vbc.UPF 10.0 16.0 48 0.05 0.002
```

冻结态三项使用 $10^{-5}$ Bohr 的位移，不受 SCF 噪声限制。全 SCF 检查默认依次使用
$10^{-2}$、$5\times10^{-3}$ 和 $2\times10^{-3}$ Bohr；最后一个命令行参数改变最小
步长，另外两个步长相应取其 5 倍和 2.5 倍。程序检查两个原子的全部六个笛卡尔
分量，因此默认需要中心点加 36 次位移 SCF。所有位移 SCF 都从同一个中心收敛态
出发，避免 `+h` 和 `-h` 使用不同历史造成不对称误差。

默认通过条件为：

- frozen $F_{\mathrm{II}}$ 误差小于 $10^{-8}$ Ha/Bohr；
- frozen $F_{\mathrm{loc}}$ 和 $F_{\mathrm{NL}}$ 误差小于 $10^{-7}$ Ha/Bohr；
- 最小步长的全 SCF 总力误差小于 $2\times10^{-4}$ Ha/Bohr；
- $\max|\sum_I\mathbf F_I|<10^{-6}$ Ha/Bohr。

有限电子温度下必须差分 `TOTEN`，不能差分 `energy without entropy` 或
`energy(sigma->0)`。程序以非零退出码报告任何一项失败。

### 1.5 固定晶胞 BFGS 离子弛豫

通用驱动支持 `calculation = relax`。优化变量是可移动原子的笛卡尔坐标，梯度取
Mermin 自由能 `TOTEN` 的导数，即解析力的负值。相邻离子步直接复用上一步的密度
和每个 k 点/自旋通道所属 MPI rank 上的轨道；固定晶胞下平面波基组不变，因此
不需要轨道插值。体系信息和 k 点列表只在弛豫开始时输出一次；SCF 默认输出紧凑
DAV 诊断，可用 `verbosity = silent` 隐藏，或用 `verbosity = detailed` 增加
能带等细节。紧凑 SCF
摘要中 `F=` 左侧的整数是该次电子计算实际完成的 SCF 轮数。

```bash
make pwdft
./pwdft examples/si8_displaced_mp2_relax.in

# 多 k 点体系优先使用现有的 k 点 MPI
make pwdft_mpi
mpiexec -n 8 ./pwdft_mpi examples/si8_displaced_mp2_relax.in
```

示例从一个原子发生非对称位移的 8 原子金刚石 Si 常规胞出发，先用完整 MP
$2^3$ 网格降低开发成本。收敛后可把输入改成 MP $4^3$ 进行最终检查。主要离子参数
为：

```text
ion_algorithm = bfgs
max_ionic_steps = 30
max_backtracks = 5
force_tolerance_ha_bohr = 2.0e-4
max_step_angstrom = 0.05
bfgs_initial_curvature_ha_bohr2 = 0.10
contcar = CONTCAR
trajectory = si8_relaxation.xyz
```

实现使用完整逆 Hessian BFGS，并包含最大原子位移限制、非下降方向重置、曲率
检查和能量上升时的二分缩步。曲率除法保护阈值与允许的 SCF 能量噪声均作为内部
$10^{-8}$ 常量，不再暴露为输入参数。所有原子完全自由时，会投影掉三个整体平移零模；
存在选择性约束时不做这一投影。`Selective dynamics` 保留 POSCAR 的
坐标语义：Cartesian 输入的 `T/F` 约束笛卡尔方向，Direct 输入则约束相应晶格
方向；后者会先正交化为同一允许位移子空间，再交给 BFGS。每个接受步都会更新当前
目录下的 `CONTCAR` 和 XYZ 轨迹，并输出：

```text
ION:    3  F= ...  dF= ...  max|force|= ...  SCF= ...
     search= BFGS  dE(linear)= ...  max|dR|= ... Angstrom
```

这里的 `F` 是有限电子温度变分自由能，不是 `energy(sigma->0)`；`dF` 是接受步的
实际能量变化，`dE(linear)` 是旧梯度沿实际位移给出的线性预测，`max|dR|` 是任一
原子的最大笛卡尔位移。可用
[`examples/si8_displaced_qe_mp2_relax.in`](examples/si8_displaced_qe_mp2_relax.in)
进行 Quantum ESPRESSO 固定晶胞 BFGS 对照；QE 的
`forc_conv_thr = 4.0d-4` Ry/Bohr 对应本程序的
$2.0\times10^{-4}$ Ha/Bohr。

独立的谐振势回归覆盖一步精确 BFGS、固定坐标、SCF 初猜复用和上坡步回退：

```bash
make test_relaxation
./test_relaxation
```

### 1.6 Davidson 性能诊断

高 cutoff 下，基组规模近似按

$$
N_{\mathrm{PW}} \propto G_{\max}^3
\propto E_{\mathrm{cut}}^{3/2}
$$

增长，因此从 10 Ha 提高到 20 Ha 并不只是把工作量乘二。Davidson 的主要开销是
对一组轨道施加 Hamiltonian。令当前正交子空间为 $V$，缓存

$$
W=HV,
$$

加入新的修正方向 $T$ 时只计算 $HT$，然后同步扩展

$$
V'=[V,T],\qquad W'=[W,HT].
$$

Rayleigh--Ritz 所需的矩阵因此仍为

$$
H_{\mathrm{sub}}=V^\dagger W,
$$

但不再在每次 Davidson 迭代中对旧的 $V$ 重复计算 $HV$。代码还缓存
$H_{\mathrm{sub}}$：初始子空间完整构造一次，扩展后只新增
$V_{\mathrm{old}}^\dagger HT$ 与 $T^\dagger HT$，另一半由 Hermitian 对称性填充。
厚重启时使用 Ritz 对

$$
X=VA,\qquad HX=WA,
$$

所以重启本身也不需要额外的 $H\psi$。$V$ 和 $W$ 在每次 Davidson 求解开始时按
`max_subspace` 一次性分配，后续扩展与厚重启只更新有效列，不再重新分配并复制
整个长矩阵。修正方向组成块 $T$，先以块矩阵乘法执行

$$
T\leftarrow T-V(V^\dagger T),
$$

再由小 Gram 矩阵 $T^\dagger T$ 做对称正交化并删除线性相关方向。第二遍投影采用
DGKS 判据：仅当第一遍后至少一个方向的范数下降到原来的 50% 以下时执行。这样
保留困难修正方向的两遍稳定性，并让已经近似正交的块少做两次长矩阵乘法。如果
没有留下新的线性无关方向，代码就从 $(X,HX)$ 重启，而不是在同一个子空间里无限
循环。

默认 SCF 行采用 VASP 风格：

```text
       N       E                     dE             d eps       ncg     rms          rms(c)
DAV:   8    -0.3167089E+02        -0.1234E-07    -0.3921E-06    180   0.241E-08    0.715E-07
```

`E` 是含固定离子能的本轮变分能；`dE` 是相邻 SCF 轮的变分能变化；`d eps` 是
占据加权带能
$\sum_{\sigma\mathbf kn}w_{\mathbf k}f_{\sigma n\mathbf k}
\varepsilon_{\sigma n\mathbf k}$ 的变化；`ncg` 是本轮实际 $H\psi$ 次数；`rms`
是全部已求解能带的 Davidson 残差均方根；`rms(c)` 是输入和输出密度之差的
范数。自旋计算对电荷密度与磁化密度残差取较大者；该值用于诊断和调整下一轮
Davidson 精度，但不参与外层收敛判断。只有 `|dE|` 和 `|d eps|` 同时小于
`energy_tolerance_ha` 才满足外层判据。末尾还会给出累计
`N_Hpsi`、`N_Hblock`、Davidson 迭代/重启数、`Hpsi_time`、子空间对角化时间和
SCF 总时间。平均 block 宽度可由 `N_Hpsi/N_Hblock` 估计。性能汇总进一步拆分为

- SCF 阶段墙钟时间：输入势、本征求解、占据、密度/动能、输出势/能量和混合；
- $H\psi$：scatter、反向/正向 FFT、$V(\mathbf r)\psi$、gather+动能和非局域势；
- 密度构造：有效轨道数、scatter、反向 FFT 和实空间累加。

其中 `subspace_time` 只统计小矩阵对角化；`ortho/Ritz/other` 是 Davidson 总时间
扣除 $H\psi$ 和小矩阵对角化后的余量。后续两行会进一步报告：

- `initial_ortho`：初始试探子空间的两遍 Gram--Schmidt；
- `VtW`：构造并数值 Hermitian 化 $H_{\mathrm{sub}}=V^\dagger W$；
- `Ritz(X/HX)`：两次长矩阵旋转 $X=VA$ 和 $HX=WA$；
- `residual+prec`：残差构造、范数计算和 Davidson 对角预处理；
- `result_copy`：把本轮本征值、Ritz 轨道和残差复制到返回结果；
- `correction_ortho`：修正块对旧子空间的两遍块投影以及基于
  $T^\dagger T$ 的对称正交化；
- `restart`：厚重启时用 $(X,HX)$ 替换 $(V,W)$；
- `assemble(T)`：保留的兼容计时项；修正方向现已直接构造为块，因此应接近零；
- `expand/copy`：把新 $(T,HT)$ 写入预分配的 $(V,W)$ 有效列；
- `unaccounted`：原 `ortho/Ritz/other` 扣除上述细项后的剩余时间。

`Davidson reuse` 还会输出 `VtW full/incremental/Ritz` 和
`correction reorth/blocks`。前者用于确认完整 $V^\dagger W$ 只在每次 Davidson
求解开始时构造一次，后续走增量更新或厚重启复用；后者给出 DGKS 实际触发第二遍
投影的块数。

`unaccounted` 可用于发现尚未单独包住的对象构造、结果复制或日志开销。
MPI 输出中的 SCF 阶段取最慢 rank，$H\psi$ 与密度内部计时则为 rank 求和；单 rank
计算时二者都是普通墙钟时间。
`h2_opt` 的每个几何点也会显示 `N_Hpsi`、`N_Hblock`、$H\psi$ 耗时与 SCF 耗时。
可以用下面三组输入建立串行基线：

```bash
./h2_opt pseudopotentials/H.pz-vbc.UPF 10.0 12.0 36
./h2_opt pseudopotentials/H.pz-vbc.UPF 15.0 12.0 44
./h2_opt pseudopotentials/H.pz-vbc.UPF 20.0 12.0 52
```

真实 NC-UPF 的非对称 Si2 力验证可运行：

```bash
make si2_force_check
./si2_force_check pseudopotentials/Si.pz-vbc.UPF 10.0 16.0 44 0.05 0.002
```

该程序把 0 号 Si 的解析力分别与中心差分比较：固定密度局域力、
离子--离子 Ewald 力、固定轨道非局域力，以及每个位移点重新收敛后
Mermin 自由能的总力。最后一个比较必须使用有限温度自由能，而不是
`energy without entropy` 或 `energy(sigma->0)`。

FFT 尺寸不能只为了速度任意减小，因为实空间乘积含有平面波频率之差；`h2_opt`
会在网格不足时主动报错。新增的 `test_davidson` 用同一 FFT Hamiltonian 建立稠密矩阵，
比较最低本征值和残差，并检查增量 $W=HV$ 缓存没有退化成每轮全子空间重算：

```bash
make test_davidson
./test_davidson
```

### 1.7 批量 Hamiltonian 与 FFT

`apply_hamiltonian_to_block` 不再逐列调用标量 $H\psi$。对一个含 $m$ 个轨道的 block，
平面波系数被布置成 $m$ 个连续的三维 reciprocal grids，然后由
`fftw_plan_many_dft` 批量执行

$$
C(\mathbf G)
\xrightarrow{\mathrm{FFT}^{-1}}
\psi(\mathbf r)
\xrightarrow{V(\mathbf r)}
V(\mathbf r)\psi(\mathbf r)
\xrightarrow{\mathrm{FFT}}
(V\psi)(\mathbf G).
$$

FFTW plan 按 block 宽度缓存并复用；共享缓冲区扩容前会先销毁旧 plan，避免 plan
持有失效指针。单批最多处理 16 个轨道，因此临时内存为 $O(16N_{\mathrm{FFT}})$，
而不是随总能带数无限增长。非局域算符也按 block 计算：

$$
V_{\mathrm{NL}}C
=\sum_\alpha
\beta_\alpha D_\alpha(\beta_\alpha^\dagger C).
$$

`test_batched_hamiltonian` 对 1、4、7、19 和 3 列的 block 依次比较批量与旧标量路径；
19 列同时覆盖 16+3 的分块边界：

```bash
make test_batched_hamiltonian
./test_batched_hamiltonian
```

批量 plan 使用与标量 plan 相同的 `fft_threads`。FFT 前后的连续网格操作由 OpenMP
处理；FFT 本身由 `libfftw3_threads` 处理。`test_batched_hamiltonian` 还比较一线程
和两线程的 $H\psi$、密度与 Hartree 势，并检查计时计数器。实际任务应根据新的分阶段输出做
1/2/4/8/16 线程强标度测试；Rayleigh--Ritz/正交化只有在 `eigensolver` 与
`Hpsi_time` 的差额明显上升后才值得优先接入并行线性代数。

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

Bloch 轨道的周期部分采用平面波展开：

$$
\psi_{n\mathbf k}(\mathbf r)
=e^{i\mathbf k\cdot\mathbf r}u_{n\mathbf k}(\mathbf r),
\qquad
u_{n\mathbf k}(\mathbf r)=\frac{1}{\sqrt\Omega}
\sum_{\mathbf G}c_{n\mathbf k}(\mathbf G)e^{i\mathbf G\cdot\mathbf r}.
$$

每个 k 点分别按动能截断：

$$
\frac{|\mathbf G+\mathbf k|^2}{2}\le E_{\mathrm{cut}}.
$$

FFTW 的 forward transform 没有 $1/N$ 归一化，所以代码构造 Fourier 系数
时显式除以实空间网格点数；backward transform 则直接实现 Fourier 求和。

## 3. Kohn–Sham 方程和 FFT-based $H\psi$

当前单粒子方程为

$$
\hat H\psi_{n\mathbf k}
=\left[-\frac12\nabla^2+V_{\mathrm{loc}}(\mathbf r)
+V_H(\mathbf r)+V_{\mathrm{xc}}(\mathbf r)
+\hat V_{\mathrm{NL}}\right]\psi_{n\mathbf k}
=\varepsilon_{n\mathbf k}\psi_{n\mathbf k}.
$$

动能在倒空间是对角的：

$$
(Tu_{n\mathbf k})_{\mathbf G}
=\frac{|\mathbf G+\mathbf k|^2}{2}c_{n\mathbf k}(\mathbf G).
$$

局域势部分通过

$$
c_{n\mathbf k}(\mathbf G)
\xrightarrow{\mathrm{IFFT}}u_{n\mathbf k}(\mathbf r)
\xrightarrow{\times V_{\mathrm{eff}}(\mathbf r)}
V_{\mathrm{eff}}(\mathbf r)u_{n\mathbf k}(\mathbf r)
\xrightarrow{\mathrm{FFT}}(V_{\mathrm{eff}}u_{n\mathbf k})_{\mathbf G}
$$

计算，因此不需要显式建立稠密矩阵
$V_{\mathbf G\mathbf G'}=V(\mathbf G-\mathbf G')$。

电子数密度为

$$
n(\mathbf r)=\sum_{\mathbf k}w_{\mathbf k}\sum_n
f_{n\mathbf k}|u_{n\mathbf k}(\mathbf r)|^2,
\qquad
\int_\Omega n(\mathbf r)\,d\mathbf r=N_e.
$$

这里 $w_{\mathbf k}>0$ 且 $\sum_{\mathbf k}w_{\mathbf k}=1$。零温和
Fermi–Dirac 占据都通过同一个化学势满足

$$
N_e=\sum_{\mathbf k}w_{\mathbf k}\sum_n f_{n\mathbf k}.
$$

SCF 的每一步先在共同的 $V_{\mathrm{eff}}[n]$ 下依次求解所有 k 点，再汇总带权
输出密度；因此 k 点之间只通过密度、势和全局化学势耦合，后续可以直接在 k 点层面
并行。

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

`nspin = 1` 使用 `XC_UNPOLARIZED`；`nspin = 2` 将
$\rho_\uparrow,\rho_\downarrow$ 按 LibXC 的交错布局送入 `XC_POLARIZED`，并取得
$v_{\mathrm{xc},\uparrow}$ 和 $v_{\mathrm{xc},\downarrow}$。LibXC 要求输入密度
非负，因此进入库前只应清理数值噪声导致的微小负值，而不能用 density floor
改变正常低密度区域。
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

非自旋极化 Fermi–Dirac 占据的最大占据数为2：

$$
f_n=\frac{2}{\exp[(\varepsilon_n-\mu)/\sigma]+1},
\qquad \sum_n f_n=N_e,
$$

共线自旋极化时，每个自旋轨道的最大占据数为1，且两个自旋通道共用一个费米能级：

$$
f_{n\mathbf k\sigma}
=\frac{1}{\exp[(\varepsilon_{n\mathbf k\sigma}-\mu)/\sigma]+1},
\qquad
\sum_{\sigma\mathbf k n}w_{\mathbf k}f_{n\mathbf k\sigma}=N_e.
$$

输入中的

```text
nspin = 2
starting_magnetization = 1.0
```

只用来构造第一轮的均匀自旋密度：

$$
N_\uparrow^{(0)}=\frac{N_e+M_0}{2},\qquad
N_\downarrow^{(0)}=\frac{N_e-M_0}{2}.
$$

后续 $N_\uparrow-N_\downarrow$ 由共同费米能级和自洽能带决定，并没有被约束在
$M_0$。最终输出

$$
M=N_\uparrow-N_\downarrow
=\int_\Omega(\rho_\uparrow-\rho_\downarrow)\,d\mathbf r
$$

并以 $\mu_B$ 报告。可用单H原子示例做最小测试：

```bash
./pwdft examples/h_atom_spin_scf.in
```

也可用 O₂ 示例检查非受限 SCF 能否从非整数初猜磁矩回到三重态：

```bash
OMP_NUM_THREADS=8 ./pwdft examples/o2_triplet_scf.in
```

在 12 Å 立方盒、$r_{\mathrm{OO}}=1.21$ Å、`ecut_ha = 30` 的一次参考计算中，
$M_0=1\ \mu_B$ 最终得到

$$
N_\uparrow/N_\downarrow=5/7,\qquad M=-2\ \mu_B,
$$

以及 $E=-31.73006235859$ Ha、轴向力
$F_z=\mp0.04904463$ Ha/Bohr。这里 $M=-2\ \mu_B$ 与
$M=+2\ \mu_B$ 通过全局自旋翻转相连；无外磁场时二者物理等价，也说明
`starting_magnetization` 只是初猜而不是约束。该结果目前是程序内参考值，仍需
使用相同赝势、截断能、超胞和 FFT 网格与 Quantum ESPRESSO 交叉验证。
该示例采用 `mixing_alpha = 0.30` 作为后续性能基线；`0.10` 的测试出现明显能量
振荡并需要 141 轮 SCF，因此比较代码优化前后性能时应保持 `0.30` 不变。

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

SCF 初期输入势仍在快速变化，无需把每个 Davidson 本征问题都求到最终精度。令
$\tau_f$ 为最终本征残差阈值、$\tau_0$ 为初始阈值、$r_n$ 为上一轮密度残差、
$r_f$ 为目标密度残差，下一轮期望阈值取为

$$
\tau_{\mathrm{desired}}=
\operatorname{clip}\left(
\tau_f\sqrt{\frac{r_n}{r_f}},\tau_f,\tau_0
\right).
$$

实际阈值不会因密度残差反弹而放宽，并且通常每轮最多收紧一个数量级：

$$
\tau_{n+1}=\min\left[
\tau_n,\max\left(\tau_{\mathrm{desired}},\frac{\tau_n}{10}\right)
\right].
$$

当 $|dE|$ 和 $|d\varepsilon|$ 第一次同时达标但 $\tau_n>\tau_f$ 时，下一轮直接
切换到 $\tau_f$；只有这次严格求解后两个能量判据仍然成立，SCF 才被标记为收敛。最终
`eigensolver work` 摘要仍会报告累计工作量；`verbosity = detailed` 额外打印
本轮 `eig_tol` 和最大能带残差。

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

解析回归模型使用的平滑离子 Coulomb 能是

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
\mathrm{erfc}\left(\frac{r_{IJ\mathbf L}}{2\sigma}\right)
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
\widetilde\beta_l(q)=4\pi\int_0^\infty
rj_l(qr)u_{\beta,l}(r)\,dr,
$$

避免在 $r\to0$ 时先除以 $r$。离散形式是

$$
\widetilde\beta_l(q)\simeq4\pi\sum_i
r_i j_l(qr_i)u_{\beta,l}(r_i)\,w_i.
$$

对于位于 $\mathbf R_I$ 的角动量 projector，平面波矩阵元还包含

$$
\langle\mathbf G+\mathbf k|\beta_{I,lm}\rangle
=\frac{4\pi}{\sqrt\Omega}(-i)^lY_{lm}(\widehat{\mathbf q})
\left[\int r^2j_l(qr)\beta_l(r)\,dr\right]
e^{-i\mathbf q\cdot\mathbf R_I},
\qquad \mathbf q=\mathbf G+\mathbf k,
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
=\frac{\mathrm{erf}[r/(\sqrt{2}\sigma)]}{r},
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
- 在远处，$\mathrm{erf}[r/(\sqrt{2}\sigma)]\to1$，所以
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
$1.5\times10^{-7}$ Ha/Bohr，说明能量和力实现彼此一致。相同 UPF、晶胞、截断能
和 PZ-LDA 设置下，Quantum ESPRESSO 给出
$R_{\mathrm{H-H}}=1.4904091638$ Bohr，本程序给出 $1.49041165$ Bohr，差值约
$2.5\times10^{-6}$ Bohr；总能量分别为 $-1.1117005731$ Ha 和
$-1.1117006841$ Ha，差值约 $1.1\times10^{-7}$ Ha。
[NIST CCCBDB 给出的实验平衡键长](https://cccbdb.nist.gov/exp2x.asp?casno=1333740)
约为 $0.7414$ Å；当前模型结果仍偏长。因此现阶段只能断言“代码内部的能量–力一致”，
以及“复现了同设置的 QE 结果”，不能断言“赝势结果已经复现实验”。这里偏长主要是
所选 LDA/赝势和有限周期晶胞的物理近似，不是当前能量–力导数不一致造成的。

## 9. 非局域赝势

Kleinman–Bylander 形式写作

$$
\hat V_{\mathrm{NL}}
=\sum_I\sum_{ij}|\beta_i^I\rangle D_{ij}
\langle\beta_j^I|.
$$

作用在轨道上为

$$
\hat V_{\mathrm{NL}}|\psi_{n\mathbf k}\rangle
=\sum_I\sum_{ij}|\beta_i^I\rangle D_{ij}
\langle\beta_j^I|\psi_{n\mathbf k}\rangle,
$$

非局域能为

$$
E_{\mathrm{NL}}
=\sum_{\mathbf k}w_{\mathbf k}\sum_n f_{n\mathbf k}
\sum_I\sum_{ij}
\langle\psi_{n\mathbf k}|\beta_i^I\rangle D_{ij}
\langle\beta_j^I|\psi_{n\mathbf k}\rangle.
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

### 9.1 实球谐展开

对第 $i$ 个径向 projector 及其角动量 $l_i$，代码生成 $2l_i+1$ 个磁量子数
通道：

$$
B_{I,i,lm}(\mathbf G+\mathbf k)
=\langle\mathbf G+\mathbf k|\beta^I_{i,lm}\rangle
=\frac{(-i)^l}{\sqrt\Omega}
Y_{lm}^{\mathrm{real}}(\widehat{\mathbf q})
\widetilde\beta_{i,l}(q)
e^{-i\mathbf q\cdot\mathbf R_I},
\qquad \mathbf q=\mathbf G+\mathbf k.
$$

这里 $\widetilde\beta$ 已经由径向函数中的 $4\pi$ Fourier–Bessel 变换得到，
所以式子外面不再重复乘 $4\pi$。实球谐采用 Quantum ESPRESSO 的顺序

$$
m=0,\ \cos\phi,\ \sin\phi,\ \cos2\phi,\ \sin2\phi,\ldots
$$

并包含 Condon–Shortley 相位。例如 $l=1$ 三个通道依次正比于
$z/r,-x/r,-y/r$。测试同时检查

$$
\sum_{m=1}^{2l+1}\left[Y_{lm}^{\mathrm{real}}(\widehat{\mathbf q})\right]^2
=\frac{2l+1}{4\pi}.
$$

### 9.2 稠密 $D_{ij}$ 如何接入现有标量通道

标量、无自旋轨道 NC-UPF 中，$D$ 只在相同 $l$、相同实球谐通道之间耦合。
对每个 $l$ 的径向块作实对称本征分解

$$
D^{(l)}=U^{(l)}\Lambda^{(l)}U^{(l)T},
$$

并定义新 projector

$$
|\widetilde\beta_{\alpha lm}\rangle
=\sum_i U^{(l)}_{i\alpha}|\beta_{ilm}\rangle.
$$

于是

$$
\sum_{ij}|\beta_{ilm}\rangle D_{ij}^{(l)}
\langle\beta_{jlm}|
=\sum_\alpha|\widetilde\beta_{\alpha lm}\rangle
\lambda_\alpha^{(l)}
\langle\widetilde\beta_{\alpha lm}|.
$$

右边正好是原有 Hψ、$E_{\mathrm{NL}}$ 和 $F_{\mathrm{NL}}$ 接口所需的“一个
projector 配一个标量 $D$”形式。这不是忽略非对角元，而是对原算符的精确基变换。
代码会拒绝非零的跨 $l$ 耦合，而不是静默套用错误的简并展开。

## 10. Hellmann–Feynman 离子力

当前解析力分为

$$
\mathbf F_I=\mathbf F_I^{\mathrm{loc}}
+\mathbf F_I^{\mathrm{II}}+\mathbf F_I^{\mathrm{NL}}.
$$

局域势和离子–离子项的周期结构因子含有晶格倒矢：

$$
e^{-i\mathbf G\cdot\mathbf R_I},
$$

因此

$$
\frac{\partial}{\partial R_{I,a}}
e^{-i\mathbf G\cdot\mathbf R_I}
=-iG_a e^{-i\mathbf G\cdot\mathbf R_I}.
$$

非局域 projector 则含有 $\mathbf q=\mathbf G+\mathbf k$，所以其位置导数产生
$-iq_a$。这一区别在 Gamma 点不可见，却是多 k 点解析力必须显式处理的部分。

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
b_{i,n\mathbf k}^I=\langle\beta_i^I|\psi_{n\mathbf k}\rangle.
$$

对 Hermitian $D$，非局域力可写为

$$
F_{I,a}^{\mathrm{NL}}
=-2\,\mathrm{Re}\sum_{\mathbf k}w_{\mathbf k}\sum_n f_{n\mathbf k}
\sum_{ij}(b_{i,n\mathbf k}^I)^*D_{ij}
\frac{\partial b_{j,n\mathbf k}^I}{\partial R_{I,a}}.
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

实现中先把每个 $D^{(l)}$ 块变成上一节的本征 projector。对其中一个标量通道
$\alpha$，有

$$
\frac{\partial B_{I,\alpha lm}(\mathbf G+\mathbf k)}{\partial R_{I,a}}
=-iq_aB_{I,\alpha lm}(\mathbf G+\mathbf k),
$$

所以现有标量通道解析力逐项求和就与原稠密 $D_{ij}$ 公式完全等价。
`test_upf_nonlocal` 直接比较“原始稠密 $D$ 算符”和“对角化后的标量通道算符”，
并在固定轨道上对非对称离子位置做三方向中心有限差分。

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

1. 用相同 NC-UPF、cutoff、FFT 网格和超胞分别对H原子与 O₂ 做
   Quantum ESPRESSO 共线自旋交叉验证；
2. 为自旋极化总能量补充真实 UPF 的三维全 SCF 力有限差分；
3. 在 LDA 自旋链路稳定后接入 LibXC PBE/GGA；
4. 增加能带、总态密度和自旋分辨态密度输出。
