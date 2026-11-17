# SIP 2026 实验：在一台电脑上用一条命令跑完全部方法

## 大白话版

- 这个仓库里有海报要比较的 9 个方法的代码、两张测试图（PaviaU 和 Pavia Centre，case 5 噪声，seed 0），以及需要的模型权重。
- 在 Mac Studio 上只做三件事：
  1. 下载并解压；
  2. 在"终端"里输入一行命令；
  3. 等它跑完，打开 `results/summary.md` 看结果表。
- 所有方法在同一台电脑上运行和计时。表里有 MPSNR、MSSIM、SAM、运行时间，还有 MacBook 上的旧结果，方便对照。
- 只有 HIR-Diff 的模型文件（1.56 GB）太大，GitHub 放不下，要另外用 AirDrop 传过去。放在"下载"文件夹就行，脚本会自己找到。没有这个文件也能跑，只是跳过 HIR-Diff。

## 在 Mac Studio 上的步骤

1. **下载**
   - 用 Safari 登录 GitHub，打开这个仓库的页面。
   - 点绿色的 **Code** 按钮 → **Download ZIP**。
   - 在"下载"文件夹里双击 ZIP 解压。把解压出来的文件夹（`sip2026-hsi-experiments-main`）改名为 `sip2026-hsi-experiments`，再拖到个人文件夹里（Finder 左栏带小房子图标的那个）。
2. **（可选）传 HIR-Diff 模型**
   - 文件在 MacBook 的 `03_实验工程/外部代码/HIRDiff/checkpoints/diffusion/I190000_E97_gen.pth`。
   - 用 AirDrop 发到 Mac Studio，留在"下载"文件夹即可。
3. **运行**
   - 打开"终端"（Spotlight 搜 Terminal），粘贴下面一行，回车：

     ```bash
     bash ~/sip2026-hsi-experiments/run_all.sh
     ```

   - 第一次运行会：
     - 自动安装 uv（管理 Python 环境的小工具）。它装在自己的用户目录 `~/.local/bin`，不需要管理员密码。如果研究室规定装任何软件都要先问久米老师，就先问一下。
     - 下载 Python 3.10 和 PyTorch 等，约 1 GB，需要几分钟。
   - 接着先"试跑"：每个方法只跑几步，约 10 分钟。没问题就自动开始正式运行。
   - 运行期间：
     - **不要注销（log out）**，锁屏没关系；
     - **不要关终端窗口**；
     - 脚本会阻止电脑睡眠。
   - 中途停了（Ctrl+C、断电）：重新运行同一行命令就会接着跑，已完成的不会重跑。
   - 如果弹出"终端想访问'下载'文件夹"，点"好"（脚本要从那里找 HIR-Diff 的模型）。
4. **看结果**
   - 用"文本编辑"打开 `~/sip2026-hsi-experiments/results/summary.md`。
   - 想重新生成汇总表，运行：

     ```bash
     bash ~/sip2026-hsi-experiments/run_all.sh summary
     ```

5. **带回 MacBook**：在 Finder 里右键 `results` 文件夹 → 压缩 → AirDrop。

只想先确认能不能跑（不保存结果），可以运行：

```bash
bash ~/sip2026-hsi-experiments/run_all.sh test
```

## 运行顺序和大概时间

括号里是在 MacBook Pro（M2 Pro）上的时间，仅供参考。

1. 试跑（约 10 分钟）。
2. 两张图依次运行：
   - SSTV（GPU，约 10 秒）
   - HSDT（GPU，约 15–30 秒）
   - HIR-Diff（GPU，几分钟）
   - 提案方法 (3)、δ = 0、(2) 共 3 个（GPU，各约 9–11 分钟）
3. LRTV 和 LRTDTV，两张图（MATLAB，CPU，每次约 6 秒 / 35 秒）。
4. 生成一次汇总表：到这里大约 1–1.5 小时。
5. **S3TTV** 两张图（CPU，最慢）。试跑时会打印估计时间，可能要几个到十几个小时，所以放在最后。
6. 再生成一次汇总表。

## 方法和参数

所有参数都只在 PaviaU 上选定，Pavia Centre 直接沿用。

| 方法 | 代码 | 参数 |
|---|---|---|
| 提案 (3) | 我们的代码 `src/solver_pvs.py` | τ = 0.3，κ = 2，λ = 0.2，β = 0.4，r = 2，600 次迭代，γ_init = 0.031 |
| (3)，δ = 0 | 同上 | δ 固定为 0，其余同上 |
| (2) | 同上 | δ = 0，τ = 0（只有 ν 上的 MCP 项） |
| SSTV [2] | 按原文 (4) 式自己实现，`src/sstv_sparse.py` | λ = 0.2，μ = 0.4，1500 次迭代 |
| S3TTV | 官方 Python 版（MIT 许可） | 块 8×8（256 不能被 10 整除），半径 = 0.95 × 真实噪声范数 |
| LRTV | 官方 MATLAB 代码（作者 He Wei 主页） | τ = 0.0025，λ = 10/√(MN)，秩 3 |
| LRTDTV | 官方 MATLAB 代码 | τ = 1，λ = 10，秩 [0.8M, 0.8N, 4] |
| HSDT | 官方代码 + 公开权重（ICVL，complex noise） | GSSA 用 flex 模式，以适应 48 个波段 |
| HIR-Diff | 官方代码 + ddpm-cd 的预训练扩散模型 | 官方设定（exp，k = 8，20 步，η1 = 16，η2 = 10，Rr = 3），等间隔选波段，seed 0 |

- **线程**：CPU 方法（S3TTV、LRTV、LRTDTV）用一半的 CPU 核。GPU 方法（提案、SSTV、HSDT、HIR-Diff）用 Apple GPU（MPS）。
- **计时**：只计复原本身，不含读数据和加载模型。
- **HIR-Diff 的注意点**：官方采样器每一步都和真值比 PSNR，并返回最好的一步。这里保留官方做法，论文里要写明。
- **指标**：
  - 用海报同一套代码 `src/metrics.py`。
  - 另外单独计算"34 个真实波段"的 MPSNR。48 个波段里，第 0–2 和 37–47 个（从 0 数）是两端波段的复制，ROSIS 只覆盖 430–860 nm。

## 单独运行某一个方法

先 `cd ~/sip2026-hsi-experiments`，再运行对应的命令（`--force` 表示已有结果也重跑）：

```bash
.venv/bin/python scripts/run_proposed.py --scene PaviaU --variant proposed --force
```

`--variant` 可以是 `proposed`、`delta0` 或 `naive2`，`--scene` 可以是 `PaviaU` 或 `PaviaC`。其他方法：

- `scripts/run_sstv.py`
- `scripts/run_hsdt.py`
- `scripts/run_hirdiff.py`
- `scripts/run_s3ttv.py`

MATLAB 方法：在 MATLAB 里把当前文件夹设为 `matlab/`，输入 `run_matlab_all`。

## 出问题时

- **某个方法失败**：脚本会继续跑其他方法，最后列出失败的步骤。把 `results/run_all.log` 带回来看。
- **MATLAB 失败**：先从"应用程序"里打开一次 MATLAB，登录或激活后关掉，再重新运行命令。
- **uv 装不上**：可能是网络限制，或者需要老师批准。其余步骤都依赖它，先解决这一步。
- **提示 MPS 不可用**：GPU 方法会改用 CPU，会慢很多。把 log 带回来看。

## 文件结构

```text
run_all.sh            唯一的入口：准备环境、试跑、全部方法、汇总
pyproject.toml        Python 环境（版本和 MacBook 上跑通的一致）
data/                 两张测试图（.mat：input = 含噪图，gt = 真值，256×256×48）
configs/              S3TTV 的半径
src/                  公共代码：数据读取、指标、提案方法的求解器、SSTV
scripts/              每个方法的运行脚本、环境检查、汇总表
matlab/               LRTV 和 LRTDTV 的运行脚本
external/             第三方代码：HSIGene（VAE）、HSDT、HIR-Diff、S3TTV、LRTV、LRTDTV
weights/              HSDT 的权重
results/              运行结果（自动生成，不上传）
```

## 注意

**仓库请保持 Private。** 里面有别人的代码和模型权重（各自的许可证见 `external/*/LICENSE` 和 `readme`），只用于我们自己的实验。
