#!/bin/bash
# =====================================================================================================
#  SIP 2026 实验：一条命令跑完全部方法（在终端里运行）
#
#    bash run_all.sh            准备环境 → 试跑 → 正式运行全部方法 → 汇总表 results/summary.md
#    bash run_all.sh test       只准备环境并试跑（约 10 分钟），不保存结果
#    bash run_all.sh summary    只重新生成汇总表
#
#  已完成的结果会自动跳过：中途停了（Ctrl+C、断电、重启），重新运行同一条命令就会接着跑。
#  屏幕上的输出同时记在 results/run_all.log。
# =====================================================================================================
set -u
set -o pipefail
cd "$(dirname "$0")" || exit 1
REPO="$(pwd)"
MODE="${1:-all}"
case "$MODE" in all|test|summary) ;; *) echo "用法: bash run_all.sh [test|summary]"; exit 1 ;; esac
mkdir -p results
exec > >(tee -a results/run_all.log) 2>&1
trap 'echo; echo "已中断。重新运行同一条命令即可接着跑（已完成的结果不会重跑）。"; exit 130' INT

PY="$REPO/.venv/bin/python"
CK="external/HIRDiff/checkpoints/diffusion/I190000_E97_gen.pth"
CK_SHA="fbc657dad536ec4ee7a43599d9d6f497d68dda7ebff49fa6d3af66f3432d56df"
STATUS=()
T_SSTV=0; T_HSDT=0; T_HIRDIFF=0; T_PROP=0; T_S3TTV=0; T_MATLAB=0
MATLAB=""

say() { echo; echo "==== $(date '+%m-%d %H:%M:%S')  $*"; }

# run <名字> <命令...>：运行一步并记录（返回 0 = 完成，3 = 跳过，其他 = 失败；失败后继续下一步）
run() {
    local name="$1"; shift
    say "$name"
    "$@"
    local rc=$?
    if [ $rc -eq 0 ]; then STATUS+=("完成  $name")
    elif [ $rc -eq 3 ]; then STATUS+=("跳过  $name")
    else echo "!!!! $name 失败（exit ${rc}），继续下一步"; STATUS+=("失败  $name"); fi
    return $rc
}

setup() {
    say "1/4 准备环境"
    # uv：Python 环境管理工具，装在自己的用户目录 ~/.local/bin，不需要管理员密码
    local UV
    UV="$(command -v uv 2>/dev/null || true)"
    if [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"; fi
    if [ -z "$UV" ]; then
        echo "没有找到 uv，现在安装到 ~/.local/bin（https://astral.sh/uv）..."
        curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh || return 1
        UV="$HOME/.local/bin/uv"
    fi
    echo "uv: $("$UV" --version)"
    echo "安装 Python 3.10 和所需的包（第一次约 1 GB，需要几分钟）..."
    "$UV" sync || return 1

    # HIR-Diff 的扩散模型（1.56 GB，超过 GitHub 的上限，需要另外拷过来）
    if [ ! -f "$CK" ]; then
        for f in "$HOME/Downloads/I190000_E97_gen.pth" "$HOME/Desktop/I190000_E97_gen.pth" "$REPO/I190000_E97_gen.pth"; do
            if [ -f "$f" ]; then echo "找到 ${f}，移动到 $CK"; mv "$f" "$CK"; break; fi
        done
    fi
    if [ -f "$CK" ]; then
        echo "校验 HIR-Diff 模型文件..."
        if [ "$(shasum -a 256 "$CK" | cut -d' ' -f1)" = "$CK_SHA" ]; then
            echo "HIR-Diff 模型文件 OK"
        else
            echo "HIR-Diff 模型文件不完整（SHA-256 不符），改名为 .bad，HIR-Diff 将跳过"
            mv "$CK" "$CK.bad"
        fi
    else
        echo "没有 HIR-Diff 模型文件 I190000_E97_gen.pth：HIR-Diff 将跳过，其余方法照常运行。"
    fi
    "$PY" -u scripts/check_env.py
}

find_matlab() {
    local d
    for d in /Applications/MATLAB_R*.app; do
        if [ -x "$d/bin/matlab" ]; then MATLAB="$d/bin/matlab"; fi     # 有多个版本时用最新的
    done
    if [ -z "$MATLAB" ] && command -v matlab >/dev/null 2>&1; then MATLAB="$(command -v matlab)"; fi
}

matlab_check() {
    find_matlab
    if [ -z "$MATLAB" ]; then
        echo "没有找到 MATLAB（/Applications/MATLAB_R*.app）：LRTV 和 LRTDTV 将跳过"
        return 3
    fi
    echo "MATLAB: $MATLAB"
    "$MATLAB" -batch "disp(version); fprintf('cores: %d\n', feature('numcores'));"
}

tests() {
    say "2/4 试跑：每个方法只跑几步，确认能运行"
    run "试跑 SSTV" "$PY" -u scripts/run_sstv.py --scene PaviaU --test && T_SSTV=1
    run "试跑 HSDT" "$PY" -u scripts/run_hsdt.py --scene PaviaU --test && T_HSDT=1
    run "试跑 HIR-Diff" "$PY" -u scripts/run_hirdiff.py --scene PaviaU --test && T_HIRDIFF=1
    run "试跑 提案方法" "$PY" -u scripts/run_proposed.py --scene PaviaU --variant proposed --test && T_PROP=1
    run "试跑 S3TTV（30 次迭代，估计总时间）" "$PY" -u scripts/run_s3ttv.py --scene PaviaU --test && T_S3TTV=1
    run "试跑 MATLAB" matlab_check && T_MATLAB=1
}

full() {
    say "3/4 正式运行（已完成的自动跳过）"
    local S V
    for S in PaviaU PaviaC; do
        if [ $T_SSTV = 1 ]; then run "SSTV $S" "$PY" -u scripts/run_sstv.py --scene $S; fi
        if [ $T_HSDT = 1 ]; then run "HSDT $S" "$PY" -u scripts/run_hsdt.py --scene $S; fi
        if [ $T_HIRDIFF = 1 ]; then run "HIR-Diff $S" "$PY" -u scripts/run_hirdiff.py --scene $S; fi
        if [ $T_PROP = 1 ]; then
            for V in proposed delta0 naive2; do
                run "提案方法 $V $S" "$PY" -u scripts/run_proposed.py --scene $S --variant $V
            done
        fi
    done
    if [ $T_MATLAB = 1 ]; then
        run "LRTV、LRTDTV（MATLAB，两张图）" "$MATLAB" -batch "cd('$REPO/matlab'); run_matlab_all"
    fi
    run "汇总表（S3TTV 之前）" "$PY" -u scripts/collect_results.py
    if [ $T_S3TTV = 1 ]; then
        for S in PaviaU PaviaC; do
            run "S3TTV ${S}（最慢，可能要数小时）" "$PY" -u scripts/run_s3ttv.py --scene $S
        done
    fi
}

caffeinate -ims -w $$ >/dev/null 2>&1 &      # 运行期间不让电脑睡眠
echo "开始 $(date '+%Y-%m-%d %H:%M:%S')，模式 ${MODE}，目录 $REPO"
if [ "$MODE" = summary ]; then
    "$PY" -u scripts/collect_results.py
    exit $?
fi
setup || { echo "环境准备失败：请把 results/run_all.log 带回来看。"; exit 1; }
tests
if [ "$MODE" = all ]; then
    full
    say "4/4 汇总表"
    "$PY" -u scripts/collect_results.py
fi
say "结束。各步骤："
for s in ${STATUS[@]+"${STATUS[@]}"}; do echo "  $s"; done
if [ "$MODE" = all ]; then echo "汇总表：$REPO/results/summary.md"; fi
