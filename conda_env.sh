
tg_dir=/home/zeta/SWE/conda
# ===== 解析命令行参数 =====
# 支持的参数：
#   --tg-dir /home/zeta/SWE/conda 默认
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tg-dir)
        tg_dir="$2"
        shift 2
        ;;
        *)
        echo "未知参数: $1"
        echo "用法示例: bash conda_env.sh --tg-dir /home/zeta/SWE/conda"
        exit 1
        ;;
    esac
    done
source $tg_dir/3.11/miniconda3/bin/activate  
conda install -c conda-forge "libstdcxx-ng>=13" "gcc" "gxx_linux-64"
source $tg_dir/3.9/miniconda3/bin/activate  
conda install -c conda-forge "libstdcxx-ng>=13" "gcc" "gxx_linux-64"