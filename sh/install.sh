basedir=/home/zeta/SWE
# ===== 解析命令行参数 =====
# 支持的参数：
#   --basedir /home/zeta/SWE 默认
while [[ $# -gt 0 ]]; do
    case "$1" in
        --basedir)
        basedir="$2"
        shift 2
        ;;
        *)
        echo "未知参数: $1"
        echo "用法示例: bash install.sh --basedir /home/zeta/SWE"
        exit 1
        ;;
    esac
    done


unset PIP_CONSTRAINT
source $basedir/miniconda3/bin/activate

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n rl python==3.11 -y
conda activate rl
pip install weave
cd $basedir/SWE-MiniSandbox/SkyRL/skyrl-train
pip install -e . --index-url https://pypi.org/simple

cd $basedir/SWE-MiniSandbox/SkyRL/skyrl-gym
pip install -e.
basep=$basedir/SWE-MiniSandbox
cd $basep/R2E-Gym
pip install -e.
cd $basep/SWE-smith
pip install -e.
cd $basep/SWE-ReX
pip install -e.
cd $basep/SWE-agent
pip install -e.
cd $basep/SWE-bench
pip install -e.
cd $basep/sandboxdev
pip install -e.

pip install ray[default]==2.50
pip install hydra-core loguru jaxtyping torchdata peft wandb 
pip install vllm==0.11
pip install --upgrade datasets
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install /us3/yuandl/resource/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install tomli tomli-w
pip install httpbin
apt-get install -y graphviz
pip install transformers==4.57
pip install flash-attn

