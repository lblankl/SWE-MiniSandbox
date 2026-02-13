

#!/usr/bin/env bash
set -e

# ===== 默认值 =====
TAR_IO=40
SANDBOX=128
CONTAINER=512
CONTAINER_START_UP=512
SANDBOX_START_UP=128
Env_path=/home/zeta/SWE/miniconda3/bin/activate  # Important! This is the conda env path for ray to create venv, it should be the same as the env you activated when installing the dependencies and running this script.

# ===== 解析命令行参数 =====
# 支持的参数：
#   --tar-io 40
#   --sandbox 128
#   --container 512
#   --container-start-up 512
#   --sandbox-start-up 128
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tar-io)
      TAR_IO="$2"
      shift 2
      ;;
    --sandbox)
      SANDBOX="$2"
      shift 2
      ;;
    --container)
      CONTAINER="$2"
      shift 2
      ;;
    --container-start-up)
      CONTAINER_START_UP="$2"
      shift 2
      ;;
    --sandbox-start-up)
      SANDBOX_START_UP="$2"
      shift 2
      ;;
    --env-path)
      Env_path="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo "用法示例: bash xx.sh --tar-io 80 --sandbox 256"
      exit 1
      ;;
  esac
done
# ===== 打印当前配置（可选）=====
echo "TAR_IO           = ${TAR_IO}"
echo "SANDBOX          = ${SANDBOX}"
echo "CONTAINER        = ${CONTAINER}"
echo "CONTAINER_START_UP = ${CONTAINER_START_UP}"
echo "SANDBOX_START_UP   = ${SANDBOX_START_UP}"
echo "Env_path          = ${Env_path}"
source $Env_path rl


ray stop


export WANDB_API_KEY=""

unset PIP_CONSTRAINT


# 2) 基本变量（来自平台）
MASTER_ADDR=${MASTER_ADDR:-$PET_MASTER_ADDR}   # 优先 MASTER_ADDR，没有就用 PET_MASTER_ADDR
MASTER_PORT=${MASTER_PORT:-${PET_MASTER_PORT:-6379}}

# 当前容器自己的 IP（取第一个非 127 的）
THIS_IP=$(hostname -I | awk '{print $1}')
worker_IP=192.168.154.246,192.168.174.182,192.168.150.33
echo "==== 环境变量 ===="
echo "MASTER_ADDR     = ${MASTER_ADDR}"
echo "MASTER_PORT     = ${MASTER_PORT}"
echo "PET_MASTER_ADDR = ${PET_MASTER_ADDR}"
echo "PET_MASTER_PORT = ${PET_MASTER_PORT}"
echo "THIS_IP         = ${THIS_IP}"
echo "================="


# 3) 解析 MASTER_ADDR：如果是 IP 就直接用；如果是主机名就尝试解析一次
if echo "${MASTER_ADDR}" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
    MASTER_IP="${MASTER_ADDR}"
else
    MASTER_IP=$(getent hosts "${MASTER_ADDR}" | awk '{print $1}')
    if [ -z "${MASTER_IP}" ]; then
        echo "WARNING: 无法通过 DNS 解析 MASTER_ADDR='${MASTER_ADDR}'，先假设当前节点是 HEAD。"
        MASTER_IP="${THIS_IP}"
    fi
fi

echo "MASTER_IP       = ${MASTER_IP}"
echo "HEAD            = ${MASTER_IP}:${MASTER_PORT}"

# 4) 判断当前是不是 head 节点（用 IP 对比）
mkdir -p /home/ray_tmp
echo "[HEAD] 当前节点是 Head，启动 Ray head: ${MASTER_IP}:${MASTER_PORT}"
# ===== 组装 resources JSON =====
RAY_RESOURCES="{\"tar_io\": ${TAR_IO}, \"sandbox\": ${SANDBOX}, \"container\": ${CONTAINER}, \"container_start_up\": ${CONTAINER_START_UP}, \"sandbox_start_up\": ${SANDBOX_START_UP}}"

ray start --head \
    --node-ip-address="${MASTER_IP}" \
    --port="${MASTER_PORT}" \
    --dashboard-host=0.0.0.0 \
    --temp-dir=/home/ray_tmp \
    --resources="${RAY_RESOURCES}" \
    --dashboard-port=8265


