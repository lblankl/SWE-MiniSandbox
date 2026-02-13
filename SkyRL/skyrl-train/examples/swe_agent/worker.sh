

#!/usr/bin/env bash
set -e

# ===== 默认值 =====
TAR_IO=40
SANDBOX=128
CONTAINER=512
CONTAINER_START_UP=512
SANDBOX_START_UP=128
Env_path=/home/zeta/SWE/miniconda3/bin/activate  # Important! This is the conda env path for ray to create venv, it should be the same as the env you activated when installing the dependencies and running this script.
MASTER_IP=192.168.150.33
MASTER_PORT=6379
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
    --master-ip)
      MASTER_IP="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
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
export WANDB_API_KEY=

unset PIP_CONSTRAINT


THIS_IP=$(hostname -I | awk '{print $1}')

echo "==== 环境变量 ===="

echo "THIS_IP         = ${THIS_IP}"
echo "================="



echo "MASTER_IP       = ${MASTER_IP}"
echo "HEAD            = ${MASTER_IP}:${MASTER_PORT}"


echo "[WORKER] 当前节点是 Worker，连接到 Ray head: ${MASTER_IP}:${MASTER_PORT}"
ray start \
    --address="${MASTER_IP}:${MASTER_PORT}" \
    --resources="{\"tar_io\": ${TAR_IO},\"sandbox\":${SANDBOX}, \"container\": ${CONTAINER}, \"container_start_up\": ${CONTAINER_START_UP},\"sandbox_start_up\":${SANDBOX_START_UP}}" \
    --node-ip-address="${THIS_IP}"
