#!/bin/bash
#SBATCH --job-name=train_xy      
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=1          
#SBATCH --cpus-per-task=96            
#SBATCH --output=sbatch/train_%j.out     
#SBATCH --error=sbatch/train_%j.err      
#SBATCH --nodelist=3090node1              #### 需要修改!
#SBATCH --gres=gpu:4                      #### 需要修改(与下方nproc_per_node一致)!

set -euo pipefail

# 环境变量（仅保留NCCL相关，去掉MASTER_PORT/ADDR）
export NCCL_SOCKET_IFNAME=eno2
export NCCL_DEBUG=INFO

# 可通过 sbatch --export 覆盖的训练参数
TASK_ID="${TASK_ID:-Instinct-Parkour-Target-Amp-G1-v0}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-}"
CONDA_ACTIVATE_PATH="${CONDA_ACTIVATE_PATH:-/mnt/slurmfs-4090node1/homes/xsheng420/miniconda3/bin/activate}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-instinct}"

# 激活环境
source "${CONDA_ACTIVATE_PATH}" "${CONDA_ENV_NAME}"

echo "[INFO] TASK_ID=${TASK_ID}"
echo "[INFO] NUM_ENVS=${NUM_ENVS}"
echo "[INFO] MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "[INFO] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[INFO] SEED=${SEED}"
echo "[INFO] RUN_NAME=${RUN_NAME}"

EXTRA_ARGS=()
if [[ -n "${RUN_NAME}" ]]; then
  EXTRA_ARGS+=("--run_name=${RUN_NAME}")
fi

# 使用srun启动分布式训练
srun python -m torch.distributed.run \
  --nnodes=1  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/instinct_rl/train.py \
  --task="${TASK_ID}" \
  --seed="${SEED}" \
  --distributed \
  --num_envs="${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --headless \
  "${EXTRA_ARGS[@]}"
  
