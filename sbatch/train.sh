#!/bin/bash
#SBATCH --job-name=sheng_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4  # 必须保留：申请GPU资源
#SBATCH --output=train_ddp_%j.out
#SBATCH --error=train_ddp_%j.err
#SBATCH --nodelist=3090node1

# 环境变量（仅保留NCCL相关，去掉MASTER_PORT/ADDR）
export NCCL_SOCKET_IFNAME=eno2
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1
export MASTER_PORT=$((29500 + RANDOM % 20000))
# 激活环境
source /mnt/slurmfs-4090node1/homes/xsheng420/miniconda3/bin/activate instinct

# 使用srun启动分布式训练
srun python -m torch.distributed.run \
  --nnodes=1  --nproc_per_node=4 \
  ../scripts/instinct_rl/train.py \
  --task=Instinct-Parkour-Target-Amp-G1-v0 \
  --distributed \
  --num_envs=512
  