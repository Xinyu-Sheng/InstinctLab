#!/bin/bash
#SBATCH --job-name=train_xy      
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=1          
#SBATCH --cpus-per-task=96            
#SBATCH --output=sbatch/train_%j.out     
#SBATCH --error=sbatch/train_%j.err      
#SBATCH --nodelist=3090node1  

# 环境变量（仅保留NCCL相关，去掉MASTER_PORT/ADDR）
export NCCL_SOCKET_IFNAME=eno2
export NCCL_DEBUG=INFO

# 激活环境
source /mnt/slurmfs-4090node1/homes/xsheng420/miniconda3/bin/activate instinct

# 使用srun启动分布式训练
srun python -m torch.distributed.run \
  --nnodes=1  --nproc_per_node=4 \
 scripts/instinct_rl/train.py \
  --task=Instinct-Parkour-Target-Amp-G1-v0 \
  --distributed \
  --num_envs=256
  