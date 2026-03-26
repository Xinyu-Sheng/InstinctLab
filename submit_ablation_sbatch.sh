#!/bin/bash
set -euo pipefail

# 批量提交 A1-A6 消融任务到 Slurm。
# 每个任务都通过 sbatch train.sh 提交，并将 task/seed/run_name 通过 --export 传入 train.sh。
#
# 用法示例：
# 1) 最小包（A1,A2,A3,A5，seed=42）
#    bash submit_ablation_sbatch.sh --ablations A1,A2,A3,A5 --seeds 42
#
# 2) 全量包（A1-A6，seed=42,43,44）
#    bash submit_ablation_sbatch.sh --ablations A1,A2,A3,A4,A5,A6 --seeds 42,43,44
#
# 3) 覆盖训练规模
#    bash submit_ablation_sbatch.sh --seeds 42 --num-envs 1024 --max-iterations 30000
#
# 4) 指定分布式进程数（需与 sbatch 的 gpu 资源匹配）
#    bash submit_ablation_sbatch.sh --nproc-per-node 4

ABLATIONS_CSV="A1,A2,A3,A4,A5,A6"
SEEDS_CSV="42"
NUM_ENVS="1024"
MAX_ITERATIONS="30000"
NPROC_PER_NODE="4"
CONDA_ENV_NAME="instinct"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ablations)
      ABLATIONS_CSV="$2"
      shift 2
      ;;
    --seeds)
      SEEDS_CSV="$2"
      shift 2
      ;;
    --num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --max-iterations)
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --nproc-per-node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

declare -A TASK_MAP=(
  [A1]="Instinct-Parkour-Target-Amp-G1-A1-NoAmp-v0"
  [A2]="Instinct-Parkour-Target-Amp-G1-A2-NoDepth-v0"
  [A3]="Instinct-Parkour-Target-Amp-G1-A3-Moe1-v0"
  [A4]="Instinct-Parkour-Target-Amp-G1-A4-LowSensorRand-v0"
  [A5]="Instinct-Parkour-Target-Amp-G1-A5-NoPenetration-v0"
  [A6]="Instinct-Parkour-Target-Amp-G1-A6-NoSymAug-v0"
)

IFS=',' read -r -a ABLATIONS <<< "$ABLATIONS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

mkdir -p sbatch
DATE_TAG="$(date +%Y%m%d)"

echo "[INFO] Submit ablations: ${ABLATIONS_CSV}"
echo "[INFO] Seeds: ${SEEDS_CSV}"
echo "[INFO] Num envs: ${NUM_ENVS}, Max iterations: ${MAX_ITERATIONS}, NPROC_PER_NODE: ${NPROC_PER_NODE}"

total=0
for ablation in "${ABLATIONS[@]}"; do
  if [[ -z "${TASK_MAP[$ablation]+x}" ]]; then
    echo "Unsupported ablation: ${ablation}"
    echo "Supported: A1,A2,A3,A4,A5,A6"
    exit 1
  fi

  task_id="${TASK_MAP[$ablation]}"

  for seed in "${SEEDS[@]}"; do
    run_name="${ablation}_seed${seed}_${DATE_TAG}"
    job_name="${ablation}_s${seed}"

    sbatch \
      --job-name="${job_name}" \
      --output="sbatch/${job_name}_%j.out" \
      --error="sbatch/${job_name}_%j.err" \
      --export=ALL,TASK_ID="${task_id}",SEED="${seed}",RUN_NAME="${run_name}",NUM_ENVS="${NUM_ENVS}",MAX_ITERATIONS="${MAX_ITERATIONS}",NPROC_PER_NODE="${NPROC_PER_NODE}",CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
      train.sh

    echo "[SUBMITTED] ${job_name} -> task=${task_id}, seed=${seed}, run_name=${run_name}"
    total=$((total + 1))
  done
done

echo "[DONE] submitted ${total} jobs."
