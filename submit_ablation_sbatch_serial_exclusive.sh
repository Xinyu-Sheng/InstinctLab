#!/bin/bash
set -euo pipefail

# 串行提交 BASE + A1-A6 到 Slurm。
# 适用于“同一时间只能有一个任务在队列/运行”的环境。
#
# 行为：
# 1) 提交一个任务
# 2) 等这个任务结束（成功或失败）
# 3) 再提交下一个任务
#
# 用法示例：
# 1) 最小包（含基线）
#    bash submit_ablation_sbatch_serial.sh --ablations BASE,A1,A2,A3,A5 --seeds 42
#
# 2) 全量包
#    bash submit_ablation_sbatch_serial.sh --ablations BASE,A1,A2,A3,A4,A5,A6 --seeds 42,43,44
#
# 3) 覆盖训练规模
#    bash submit_ablation_sbatch_serial.sh --num-envs 2048 --max-iterations 30000 --nproc-per-node 4
#
# 4) 默认独占节点（阻止其他用户任务上同一节点）
#    bash submit_ablation_sbatch_serial.sh --ablations BASE --seeds 42
#
# 5) 如需关闭独占（不推荐）
#    bash submit_ablation_sbatch_serial.sh --no-exclusive
#
# 6) 仅独占到用户级（若集群支持）
#    bash submit_ablation_sbatch_serial.sh --exclusive-user

ABLATIONS_CSV="BASE,A1,A2,A3,A4,A5,A6"
SEEDS_CSV="42"
NUM_ENVS="2048"
MAX_ITERATIONS="30000"
NPROC_PER_NODE="4"
CONDA_ENV_NAME="instinct"
POLL_SECONDS="30"
EXCLUSIVE_MODE="node"

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
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --exclusive)
      EXCLUSIVE_MODE="node"
      shift 1
      ;;
    --no-exclusive)
      EXCLUSIVE_MODE="off"
      shift 1
      ;;
    --exclusive-user)
      EXCLUSIVE_MODE="user"
      shift 1
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

SBATCH_EXCLUSIVE_ARGS=()
case "${EXCLUSIVE_MODE}" in
  node)
    SBATCH_EXCLUSIVE_ARGS+=("--exclusive")
    ;;
  user)
    SBATCH_EXCLUSIVE_ARGS+=("--exclusive=user")
    ;;
  off)
    ;;
  *)
    echo "Unsupported exclusive mode: ${EXCLUSIVE_MODE}"
    echo "Supported: node,user,off"
    exit 1
    ;;
esac

# Log directories are created under logs/instinct_rl/<experiment_name>/<timestamp>_<run_name>.
# run_name is set to <ablation>_seed<seed>_<DATE_TAG> in this script.
declare -A TASK_MAP=(
  [BASE]="Instinct-Parkour-Target-Amp-G1-v0"                # baseline AMP G1 parkour task with MoE policy + WasabiPPO; logs under logs/instinct_rl/g1_parkour/<timestamp>_<run_name>
  [A1]="Instinct-Parkour-Target-Amp-G1-A1-NoAmp-v0"         # same env, same MoE policy, but AMP reward disabled; logs under logs/instinct_rl/g1_parkour_a1_no_amp/<timestamp>_<ablation>_seed<seed>_<DATE_TAG>
  [A2]="Instinct-Parkour-Target-Amp-G1-A2-NoDepth-v0"       # same env, pure state-based MoE policy without depth encoder input; logs under logs/instinct_rl/g1_parkour_a2_no_depth/<timestamp>_<ablation>_seed<seed>_<DATE_TAG>
  [A3]="Instinct-Parkour-Target-Amp-G1-A3-Moe1-v0"          # same env, MoE collapsed to a single expert; logs under logs/instinct_rl/g1_parkour_a3_moe1/<timestamp>_<ablation>_seed<seed>_<DATE_TAG>
  [A4]="Instinct-Parkour-Target-Amp-G1-A4-LowSensorRand-v0" # env variation: reduced sensor randomization and no depth-frame delay; logs under logs/instinct_rl/g1_parkour/<timestamp>_<ablation>_seed<seed>_<DATE_TAG>
  [A5]="Instinct-Parkour-Target-Amp-G1-A5-NoPenetration-v0" # env variation: terrain penetration reward shaping disabled; logs under logs/instinct_rl/g1_parkour/<timestamp>_<ablation>_seed<seed>_<DATE_TAG>
  [A6]="Instinct-Parkour-Target-Amp-G1-A6-NoSymAug-v0"      # env variation: symmetric motion augmentation disabled; logs under logs/instinct_rl/g1_parkour/<timestamp>_<ablation>_seed<seed>_<DATE_TAG>
  [A7]="Instinct-Parkour-Target-Amp-G1-A7-Recurrent-v0"     # same env, switch from feed-forward MoE policy to recurrent actor-critic; logs under logs/instinct_rl/g1_parkour_a7_recurrent/<timestamp>_<ablation>_seed<seed>_<DATE_TAG>
  [A8]="Instinct-Parkour-Target-Attention-G1-v0"            # same env; attention-based policy using MapAttentionEncoder; algorithm=PPO (no AMP/discriminator); logs under logs/instinct_rl/g1_parkour_attention/<timestamp>_<ablation>_seed<seed>_<DATE_TAG>
  [A9]="Instinct-Parkour-Target-Attention-Amp-Old-G1-v0"        # same env; attention-based policy using MapAttentionEncoder; algorithm=WasabiPPO (AMP enabled — discriminator present, discriminator_reward_coef=0.25); logs under logs/instinct_rl/g1_parkour_attention_amp/<timestamp>_<ablation>_seed<seed>_<DATE_TAG>
)

IFS=',' read -r -a ABLATIONS <<< "$ABLATIONS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

mkdir -p sbatch
DATE_TAG="$(date +%Y%m%d)"

echo "[INFO] Serial submit mode enabled."
echo "[INFO] Ablations: ${ABLATIONS_CSV}"
echo "[INFO] Seeds: ${SEEDS_CSV}"
echo "[INFO] Num envs: ${NUM_ENVS}, Max iterations: ${MAX_ITERATIONS}, NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "[INFO] Exclusive mode: ${EXCLUSIVE_MODE}"

total=0
for ablation in "${ABLATIONS[@]}"; do
  if [[ -z "${TASK_MAP[$ablation]+x}" ]]; then
    echo "Unsupported ablation: ${ablation}"
    echo "Supported: BASE,A1,A2,A3,A4,A5,A6,A7,A8,A9"
    exit 1
  fi

  task_id="${TASK_MAP[$ablation]}"

  for seed in "${SEEDS[@]}"; do
    run_name="${ablation}_seed${seed}_${DATE_TAG}"
    job_name="${ablation}_s${seed}"

    submit_output=$(sbatch \
      --job-name="${job_name}" \
      --output="sbatch/${job_name}_%j.out" \
      --error="sbatch/${job_name}_%j.err" \
      "${SBATCH_EXCLUSIVE_ARGS[@]}" \
      --export=ALL,TASK_ID="${task_id}",SEED="${seed}",RUN_NAME="${run_name}",NUM_ENVS="${NUM_ENVS}",MAX_ITERATIONS="${MAX_ITERATIONS}",NPROC_PER_NODE="${NPROC_PER_NODE}",CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
      train.sh)

    job_id=$(echo "$submit_output" | awk '{print $4}')
    if [[ -z "$job_id" ]]; then
      echo "[ERROR] Failed to parse job id from: $submit_output"
      exit 1
    fi

    echo "[SUBMITTED] ${job_name} -> task=${task_id}, seed=${seed}, run_name=${run_name}, job_id=${job_id}"
    total=$((total + 1))

    # wait until job leaves queue/running list
    while squeue -h -j "$job_id" >/dev/null 2>&1 && [[ -n "$(squeue -h -j "$job_id")" ]]; do
      echo "[WAIT] job ${job_id} is still active. sleep ${POLL_SECONDS}s..."
      sleep "$POLL_SECONDS"
    done

    # check final state from sacct if available
    if command -v sacct >/dev/null 2>&1; then
      state=$(sacct -j "$job_id" --format=State --noheader | head -n 1 | xargs || true)
      echo "[DONE] job ${job_id} finished with state: ${state:-UNKNOWN}"
      if [[ -n "${state}" && "${state}" != "COMPLETED" ]]; then
        echo "[ERROR] job ${job_id} did not complete successfully. Stop serial pipeline."
        exit 1
      fi
    else
      echo "[WARN] sacct not available; cannot validate final state. Continue by queue exit only."
    fi
  done
done

echo "[DONE] serially submitted and finished ${total} jobs."
