#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=vqa_tinyvla_eval
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/vqa_tinyvla_eval_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/vqa_tinyvla_eval_%j.err

# Optional CLI overrides:
#   sbatch tinyvla_vqa_eval.sh /path/model_checkpoint /path/model_base
#   For base VLM checkpoints (e.g. .../1.3B), omit model_base.
#   Optional tuning:
#   sbatch tinyvla_vqa_eval.sh /path/model_checkpoint /path/model_base 1.0 0.55 token_cat
#   Last arg (visual concat): token_cat for dual-image, or none for single-image.
MODEL_PATH=${1:-"/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/checkpoints_saving_folder/tinyvla/llava_pythia_base/1.3B"}
MODEL_BASE=${2:-""}
SIMILARITY_THRESHOLD=${4:-"0.80"}
VISUAL_CONCAT=${5:-"token_cat"}

WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA/test/vqa_test"
TINYVLA_ROOT="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"

PROMPTS_JSON="/mnt/beegfs/a.cardamone7/outputs/vqa_test/spatial_benchmark.json"
OUTPUT_JSON="${WORK_DIR}/vqa_results.json"

QUESTION_DELAY_SEC=3
TASK_DELAY_SEC=5

echo "=========================================="
echo "TinyVLA VQA Eval"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "MODEL_PATH=${MODEL_PATH}"
if [ -n "${MODEL_BASE}" ] && [ "${MODEL_BASE}" != "None" ] && [ "${MODEL_BASE}" != "none" ]; then
  echo "MODEL_BASE=${MODEL_BASE}"
else
  echo "MODEL_BASE=<empty> (not used)"
fi
echo "PROMPTS_JSON=${PROMPTS_JSON}"
echo "OUTPUT_JSON=${OUTPUT_JSON}"
echo "QUESTION_DELAY_SEC=${QUESTION_DELAY_SEC}"
echo "TASK_DELAY_SEC=${TASK_DELAY_SEC}"
echo "SIMILARITY_THRESHOLD=${SIMILARITY_THRESHOLD}"
echo "VISUAL_CONCAT=${VISUAL_CONCAT}"

export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${LIBERO_PATH}:${TINYVLA_ROOT}:${WORK_DIR}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS##*:}
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate tinyvla_libero

cd "${WORK_DIR}"

echo "Working directory: ${WORK_DIR}"
echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo ""

CMD=(
  python tinyvla_vqa_eval.py
  --model_path "${MODEL_PATH}"
  --prompts_json "${PROMPTS_JSON}"
  --output_json "${OUTPUT_JSON}"
  --device cuda
  --max_new_tokens 100
  --conv_mode pythia
  --frames camera
  --question_delay_sec "${QUESTION_DELAY_SEC}"
  --task_delay_sec "${TASK_DELAY_SEC}"
  --similarity_threshold "${SIMILARITY_THRESHOLD}"
)

if [ -n "${VISUAL_CONCAT}" ] && [ "${VISUAL_CONCAT}" != "None" ] && [ "${VISUAL_CONCAT}" != "none" ]; then
  CMD+=(--visual_concat "${VISUAL_CONCAT}")
fi

# Pass model_base only for LoRA checkpoints.
if [ -n "${MODEL_BASE}" ] && [ "${MODEL_BASE}" != "None" ] && [ "${MODEL_BASE}" != "none" ]; then
  CMD+=(--model_base "${MODEL_BASE}")
fi

srun "${CMD[@]}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
  echo "VQA evaluation completed successfully"
else
  echo "VQA evaluation failed with exit code ${EXIT_CODE}"
fi
echo "Finish time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
