#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=extract_emb_tinyvla
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:59:00
#SBATCH --array=2
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/tinyvla_extract_emb_%A_%a.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/tinyvla_extract_emb_%A_%a.err

# ==========================================
# TinyVLA - Extract Pre-Action-Head Embeddings
# ==========================================
# Captures hidden_states from GPT-NeoX (Pythia) backbone
# BEFORE they are passed to the diffusion action head.
# Uses a forward hook on the backbone for clean extraction.
# ==========================================

# Configuration
MODEL_PATH="/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/checkpoints_saving_folder/tinyvla/post_processed_tiny_vla_llava_pythia_lora_libero_goal_no_noops_lora_r_64_processed/checkpoint-54000"
MODEL_BASE="/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/checkpoints_saving_folder/tinyvla/parte2_llava_pythia_libero_goal_no_noops_64/1.3B"
WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA/test/libero_test"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"
TINYVLA_ROOT="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA"
OUTPUT_DIR="/mnt/beegfs/a.cardamone7/outputs/embeddings/tinyvla/10_rollouts_first_step_only"

# Command levels array (index 0-3)
COMMAND_LEVELS=("default" "l1" "l2" "l3")
COMMAND_LEVEL=${COMMAND_LEVELS[$SLURM_ARRAY_TASK_ID]}

# Rollout parameters
TASK_SUITE="libero_goal"
NUM_ROLLOUTS=10  # 10 rollouts per task (10 tasks = 100 episodes per command level)
FIRST_STEP_ONLY=true  # Set to false for full rollout

echo "=========================================="
echo "TinyVLA - Extracting Pre-Action-Head Embeddings"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID (Array: $SLURM_ARRAY_TASK_ID)"
echo "Start time: $(date)"
echo "Model: $MODEL_PATH"
echo "Model base: $MODEL_BASE"
echo "Task suite: $TASK_SUITE"
echo "Command level: $COMMAND_LEVEL"
echo "Rollouts per task: $NUM_ROLLOUTS"
echo "First step only: $FIRST_STEP_ONLY"
echo "Total episodes: $((NUM_ROLLOUTS * 10))"
echo "Output dir: $OUTPUT_DIR"
echo ""

# ==========================================
# Environment Setup
# ==========================================

export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${LIBERO_PATH}:${TINYVLA_ROOT}:${WORK_DIR}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate tinyvla_libero

echo "Working directory: ${WORK_DIR}"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Change to work directory
cd ${WORK_DIR}

# Build command
FIRST_STEP_FLAG=""
MODE_SUFFIX="full"
if [ "$FIRST_STEP_ONLY" = true ]; then
    FIRST_STEP_FLAG="--first_step_only"
    MODE_SUFFIX="first_step"
fi

python extract_embeddings_rollout.py \
    --model_path ${MODEL_PATH} \
    --model_base ${MODEL_BASE} \
    --task_suite ${TASK_SUITE} \
    --command_levels ${COMMAND_LEVEL} \
    --output_dir ${OUTPUT_DIR} \
    --num_rollouts ${NUM_ROLLOUTS} \
    ${FIRST_STEP_FLAG}

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Finish time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Embedding extraction completed!"
else
    echo "Extraction failed with exit code: $EXIT_CODE"
fi
echo "Command level: ${COMMAND_LEVEL}"
echo "Mode: ${MODE_SUFFIX}"
echo "Output saved to: ${OUTPUT_DIR}/rollout_embeddings_tinyvla_${TASK_SUITE}_${COMMAND_LEVEL}_${MODE_SUFFIX}_r${NUM_ROLLOUTS}.pkl"
echo "=========================================="
