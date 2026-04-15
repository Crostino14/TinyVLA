#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=ablation_tinyvla
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:59:00
#SBATCH --array=0 # 3 seeds × 9 variants (task7: stove1..4 + task8: bowl_plate1..5)
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/ablation_tinyvla_Task8_%a_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/ablation_tinyvla_Task8_%a_%j.err

# ==========================================
# Usage (override --array to select a single task):
#   Both tasks:   sbatch run_libero_ablation.sh                              (27 jobs)
#   Task 8 only:  sbatch --array=0-3,9-12,18-21 run_libero_ablation.sh      (12 jobs)
#   Task 9 only:  sbatch --array=4-8,13-17,22-26 run_libero_ablation.sh     (15 jobs)
# ==========================================

# ==========================================
# Derive seed, task, and variant from array index
# ==========================================
ARRAY_ID=$SLURM_ARRAY_TASK_ID
NUM_VARIANTS=9                            # 4 (stove) + 5 (bowl_plate)
SEED=$((ARRAY_ID / NUM_VARIANTS))         # 0-8 → seed0 | 9-17 → seed1 | 18-26 → seed2
VARIANT_IDX=$((ARRAY_ID % NUM_VARIANTS))  # 0-3=stove | 4-8=bowl_plate

case $VARIANT_IDX in
    0) ABLATION_TASK_ID=7; ABLATION_TEST_KEY="stove1" ;;
    1) ABLATION_TASK_ID=7; ABLATION_TEST_KEY="stove2" ;;
    2) ABLATION_TASK_ID=7; ABLATION_TEST_KEY="stove3" ;;
    3) ABLATION_TASK_ID=7; ABLATION_TEST_KEY="stove4" ;;
    4) ABLATION_TASK_ID=8; ABLATION_TEST_KEY="bowl_plate1" ;;
    5) ABLATION_TASK_ID=8; ABLATION_TEST_KEY="bowl_plate2" ;;
    6) ABLATION_TASK_ID=8; ABLATION_TEST_KEY="bowl_plate3" ;;
    7) ABLATION_TASK_ID=8; ABLATION_TEST_KEY="bowl_plate4" ;;
    8) ABLATION_TASK_ID=8; ABLATION_TEST_KEY="bowl_plate5" ;;
esac

ID_NOTE="ablation_tinyvla_task${ABLATION_TASK_ID}_seed${SEED}_${ABLATION_TEST_KEY}"

# ==========================================
# Model configuration
# ==========================================
MODEL_PATH="/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/checkpoints_saving_folder/tinyvla/post_processed_tiny_vla_llava_pythia_lora_libero_goal_no_noops_lora_r_64_processed/checkpoint-54000"
MODEL_BASE="/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/checkpoints_saving_folder/tinyvla/parte2_llava_pythia_libero_goal_no_noops_64/1.3B"

WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA/test/libero_test"
TINYVLA_ROOT="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"
OUTPUT_DIR="/mnt/beegfs/a.cardamone7/outputs"

# Evaluation parameters
TASK_SUITE="libero_goal"
# ABLATION_TASK_ID set above from array index (7="Turn on the stove", 8="Put the bowl on the plate")
NUM_TRIALS_PER_TASK=50
NUM_STEPS_WAIT=10
ENV_IMG_RES=256

echo "=========================================="
echo "Starting ABLATION STUDY (TinyVLA)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $ARRAY_ID  →  Seed: $SEED, Task: $ABLATION_TASK_ID, Variant: $ABLATION_TEST_KEY"
echo "Model: $MODEL_PATH"
echo "Start time: $(date)"
echo "=========================================="

# ==========================================
# Environment setup
# ==========================================
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${LIBERO_PATH}:${TINYVLA_ROOT}:${WORK_DIR}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS##*:}
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

# ==========================================
# Activate conda environment
# ==========================================
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate tinyvla_libero

echo "Working directory: ${WORK_DIR}"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# ==========================================
# Change to work directory
# ==========================================
cd ${WORK_DIR}

# ==========================================
# Run ablation study (single variant)
# ==========================================
srun python run_libero_ablation.py \
    --model_path ${MODEL_PATH} \
    --model_base ${MODEL_BASE} \
    --model_family tiny_vla \
    --task_suite_name ${TASK_SUITE} \
    --ablation_task_id ${ABLATION_TASK_ID} \
    --ablation_test_key ${ABLATION_TEST_KEY} \
    --num_trials_per_task ${NUM_TRIALS_PER_TASK} \
    --num_steps_wait ${NUM_STEPS_WAIT} \
    --env_img_res ${ENV_IMG_RES} \
    --initial_states_path DEFAULT \
    --seed ${SEED} \
    --run_number ${SEED} \
    --run_id_note ${ID_NOTE} \
    --local_log_dir ${OUTPUT_DIR}/logs \
    --use_wandb False \
    --debug False

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Ablation study completed successfully!"
else
    echo "Ablation study failed with exit code: $EXIT_CODE"
fi
echo "Seed: $SEED | Variant: $ABLATION_TEST_KEY"
echo "Finish time: $(date)"
echo "=========================================="

exit $EXIT_CODE
