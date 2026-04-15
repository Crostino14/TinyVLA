#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=l2v_tinyvla_libero_eval
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-11        # 0-2=tasks0-2, 3-5=tasks3-5, 6-8=tasks6-8, 9-11=task9
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/L2_Variations_Eval_tinyvla_54000_libero_goal_l2_seed_%a_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/L2_Variations_Eval_tinyvla_54000_libero_goal_l2_seed_%a_%j.err


# ==========================================
# TinyVLA LIBERO Evaluation with Array Jobs
# ==========================================


# Get seed from SLURM array task ID
ARRAY_ID=$SLURM_ARRAY_TASK_ID
SEED=$((ARRAY_ID / 4))           # 0-3→seed0, 4-7→seed1, 8-11→seed2
TASK_GROUP=$((ARRAY_ID % 4))     # 0=tasks0-2, 1=tasks3-5, 2=tasks6-8, 3=task9


# Mappa task group → range
case $TASK_GROUP in
    0) TASK_RANGE="0-2";   TASK_GROUP_NAME="tasks0-2" ;;
    1) TASK_RANGE="3-5";   TASK_GROUP_NAME="tasks3-5" ;;
    2) TASK_RANGE="6-8";   TASK_GROUP_NAME="tasks6-8" ;;
    3) TASK_RANGE="9-9";   TASK_GROUP_NAME="task9"   ;;
esac


echo "ARRAY_ID=$ARRAY_ID → SEED=$SEED, TASKS=$TASK_RANGE"


# ==========================================
# Configuration Parameters
# ==========================================


# Command variation settings (MODIFY THESE)
CHANGE_COMMAND=true   # Set to 'true' to use command variations, 'false' for default
COMMAND_LEVEL="l2"    # Options: 'default', 'l1', 'l2', 'l3', 'all', 'all_no_default'


# Task suite
TASK_SUITE="libero_goal"     # Options: libero_goal, libero_spatial, libero_object, libero_10, libero_90


# Model configuration
MODEL_PATH="/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/checkpoints_saving_folder/tinyvla/post_processed_tiny_vla_llava_pythia_lora_libero_goal_no_noops_lora_r_64_processed/checkpoint-54000"
MODEL_BASE="/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/checkpoints_saving_folder/tinyvla/llava_pythia_libero_goal_no_noops_64/1.3B"


# Directories
WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA/test/libero_test"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"
OUTPUT_DIR="/mnt/beegfs/a.cardamone7/outputs"


# Evaluation parameters
NUM_TRIALS_PER_TASK=50
NUM_STEPS_WAIT=10
ENV_IMG_RES=256


# ==========================================
# Create ID Note for Logging
# ==========================================


if [ "$CHANGE_COMMAND" == "true" ]; then
    ID_NOTE_SUFFIX="L2_Variations_Eval_tinyvla_${TASK_SUITE}_54000_${COMMAND_LEVEL}_seed${SEED}_${TASK_GROUP_NAME}"
else
    ID_NOTE_SUFFIX="L2_Variations_Eval_tinyvla_${TASK_SUITE}_default_54000_seed${SEED}_${TASK_GROUP_NAME}"
fi


# ==========================================
# Print Configuration
# ==========================================


echo "=========================================="
echo "TinyVLA LIBERO Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID (Seed): $SLURM_ARRAY_TASK_ID"
echo "Seed: $SEED, Tasks: $TASK_RANGE"
echo "Task Suite: $TASK_SUITE"
echo "Change Command: $CHANGE_COMMAND"
echo "Command Level: $COMMAND_LEVEL"
echo ""
echo "Model Path: $MODEL_PATH"
echo "Model Base: $MODEL_BASE"
echo ""
echo "Start time: $(date)"
echo "=========================================="
echo ""


# ==========================================
# Environment Setup
# ==========================================


# MuJoCo setup
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia


TINYVLA_ROOT="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA"
export PYTHONPATH=${LIBERO_PATH}:${TINYVLA_ROOT}:${WORK_DIR}:$PYTHONPATH


# CUDA setup
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS##*:}
export CUDA_LAUNCH_BLOCKING=1


# Disable warnings
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true


# ==========================================
# Activate Conda Environment
# ==========================================


source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate tinyvla_libero


echo "Working directory: ${WORK_DIR}"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""


# ==========================================
# Change to Work Directory
# ==========================================


cd ${WORK_DIR}


# ==========================================
# Run Evaluation
# ==========================================


echo "Starting evaluation: SEED $SEED, TASKS $TASK_RANGE"
echo ""


if [ "$CHANGE_COMMAND" == "true" ]; then
    # Run with command variations
    srun python run_libero_eval.py \
        --model_path ${MODEL_PATH} \
        --model_base ${MODEL_BASE} \
        --model_family tiny_vla \
        --task_suite_name ${TASK_SUITE} \
        --task_range ${TASK_RANGE} \
        --change_command True \
        --command_level ${COMMAND_LEVEL} \
        --run_number ${SEED} \
        --num_trials_per_task ${NUM_TRIALS_PER_TASK} \
        --num_steps_wait ${NUM_STEPS_WAIT} \
        --env_img_res ${ENV_IMG_RES} \
        --initial_states_path DEFAULT \
        --seed ${SEED} \
        --run_id_note ${ID_NOTE_SUFFIX} \
        --local_log_dir ${OUTPUT_DIR}/logs \
        --summary_file ${OUTPUT_DIR}/logs/summary/${COMMAND_LEVEL}/seed${SEED}_${TASK_RANGE}.json \
        --use_wandb False \
        --debug False
else
    # Run with default commands only
    srun python run_libero_eval.py \
        --model_path ${MODEL_PATH} \
        --model_base ${MODEL_BASE} \
        --model_family tiny_vla \
        --task_suite_name ${TASK_SUITE} \
        --task_range ${TASK_RANGE} \
        --change_command False \
        --run_number ${SEED} \
        --num_trials_per_task ${NUM_TRIALS_PER_TASK} \
        --num_steps_wait ${NUM_STEPS_WAIT} \
        --env_img_res ${ENV_IMG_RES} \
        --initial_states_path DEFAULT \
        --seed ${SEED} \
        --run_id_note ${ID_NOTE_SUFFIX} \
        --local_log_dir ${OUTPUT_DIR}/logs \
        --summary_file ${OUTPUT_DIR}/logs/summary/default/seed${SEED}_${TASK_RANGE}.json \
        --use_wandb False \
        --debug False
fi


EXIT_CODE=$?


# ==========================================
# Completion
# ==========================================


echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Seed: $SEED, Tasks: $TASK_RANGE"
    echo "Command Level: $COMMAND_LEVEL"
else
    echo "Evaluation failed with exit code: $EXIT_CODE"
fi
echo "Finish time: $(date)"
echo "=========================================="


exit $EXIT_CODE