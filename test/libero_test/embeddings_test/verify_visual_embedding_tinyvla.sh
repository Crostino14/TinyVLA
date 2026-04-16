#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=verify_visual_emb_tinyvla
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/verify_visual_emb_tinyvla_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/verify_visual_emb_tinyvla_%j.err

WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA/test/libero_test"
TINYVLA_ROOT="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"

echo "=========================================="
echo "Verify Visual Embedding Independence — TinyVLA"
echo "=========================================="
echo "Start time: $(date)"

export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${LIBERO_PATH}:${TINYVLA_ROOT}:${WORK_DIR}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS##*:}
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate tinyvla_libero

cd ${WORK_DIR}

python verify_visual_embedding_tinyvla.py

echo ""
echo "Finish time: $(date)"
echo "=========================================="
