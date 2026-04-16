#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=vqa_libero
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/vqa_test_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/vqa_test_%j.err

# Optional CLI overrides:
#   sbatch vqa_libero.sh
#   sbatch vqa_libero.sh libero_spatial 10
#   sbatch vqa_libero.sh libero_goal 6 /custom/output/dir
SUITE=${1:-"libero_goal"}
MAX_PAIRS=${2:-"6"}
OUTPUT_DIR=${3:-"/mnt/beegfs/a.cardamone7/outputs/vqa_test"}

WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA/test/vqa_test"
TINYVLA_ROOT="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"

SCRIPT="${WORK_DIR}/vqa_libero.py"
OUTPUT_JSON="${OUTPUT_DIR}/vqa_libero_output.json"

echo "=========================================="
echo "LIBERO Spatial Benchmark Generator"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "SUITE=${SUITE}"
echo "MAX_PAIRS=${MAX_PAIRS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "OUTPUT_JSON=${OUTPUT_JSON}"
echo "SCRIPT=${SCRIPT}"

export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${LIBERO_PATH}:${TINYVLA_ROOT}:${WORK_DIR}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS##*:}
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate tinyvla_libero

mkdir -p "${OUTPUT_DIR}/frames"
mkdir -p "${OUTPUT_DIR}/positions"
mkdir -p "/mnt/beegfs/a.cardamone7/outputs/logs"

cd "${WORK_DIR}"

echo "Working directory: ${WORK_DIR}"
echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo ""

srun python "${SCRIPT}" \
    --libero_path "${LIBERO_PATH}" \
    --output_dir  "${OUTPUT_DIR}" \
    --suite       "${SUITE}" \
    --max_pairs   "${MAX_PAIRS}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Benchmark generation completed successfully"
    echo "Output JSON: ${OUTPUT_JSON}"
    echo "Frames:      ${OUTPUT_DIR}/frames/"
    echo "Positions:   ${OUTPUT_DIR}/positions/"
else
    echo "Benchmark generation failed with exit code ${EXIT_CODE}"
fi
echo "Finish time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
