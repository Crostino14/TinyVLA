#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=train_tinyvla
#SBATCH --partition=gpuq          # Partition (queue) name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1           # Only ONE task per node!
#SBATCH --gres=gpu:4                # Request 4 GPUs per node
#SBATCH --cpus-per-task=64             # Adjust for data loading, etc.
#SBATCH --exclusive
#SBATCH --export=ALL

export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p /tmp/$USER/triton_cache
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache


ACTION_HEAD=droid_diffusion # specify action policy head type

# define OUTPUT path
TASK_NAME="${1:-libero_object_no_noops}"
RESUME_FROM_CHECKPOINT="${2:-False}"
LORA_R="${3:-64}"
MODEL_NAME_PATH="${4:-/mnt/beegfs/a.cardamone7/checkpoints_saving_folder/tinyvla/llava_pythia_libero_goal_no_noops_64/1.3B}"
OUTPUT=/mnt/beegfs/a.cardamone7/checkpoints_saving_folder/tinyvla/tiny_vla_llava_pythia_lora_${TASK_NAME}_lora_r_${LORA_R}


echo "Training on dataset: $TASK_NAME"
echo "Resume from checkpoint: $RESUME_FROM_CHECKPOINT"
echo "LoRA rank: $LORA_R"
echo "Model base path: $MODEL_NAME_PATH"
echo "Output path: $OUTPUT"
if [ -d "$OUTPUT" ]; then
   echo 'output exists'
else
   echo '!!output not exists!!'
   mkdir -p $OUTPUT
fi
# backup the train scripts
# cp ./scripts/train.sh $OUTPUT
# detailed usage of each parameter can be found in train_tinyvla.py
# assign a unique port based on the Slurm job ID
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
# echo "Using MASTER_PORT=$MASTER_PORT"
deepspeed --master_port $MASTER_PORT --include="localhost:0,1,2,3" ../train_tinyvla.py \
  --deepspeed "/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA/llava-pythia/scripts/zero2.json" \
  --lora_enable True \
  --lora_module 'vit llm' \
  --load_pretrain False \
  --pretrain_image_size 320 \
  --resume_from_checkpoint ${RESUME_FROM_CHECKPOINT} \
  --lora_r ${LORA_R} \
  --lora_alpha 256 \
  --non_lora_lr 2e-5 \
  --task_name "$TASK_NAME" \
  --model_name_or_path "$MODEL_NAME_PATH" \
  --version v0 \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower True \
  --freeze_backbone True \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length False \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 100000 \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 50 \
  --seed 0 \
  --learning_rate 2e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.005 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 32 \
  --lazy_preprocess False \
  --find_unused_parameters True \
  --action_head_type $ACTION_HEAD \
  --use_state True \
  --concat "token_cat" \
  --window_size 6 \
  --evaluation_strategy "no" \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log

for dir in "$OUTPUT"/*/ ; do
    # 检查文件夹名称是否包含'checkpoint'
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp /home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA/scripts/preprocessor_config.json $dir
    fi
done

