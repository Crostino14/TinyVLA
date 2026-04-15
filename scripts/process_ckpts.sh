#!/bin/bash
#SBATCH --account=did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL



# This scripts is used to process the trained weights and generates a smaller and compact weights
LLM_MODEL_SIZE=1.3B
CURRENT_DIR=$(pwd)

# path to trained TinyVLA weights
DATASET_NAME="libero_goal_no_noops"
LORA_R=64 #64 #128 #256
source_dir="/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/checkpoints_saving_folder/tinyvla/tiny_vla_llava_pythia_lora_${DATASET_NAME}_lora_r_${LORA_R}"
# new path to save weights
target_dir="/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/checkpoints_saving_folder/tinyvla/post_processed_tiny_vla_llava_pythia_lora_${DATASET_NAME}_lora_r_${LORA_R}_processed"
min_step=20000  # the minimum checkpoint step to copy

mkdir -p $target_dir

exclude_pattern="global_step*"

echo "copying checkpoints from $source_dir to $target_dir"


checkpoint_excludes=() 
for dir in "$source_dir"/checkpoint-*; do
    if [[ -d "$dir" ]]; then
        step=${dir##*/checkpoint-}
        if [[ "$step" =~ ^[0-9]+$ && "$step" -lt "$min_step" || $(("$step" % "$min_step")) != 0 ]]; then
            checkpoint_excludes+=(--exclude="checkpoint-$step" --exclude="checkpoint-$step/**")
        fi
    fi
done

printf 'Excludes: %q\n' "${checkpoint_excludes[@]}" 

rsync -av "${checkpoint_excludes[@]}" --exclude="$exclude_pattern" --exclude="post-processed" --exclude="post-processed/**" --exclude="$exclude_pattern/**" --exclude="log/**" --exclude="log" "$source_dir/" "$target_dir/"

echo 'tranfer checkpoints to non_lora_trainables.bin'
for dir in "$source_dir"/*/; do
    dir_name=$(basename "$dir")

    echo "Processing directory: $dir_name"

    if [[ "$dir_name" == checkpoint-* ]]; then
        step=${dir_name#checkpoint-}

        # Only process if step number is valid AND >= min_step
        if [[ "$step" =~ ^[0-9]+$ && "$step" -ge "$min_step" && $(("$step" % "$min_step")) -eq 0 ]]; then
            if ! find "$dir" -mindepth 1 -type f -name "non_lora_trainables.bin" | grep -q .; then
                cd "$dir" || exit
                srun python ./zero_to_fp32.py ./ "${target_dir}/${dir_name}/non_lora_trainables.bin"
                cd "$CURRENT_DIR" || exit
            else
                echo "Skipping $dir_name (non_lora_trainables.bin already exists)"
            fi
        else
            echo "Skipping $dir_name (step $step < $min_step or not divisible)"
        fi
    fi
done

cd $CURRENT_DIR 