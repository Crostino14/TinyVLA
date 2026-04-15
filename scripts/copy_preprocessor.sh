DATASET_NAME=libero_goal_no_noops
LORA_R=64
OUTPUT=/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/checkpoints_saving_folder/tiny_vla/post_processed_tiny_vla_llava_pythia_lora_${DATASET_NAME}_lora_r_${LORA_R}_processed

for dir in "$OUTPUT"/*/ ; do
    # 检查文件夹名称是否包含'checkpoint'
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        echo "Coping preprocessor_config.json to $dir"
        cp /home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA/scripts/preprocessor_config.json $dir
    fi
done
