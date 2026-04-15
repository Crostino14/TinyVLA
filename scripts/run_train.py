import subprocess
import time
import os
from huggingface_hub import snapshot_download
import shutil

START_ITERATION=1
TINY_VLA_SIZE='1.3B'
TASK_NAME='libero_goal_no_noops'
CHECKPOINT_PATH=f"/mnt/beegfs/a.cardamone7/checkpoints_saving_folder/tinyvla"
# 'ur5e_pick_place_rm_central_spawn'
# 'ur5e_pick_place_rm_one_spawn'
#'ur5e_pick_place_removed_spawn_regions'
#'ur5e_pick_place_delta_removed_0_5_10_15'
#'ur5e_pick_place_rm_12_13_14_15'
#"ur5e_pick_place_delta_all"
# "ur5e_pick_place_only_0_4_8_12"
LORA_R=64 #256 #128 #64
DOWNLOAD_MODEL=False

if __name__ == "__main__":
    
    for i in range(START_ITERATION, 10):
        
        if i == 1 and DOWNLOAD_MODEL:
            # download llava_pythia model
            os.makedirs(f"{CHECKPOINT_PATH}/llava_pythia_{TASK_NAME}_{LORA_R}/{TINY_VLA_SIZE}", exist_ok=True)
            
            # check if the model already exists
            if not os.path.exists(f"{CHECKPOINT_PATH}/llava_pythia_{TASK_NAME}_{LORA_R}/{TINY_VLA_SIZE}/model.safetensors"):
                print(f"Downloading llava_pythia model to {CHECKPOINT_PATH}/llava_pythia_{TASK_NAME}_{LORA_R}/{TINY_VLA_SIZE}...")
                snapshot_download(
                    repo_id=f"lesjie/Llava-Pythia-{TINY_VLA_SIZE}", 
                    local_dir=f"{CHECKPOINT_PATH}/llava_pythia_{TASK_NAME}_{LORA_R}/{TINY_VLA_SIZE}",
                    resume_download=True,
                )
                print("Download completed!")
            else:
                # delete the existing files except for model.safetensors
                print(f"Model already exists. Removing directory {CHECKPOINT_PATH}/llava_pythia_{TASK_NAME}_{LORA_R}/{TINY_VLA_SIZE}...")
                shutil.rmtree(f"{CHECKPOINT_PATH}/llava_pythia_{TASK_NAME}_{LORA_R}/{TINY_VLA_SIZE}")
                os.makedirs(f"{CHECKPOINT_PATH}/llava_pythia_{TASK_NAME}_{LORA_R}/{TINY_VLA_SIZE}", exist_ok=True)
                print(f"Re-downloading llava_pythia model to {CHECKPOINT_PATH}/llava_pythia_{TASK_NAME}_{LORA_R}/{TINY_VLA_SIZE}...")
                snapshot_download(
                    repo_id=f"lesjie/Llava-Pythia-{TINY_VLA_SIZE}", 
                    local_dir=f"{CHECKPOINT_PATH}/llava_pythia_{TASK_NAME}_{LORA_R}/{TINY_VLA_SIZE}",
                    resume_download=True,
                )
                print("Download completed!")

                try:
                    shutil.rmtree("/home/A.CARDAMONE7/.cache/huggingface")
                except Exception as e:
                    print(f"Error removing cache directory: {e}")

        print(f"Starting iteration {i}")

        if i != 1:
            resume_from_checkpoint = True
        else:
            resume_from_checkpoint = False

        if DOWNLOAD_MODEL:
            result = subprocess.run(['sbatch', 'train.sh'] + [f"{TASK_NAME}", f"{resume_from_checkpoint}", f"{LORA_R}", f"{CHECKPOINT_PATH}/llava_pythia_{TASK_NAME}_{LORA_R}/{TINY_VLA_SIZE}"], capture_output=True, text=True)
        else:
            result = subprocess.run(['sbatch', 'train.sh'] + [f"{TASK_NAME}", f"{resume_from_checkpoint}", f"{LORA_R}"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error submitting job: {result.stderr}")
            exit(1)
            
        # Extract job ID from sbatch output
        job_id = None
        for line in result.stdout.split('\n'):
            if "Submitted batch job" in line:
                job_id = line.split()[-1]
                break
        
        print(f"Job {job_id} submitted. Waiting for completion...")
        # Poll the job status using squeue
        while True:
            result = subprocess.run(['squeue', '--job', job_id], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error checking job status: {result.stderr}")
                exit(1)

            if job_id not in result.stdout:
                print(f"Job {job_id} completed.")
                break

            time.sleep(10)  # Wait for 10 seconds before polling again