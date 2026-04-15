from huggingface_hub import snapshot_download
import os

local_dir = "/mnt/beegfs/a.cardamone7/checkpoints_saving_folder/tinyvla/llava_pythia_libero_goal_no_noops_64/1.3B"

# Rimuovi directory esistente
import shutil
if os.path.exists(local_dir):
    shutil.rmtree(local_dir)

# Crea directory
os.makedirs(local_dir, exist_ok=True)

# Download
print(f"Scaricamento in corso in {local_dir}...")
snapshot_download(
    repo_id="lesjie/Llava-Pythia-1.3B",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    max_workers=1  # Evita race condition sui lock
)

print("Download completato!")
