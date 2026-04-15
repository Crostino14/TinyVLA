"""
verify_visual_embedding_tinyvla.py
====================================

Overview
--------
This diagnostic script investigates whether the **visual encoder** of TinyVLA
(CLIP-ViT-L/14-336 → mm_projector) produces significantly different embeddings
across the 10 tasks of the LIBERO Goal benchmark, or whether the projected patch
tokens are nearly identical for all tasks regardless of scene content.

This analysis is the TinyVLA counterpart of `verify_visual_embedding.py` written
for OpenVLA-OFT, and it addresses a core question in Vision-Language-Action (VLA)
model interpretability:

    "If the visual embeddings are nearly identical across tasks, then the model
     must rely almost exclusively on the language prompt to distinguish which
     action to execute — meaning that linguistic paraphrase quality (L1 vs L2
     vs L3 command abstraction levels) directly determines task performance."

Why This Matters for Ablation Studies
---------------------------------------
TinyVLA experiments evaluate the effect of alternative linguistic command phrasings
(levels L1, L2, L3) on task success rate. A prerequisite for attributing
performance differences to language is demonstrating that the visual encoder
does NOT produce task-discriminative features on its own. If the projected visual
patches were highly different between tasks, then performance could be explained
by visual features alone — making the linguistic ablation inconclusive.

This script provides the empirical evidence needed to validate or refute that
assumption.

Visual Encoding Pipeline in TinyVLA
--------------------------------------
TinyVLA (LLaVA-Pythia) processes two camera views at each inference step:

  1. **Primary view** (agentview): third-person scene overview
  2. **Wrist view** (robot0_eye_in_hand): close-up end-effector camera

Both views pass through the identical pipeline:

    RGB image (H, W, 3)
      → resize to (320, 180)
      → random crop-resize at 95%  (replicates training augmentation)
      → expand2square (pad to square with CLIP mean background)
      → image_processor.preprocess (CLIP normalization)
      → CLIP ViT-L/14-336 vision tower       → (1, 576, 1024)  [576 patch tokens, dim 1024]
      → mm_projector (2-layer MLP)            → (1, 576, 2048)  [projected to backbone dim]

After both views are processed, the patch token tensors are concatenated along
the sequence dimension (`token_cat` fusion):

    combined = cat([proj_primary, proj_wrist], dim=1)   → (1, 1152, 2048)

This combined tensor is then mean-pooled over the patch dimension to produce a
single fixed-size vector per image pair:

    patches_mean = combined.mean(dim=1)   → (2048,)

The `patches_mean` vector for each task is then compared pairwise using cosine
similarity and Euclidean distance to assess inter-task visual discriminability.

Sanity Check: Text Independence
---------------------------------
Unlike BLIP-2 or Flamingo-style models, TinyVLA's CLIP vision tower does NOT
receive any language tokens — it processes images entirely independently of the
text prompt. Therefore, running the same image pair through the visual encoder
with different text prompts must produce bit-for-bit identical projected patches.
The `sanity_check_text_independence()` function verifies this property holds at
runtime (checking for numerical identity up to 1e-5 tolerance). Deviations
would indicate unintended coupling (e.g., dropout layers not disabled in eval
mode, or BatchNorm statistics responding to batch composition).

Outputs
-------
The script prints to stdout:
  1. A 10×10 matrix of **cosine similarity** between task visual embeddings
     (mean-pooled combined patches). Values close to 1.0 indicate near-identical
     visual representations across tasks.
  2. A 10×10 matrix of **Euclidean distance** between the same embeddings.
     Low distances confirm the cosine similarity finding.
  3. **Off-diagonal statistics**: mean, std, min, max of both metrics over all
     10×9 = 90 off-diagonal pairs.
  4. **Automated interpretation**: textual diagnosis based on mean cosine similarity
     thresholds (>0.99 → primarily language-driven; 0.95-0.99 → mixed; <0.95 → visual).
  5. **Sanity check results**: confirmation that visual patches are text-independent.

No files are saved to disk; all output is printed to the console.

Architecture Compatibility
---------------------------
Tested on:
  - TinyVLA checkpoint-20000 (LoRA fine-tuned on LIBERO Goal, no no-ops)
  - Pythia-1.3B base model
  - CLIP-ViT-L/14-336 vision tower
  - mm_projector hidden_size = 2048 (matches GPT-NeoX backbone dimension)

Dependencies
------------
  - torch, torchvision  : GPU inference, image transforms
  - numpy, cv2          : Image resizing and numerical ops
  - libero              : LIBERO benchmark environments and task suite
  - llava_pythia        : LLaVA-Pythia model, config, builder, image processor
  - pathlib             : Robust cross-platform path construction

Usage
-----
Run directly (no CLI arguments needed — paths are hardcoded as constants):

    cd TinyVLA/test/libero_test
    python verify_visual_embedding_tinyvla.py
"""

import os       # Environment variable setting (TOKENIZERS_PARALLELISM, WANDB_DISABLED)
import sys      # sys.path manipulation for module resolution
import pickle   # (Imported but not used directly; available for potential data loading)
import numpy as np      # Numerical operations: matrix building, statistics, norm computation
import cv2              # OpenCV: image resizing (uint8 → 320×180 resize before tensor conversion)
import torch            # PyTorch: model inference, tensor operations, no_grad context
from pathlib import Path        # Object-oriented filesystem path construction and resolution
from torchvision import transforms      # ToTensor, Resize transforms for image preprocessing
from dataclasses import dataclass       # (Imported for potential configuration class; unused here)
from typing import List, Optional, Tuple  # PEP 484 type annotations for function signatures


# ─────────────────── Path Setup ───────────────────────────────────────────────

# Resolve the absolute path of this script file, regardless of the working directory
SCRIPT_DIR = Path(__file__).resolve().parent    # → .../TinyVLA/test/libero_test/

# Navigate two levels up from the script to reach the TinyVLA project root
TINYVLA_ROOT = SCRIPT_DIR.parent.parent          # → .../TinyVLA/

# Navigate to the sibling LIBERO project (assumes standard robosuite_test layout)
LIBERO_ROOT = TINYVLA_ROOT.parent.parent / "LIBERO"  # → .../robosuite_test/LIBERO/

# Insert all required roots into sys.path so that absolute imports resolve correctly.
# Using insert(0, ...) ensures these roots take priority over any conflicting packages
# that may be installed in the system environment.
for p in [str(TINYVLA_ROOT), str(SCRIPT_DIR), str(LIBERO_ROOT)]:
    if p not in sys.path:          # Avoid inserting duplicate entries
        sys.path.insert(0, p)      # Prepend to give highest import priority

# Disable Hugging Face tokenizer multi-threading to prevent fork-related deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable Weights & Biases experiment tracking (not needed for this analysis script)
os.environ["WANDB_DISABLED"] = "true"


# ── LLaVA-Pythia model imports ─────────────────────────────────────────────
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig
# LlavaPythiaConfig: HuggingFace PretrainedConfig subclass. Key attributes used here:
#   - config.hidden_size    : backbone embedding dimension (2048 for Pythia-1.3B)
#   - config.concat         : visual fusion strategy ("token_cat" in TinyVLA)
#   - config.action_head_type : action head type ("droid_diffusion")

from llava_pythia.model.builder import load_pretrained_model
# Loads tokenizer, model weights (backbone + LoRA adapters), image processor, context_len.

from llava_pythia.mm_utils import get_model_name_from_path
# Extracts a human-readable model name from the last path component of the checkpoint dir.

from llava_pythia.model import *  # noqa: F401,F403
# Wildcard import registers all custom nn.Module subclasses (vision towers, projectors,
# action heads) into PyTorch's model registry so load_pretrained_model can find them.

# ── LIBERO imports ─────────────────────────────────────────────────────────
from libero.libero import benchmark
# benchmark.get_benchmark_dict() returns the dict of all LIBERO task suite factories.

from libero_utils import (
    get_libero_env,           # Creates a MuJoCo OffScreenRenderEnv for a LIBERO task
    get_libero_dummy_action,  # Returns a 7D no-op action list [0,0,0,0,0,0,-1]
    get_libero_image,         # Extracts and flips agentview camera frame as uint8 numpy
    get_libero_wrist_image,   # Extracts and flips wrist camera frame as uint8 numpy
)


# ─────────────────── Constants ────────────────────────────────────────────────

# Absolute path to the TinyVLA LoRA checkpoint directory (step 20000)
CHECKPOINT_PATH = (
    "/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/"
    "checkpoints_saving_folder/tinyvla/"
    "post_processed_tiny_vla_llava_pythia_lora_libero_goal_no_noops_lora_r_64_processed/"
    "checkpoint-20000"
)

# Absolute path to the base Pythia-1.3B model (loaded before LoRA adapter application)
MODEL_BASE = (
    "/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/"
    "checkpoints_saving_folder/tinyvla/"
    "parte2_llava_pythia_libero_goal_no_noops_64/1.3B"
)

# Ordered list of LIBERO Goal task identifiers (0-indexed, matching task_suite ordering).
# These are the canonical task names used as row/column labels in the similarity matrices.
TASK_NAMES = [
    "put_the_wine_bottle_on_top_of_the_cabinet",    # Task 0: place wine bottle on cabinet top
    "open_the_top_drawer_and_put_the_bowl_inside",  # Task 1: open top drawer and insert bowl
    "turn_on_the_stove",                            # Task 2: toggle stove switch
    "put_the_bowl_on_top_of_the_cabinet",           # Task 3: place bowl on cabinet top
    "put_the_bowl_on_the_plate",                    # Task 4: place bowl on plate
    "put_the_wine_bottle_on_the_rack",              # Task 5: place wine bottle on rack
    "put_the_cream_cheese_in_the_bowl",             # Task 6: place cream cheese in bowl
    "open_the_middle_drawer_of_the_cabinet",        # Task 7: open middle cabinet drawer
    "push_the_plate_to_the_front_of_the_stove",     # Task 8: slide plate forward on stove
    "put_the_bowl_on_the_stove",                    # Task 9: place bowl on stove surface
]

# Four distinct language prompts used in the sanity check.
# These are different textual descriptions of the SAME visual scene.
# Since CLIP processes images independently of text, all four runs must
# produce numerically identical projected patches.
SANITY_PROMPTS = [
    "put the wine bottle on the top of the drawer",
    "open the middle layer of the drawer",
    "put the bowl on the stove",
    "place the wine bottle on the top of the drawer",
]

# Number of no-op stabilization steps before capturing the first frame.
# Allows MuJoCo physics to settle after object placement from set_init_state().
NUM_STEPS_WAIT = 10

# Camera rendering resolution (height = width = 256 pixels) for the simulation.
ENV_IMG_RES = 256

# Model family string passed to get_libero_env and get_libero_dummy_action.
# Controls environment action space and dummy action format.
MODEL_FAMILY = "tiny_vla"


# ─────────────────── Model Loading ────────────────────────────────────────────

def load_model():
    """
    Load the TinyVLA model (LLaVA-Pythia with LoRA adapters) and its image processor.

    This function calls `load_pretrained_model` to instantiate the full
    LLaVA-Pythia model (CLIP vision tower + GPT-NeoX backbone + diffusion action head)
    with LoRA adapter weights merged from the checkpoint. The model is immediately
    switched to evaluation mode to disable dropout and BatchNorm training behavior.

    The image processor (a Hugging Face `CLIPImageProcessor`) is used downstream to
    normalize image tensors with CLIP's per-channel mean and std before feeding
    them into the vision tower.

    Returns
    -------
    policy : LlavaPythiaForCausalLM
        The full TinyVLA model in eval mode on CUDA. Key sub-modules:
          - `policy.get_model().get_vision_tower()` : CLIP ViT-L/14-336
          - `policy.get_model().mm_projector`       : 2-layer MLP projector
          - `policy.config.hidden_size`             : 2048 (Pythia-1.3B backbone dim)
          - `policy.config.concat`                  : "token_cat" (visual fusion mode)
    image_processor : CLIPImageProcessor
        The image processor used during training. Provides:
          - `image_processor.image_mean` : [0.48145466, 0.4578275, 0.40821073]
            (CLIP per-channel mean, used as background padding color in expand2square)
          - `image_processor.image_std`  : [0.26862954, 0.26130258, 0.27577711]

    Notes
    -----
    The two `False` arguments to `load_pretrained_model` disable 8-bit and 4-bit
    quantization respectively, loading the model in full float32/float16 precision.
    The tokenizer and context_len are not needed for visual-only analysis and are
    discarded (assigned to `_`).
    """
    # Infer a human-readable model name from the checkpoint directory's last component
    model_name = get_model_name_from_path(CHECKPOINT_PATH)

    print(f"Loading TinyVLA model: {model_name}")
    print(f"  model_path: {CHECKPOINT_PATH}")
    print(f"  model_base: {MODEL_BASE}")

    # Load model components: tokenizer (discarded), policy model, image_processor, context_len (discarded)
    # False, False → disable 8-bit and 4-bit quantization (full precision loading)
    tokenizer, policy, image_processor, context_len = load_pretrained_model(
        CHECKPOINT_PATH, MODEL_BASE, model_name, False, False
    )

    policy.eval()   # Disable dropout and BatchNorm training mode for deterministic inference

    # Log key model configuration attributes for verification
    print(f"✓ Model loaded  (hidden_size={policy.config.hidden_size})")
    print(f"  visual_concat = {policy.config.concat}")          # Expected: "token_cat"
    print(f"  action_head   = {policy.config.action_head_type}")  # Expected: "droid_diffusion"

    return policy, image_processor


# ─────────────────── Image Preprocessing ──────────────────────────────────────

# Module-level ToTensor transform: converts HWC uint8 numpy → CHW float32 in [0,1]
# Defined at module level to avoid re-instantiating the transform object per call
_TO_TENSOR = transforms.ToTensor()

# Crop ratio for the random crop-resize augmentation.
# Keeps 95% of each spatial dimension (removes 2.5% from each side),
# matching the training-time data augmentation in eval_libero.py.
_RAND_CROP_RATIO = 0.95


def preprocess_image_pair(
    img_np: np.ndarray,
    wrist_np: np.ndarray,
    image_processor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess a pair of raw camera images into the format expected by the CLIP
    vision tower, exactly replicating the pipeline used during training in
    `eval_libero.process_batch_to_llava` for the `droid_diffusion` action head.

    Preprocessing Pipeline (per image)
    ------------------------------------
    1. **Resize to (320, 180)**:
       MuJoCo renders at resolution ENV_IMG_RES × ENV_IMG_RES. The training pipeline
       downsamples to (width=320, height=180) landscape format first, matching the
       aspect ratio of the original data collection cameras.

    2. **ToTensor**: converts the uint8 (H, W, C) numpy array to a float32 (C, H, W)
       tensor with values in [0.0, 1.0].

    3. **Random Crop-Resize at 95%** (`_RAND_CROP_RATIO = 0.95`):
       Symmetrically crop each spatial dimension to 95% of its size, then resize back
       to the original (180, 320) dimensions. This replicates the training data
       augmentation that was applied to every frame during fine-tuning. Applying the
       same augmentation at evaluation time ensures the visual distribution matches
       training (no distribution shift from augmentation mismatch).

    4. **expand2square**: Pad the (3, 180, 320) tensor to a square (3, 320, 320) tensor
       by adding equal padding on both sides of the shorter dimension. The padding color
       is CLIP's per-channel mean `image_processor.image_mean`, which was used during
       CLIP's own pre-training — using this specific color minimizes the activation
       magnitude of padded border regions in the vision encoder.

    5. **image_processor.preprocess**: Apply CLIP's normalization:
       `pixel = (pixel - mean) / std` using the official CLIP statistics.
       Returns a `(1, 3, 336, 336)` tensor — CLIP ViT-L/14 was trained on 336×336
       images, so the image processor resizes from 320 → 336 internally.

    Parameters
    ----------
    img_np : np.ndarray
        Primary camera (agentview) image as uint8 numpy array of shape (H, W, 3).
        Already flipped 180° by `get_libero_image` to correct MuJoCo's inverted Y-axis.
    wrist_np : np.ndarray
        Wrist camera (robot0_eye_in_hand) image as uint8 numpy array of shape (H, W, 3).
        Already flipped 180° by `get_libero_wrist_image`.
    image_processor : CLIPImageProcessor
        Hugging Face image processor providing `image_mean`, `image_std`, and the
        `preprocess()` method for final normalization.

    Returns
    -------
    image_tensor : torch.Tensor
        Preprocessed primary camera image of shape `(1, 3, 336, 336)` on CUDA,
        dtype float32, normalized to CLIP's expected input range.
    image_tensor_r : torch.Tensor
        Preprocessed wrist camera image of shape `(1, 3, 336, 336)` on CUDA,
        dtype float32.

    Notes
    -----
    The `device` variable computed from the image processor's class MRO is unused —
    it is a type-inspection artifact. Both output tensors are explicitly moved to
    CUDA via `.to("cuda", dtype=torch.float32)` in `_preprocess`.
    """
    # NOTE: `device` extraction below is a duck-type artifact and is not used further.
    # The actual device placement happens explicitly inside _preprocess().
    device = next(iter(image_processor.image_mean.__class__.__mro__), None)

    def _to_tensor_and_crop(np_img: np.ndarray) -> torch.Tensor:
        """
        Convert a uint8 numpy image to a cropped float tensor.

        Steps: resize (320×180) → to_tensor → rand_crop_resize (95%).

        Parameters
        ----------
        np_img : np.ndarray
            Input uint8 image of any resolution, shape (H, W, 3).

        Returns
        -------
        torch.Tensor
            Float tensor of shape (3, 180, 320) with values in [0.0, 1.0],
            after random crop-resize augmentation. Not yet normalized.
        """
        # Step 1: Resize to (width=320, height=180) — landscape training format
        resized = cv2.resize(np_img, (320, 180))  # cv2 uses (width, height) convention

        # Step 2: Convert (H, W, C) uint8 → (C, H, W) float32 in [0, 1]
        t = _TO_TENSOR(resized).float()   # Shape: (3, 180, 320)

        # Step 3: Random crop-resize at 95%
        orig_h, orig_w = t.shape[-2], t.shape[-1]  # h=180, w=320

        ratio = _RAND_CROP_RATIO   # 0.95 → crop to 95% of each spatial dimension

        # Compute symmetric pixel offsets: each edge loses (1 - ratio) / 2 of its dimension.
        # Example for height: strip int(180 * 0.025) = 4 pixels from top and bottom.
        t_crop = t[
            :,
            int(orig_h * (1 - ratio) / 2): int(orig_h * (1 + ratio) / 2),   # height crop
            int(orig_w * (1 - ratio) / 2): int(orig_w * (1 + ratio) / 2),   # width crop
        ]
        # t_crop shape: (3, ~171, ~304) — slightly smaller than the original

        # Resize the cropped tensor back to the original (180, 320) resolution
        resize_tf = transforms.Resize((orig_h, orig_w), antialias=True)
        t_out = resize_tf(t_crop)   # Shape: (3, 180, 320)

        return t_out

    # Apply the preprocessing to both camera views
    img_t   = _to_tensor_and_crop(img_np)    # (3, 180, 320)
    wrist_t = _to_tensor_and_crop(wrist_np)  # (3, 180, 320)

    # Prepare background color for square padding: CLIP's per-channel mean as Python floats
    bg_color = tuple(float(x) for x in image_processor.image_mean)
    # bg_color = (0.48145466, 0.4578275, 0.40821073) — CLIP's training mean

    def _expand2square(t: torch.Tensor) -> torch.Tensor:
        """
        Pad a (C, H, W) tensor to a square aspect ratio by centering
        and filling empty space with the CLIP mean background color.

        This is necessary because CLIP ViT-L/14-336 was trained on square
        (336×336) images. The (3, 180, 320) input is wider than tall, so
        equal padding is added to the top and bottom (height dimension).

        Parameters
        ----------
        t : torch.Tensor
            Input image tensor of shape (3, H, W) in float32.

        Returns
        -------
        torch.Tensor
            Square tensor of shape (1, max_dim, max_dim, 3) in HWC format
            (channels last) on the same device and dtype as input.
            The batch dimension (1) is added by unsqueeze(0) at the start.

        Notes
        -----
        The output is in HWC format (not CHW) because `image_processor.preprocess()`
        accepts either format and handles the conversion internally.

        Three cases:
          - H == W : already square, no padding needed.
          - H > W  : taller than wide → pad left and right (width dimension).
          - H < W  : wider than tall  → pad top and bottom (height dimension).
        """
        t = t.unsqueeze(0)             # (3, H, W) → (1, 3, H, W): add batch dim
        _, c, h, w = t.shape           # Unpack: batch=1, channels, height, width
        max_dim = max(h, w)            # Target square side length (= 320 for 180×320 input)

        # Allocate output array filled with background color.
        # Shape (1, max_dim, max_dim, c) = HWC format (channels last for numpy/image_processor)
        expanded = np.full((1, max_dim, max_dim, c), bg_color, dtype=np.float32)

        if h == w:
            # Already square: convert directly from BCHW tensor to BHWC numpy
            expanded = t.permute(0, 2, 3, 1).cpu().numpy()

        elif h > w:
            # Taller than wide: center content along width axis with symmetric padding
            offset = (max_dim - w) // 2   # Pixels of padding on each side of the width
            expanded[:, :h, offset:offset + w, :] = t.permute(0, 2, 3, 1).cpu().numpy()

        else:
            # Wider than tall (this case): center content along height axis
            offset = (max_dim - h) // 2   # Pixels of padding on each side of the height
            expanded[:, offset:offset + h, :w, :] = t.permute(0, 2, 3, 1).cpu().numpy()

        # Convert back to torch tensor, preserving the original dtype and device
        return torch.tensor(expanded, dtype=t.dtype)   # (1, max_dim, max_dim, c)

    # Apply square padding to both camera views
    img_sq   = _expand2square(img_t)    # (1, 320, 320, 3) — HWC format
    wrist_sq = _expand2square(wrist_t)  # (1, 320, 320, 3)

    def _preprocess(sq: torch.Tensor) -> torch.Tensor:
        """
        Apply CLIP normalization and move to CUDA.

        Calls `image_processor.preprocess()` with:
          - `do_normalize=True`    : Apply CLIP mean/std normalization
          - `do_rescale=False`     : Input is already in [0,1] (not uint8 0-255)
          - `do_center_crop=False` : Skip center crop (already square)

        Parameters
        ----------
        sq : torch.Tensor
            Square-padded image tensor of shape (1, max_dim, max_dim, 3).

        Returns
        -------
        torch.Tensor
            Normalized image tensor of shape `(1, 3, 336, 336)` on CUDA as float32.
            CLIP's image processor internally resizes to 336×336.
        """
        pv = image_processor.preprocess(
            sq,
            return_tensors="pt",    # Return PyTorch tensor (not numpy)
            do_normalize=True,      # Apply CLIP channel-wise normalization
            do_rescale=False,       # Skip [0,255] → [0,1] conversion (already float)
            do_center_crop=False,   # Skip cropping (already square-padded)
        )["pixel_values"]           # Extract the (1, 3, 336, 336) tensor from the output dict

        # Move to GPU as float32 for CUDA inference
        return pv.to("cuda", dtype=torch.float32)

    return _preprocess(img_sq), _preprocess(wrist_sq)


# ─────────────────── Visual Embedding Extraction ──────────────────────────────

def extract_visual_embedding(
    policy,
    image_processor,
    img_np: np.ndarray,
    wrist_np: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the projected visual patch embeddings for a dual-camera image pair.

    This function computes the `token_cat` visual embedding used internally by
    TinyVLA during inference: both camera images are passed through the CLIP
    vision tower and the mm_projector, then concatenated along the patch dimension.

    This represents the exact visual information that the GPT-NeoX backbone
    receives at each inference step, before any fusion with language tokens.

    Embedding Pipeline
    ------------------

    **Primary camera (agentview)**:

    \[
        f_\text{primary} = \text{mm\_projector}(\text{CLIP}(I_\text{primary}))
        \quad \in \mathbb{R}^{1 \times 576 \times 2048}
    \]

    **Wrist camera (robot0_eye_in_hand)**:

    \[
        f_\text{wrist} = \text{mm\_projector}(\text{CLIP}(I_\text{wrist}))
        \quad \in \mathbb{R}^{1 \times 576 \times 2048}
    \]

    **token_cat fusion**:

    \[
        f_\text{combined} = [f_\text{primary} \| f_\text{wrist}]_{\text{dim=1}}
        \quad \in \mathbb{R}^{1 \times 1152 \times 2048}
    \]

    **Mean pooling**:

    \[
        \bar{f} = \frac{1}{1152} \sum_{i=1}^{1152} f_{\text{combined},i}
        \quad \in \mathbb{R}^{2048}
    \]

    The mean-pooled vector \(\bar{f}\) is used for the cross-task similarity
    analysis. The raw patch array `patches_flat` is returned for potential
    downstream analysis (e.g., patch-level similarity).

    Parameters
    ----------
    policy : LlavaPythiaForCausalLM
        The loaded TinyVLA model in eval mode. Must be on CUDA.
    image_processor : CLIPImageProcessor
        CLIP image processor for normalization, passed through to
        `preprocess_image_pair`.
    img_np : np.ndarray
        Primary camera image as uint8 numpy array, shape (H, W, 3).
    wrist_np : np.ndarray
        Wrist camera image as uint8 numpy array, shape (H, W, 3).

    Returns
    -------
    patches_flat : np.ndarray
        Raw combined patch embeddings as float32 numpy array,
        shape `(2 * n_patches, hidden_dim)` = `(1152, 2048)`.
        Contains the per-patch projected features from both cameras.
    patches_mean : np.ndarray
        Mean-pooled combined embedding as float32 numpy array,
        shape `(hidden_dim,)` = `(2048,)`.
        Used as the representative visual embedding for this image pair.

    Notes
    -----
    `torch.no_grad()` is used (not `torch.inference_mode()`) because this
    is a short forward pass on a small model — the difference in overhead
    is negligible and `no_grad` is slightly more compatible with modules
    that check `torch.is_grad_enabled()`.

    `.detach().cpu().float()` chain ensures:
      - `.detach()` : removes from computation graph (no gradient tracking)
      - `.cpu()`    : transfers from VRAM to system RAM
      - `.float()`  : converts from model dtype (may be float16) to float32
                      for consistent downstream distance computations
    """
    with torch.no_grad():  # Disable gradient computation for pure inference
        # Preprocess both images: resize, crop, square-pad, normalize → (1, 3, 336, 336) CUDA
        img_t, wrist_t = preprocess_image_pair(img_np, wrist_np, image_processor)

        # Access sub-modules of the LLaVA-Pythia model:
        # get_model() returns the language model core (GPT-NeoX + projector + vision tower)
        vision_tower = policy.get_model().get_vision_tower()   # CLIP ViT-L/14-336
        mm_projector  = policy.get_model().mm_projector         # 2-layer MLP (1024 → 2048)

        # ── Primary camera forward pass ───────────────────────────────────
        feats = vision_tower(img_t)    # CLIP ViT-L/14-336: (1, 576, 1024) — 576 patch tokens
        proj  = mm_projector(feats)    # MLP projector:     (1, 576, 2048) — projected to backbone dim

        # ── Wrist camera forward pass ─────────────────────────────────────
        feats_r = vision_tower(wrist_t)   # (1, 576, 1024)
        proj_r  = mm_projector(feats_r)   # (1, 576, 2048)

        # ── token_cat fusion: concatenate along patch sequence dimension ──
        # This is the `visual_concat='token_cat'` mode defined in config.concat
        # Result: language backbone sees 1152 visual tokens total (576 per camera)
        combined = torch.cat([proj, proj_r], dim=1)  # (1, 1152, 2048)

        # Squeeze batch dim, move to CPU as float32 numpy
        patches_flat = (
            combined.squeeze(0)    # (1, 1152, 2048) → (1152, 2048)
            .detach()              # Detach from computation graph
            .cpu()                 # Move from VRAM to RAM
            .float()               # Ensure float32 (model may run in float16)
            .numpy()               # Convert to numpy for distance computations
        )  # Shape: (1152, 2048)

        # Compute the mean over all 1152 patch positions → single representative vector
        patches_mean = patches_flat.mean(axis=0)   # (2048,)

    return patches_flat, patches_mean


# ─────────────────── First Frame Extraction ───────────────────────────────────

def get_first_frame(
    task,
    task_id: int,
    task_suite,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize a LIBERO task environment, run physics stabilization, and return
    the first stabilized observation frame from both cameras.

    This function encapsulates the full environment lifecycle for single-frame
    extraction: creation → seeding → reset → initial state → stabilization → capture → close.

    The returned images represent the task's initial visual state (e.g., "bowl
    on table, stove to the left") that TinyVLA would see at the beginning of an
    episode. Analyzing visual embeddings from these frames captures the inter-task
    variability of initial scenes without the confound of different robot trajectories.

    Parameters
    ----------
    task : libero.libero.benchmark.Task
        LIBERO task object obtained from `task_suite.get_task(task_id)`.
        Provides the BDDL file path, initial state configurations, and task metadata.
    task_id : int
        0-indexed task identifier, used to retrieve the fixed initial states
        from `task_suite.get_task_init_states(task_id)`.
    task_suite : libero.libero.benchmark.TaskSuite
        Instantiated LIBERO benchmark task suite (e.g., `libero_goal`).
        Used to retrieve the pre-saved initial state snapshots.

    Returns
    -------
    img_np : np.ndarray
        Primary camera (agentview) frame as uint8 numpy array, shape (H, W, 3).
        Already 180°-flipped by `get_libero_image` to correct MuJoCo inversion.
    wrist_np : np.ndarray
        Wrist camera (robot0_eye_in_hand) frame as uint8 numpy array, shape (H, W, 3).
        Already 180°-flipped by `get_libero_wrist_image`.

    Notes
    -----
    Environment Seed:
      `env.seed(0)` is called before reset to ensure identical initial physics
      noise across all tasks, making the comparison fair.

    Fallback Reset:
      If `env.set_init_state(initial_states[0])` fails (e.g., due to MuJoCo
      version incompatibility or missing state file), the function falls back to
      a default `env.reset()` + `env.get_observation()` call. This ensures the
      script continues even with missing initial state data.

    Stabilization:
      `NUM_STEPS_WAIT = 10` dummy no-op steps are executed to allow MuJoCo
      physics to settle after initial state placement. Without this, the first
      observation may contain floating or vibrating objects.

    The environment is always closed via `env.close()` to release OpenGL/MuJoCo
    resources, even if an exception occurred during setup.
    """
    # Create the LIBERO simulation environment with the specified resolution.
    # change_command=False uses the default BDDL language command for this task.
    # The returned task_description and extra info (_) are not needed here.
    env, _, _ = get_libero_env(
        task, MODEL_FAMILY,
        change_command=False,    # Use default BDDL command (we only need the image)
        resolution=ENV_IMG_RES   # Render at 256×256 pixels
    )

    env.seed(0)   # Fix MuJoCo random seed for reproducible physics noise

    try:
        # Retrieve the pre-saved deterministic initial state for this task (index 0)
        initial_states = task_suite.get_task_init_states(task_id)
        env.reset()                                     # Required before set_init_state on some versions
        obs = env.set_init_state(initial_states[0])     # Place objects at saved positions
    except Exception:
        # Fallback: if initial state loading fails, use the default random reset
        env.reset()
        obs = env.get_observation()    # Get current observation after random reset

    # Run stabilization steps: send no-op actions to let MuJoCo physics settle
    for _ in range(NUM_STEPS_WAIT):
        # get_libero_dummy_action("tiny_vla") returns [0, 0, 0, 0, 0, 0, -1]
        # (zero delta position, zero rotation, gripper open)
        obs, _, _, _ = env.step(get_libero_dummy_action(MODEL_FAMILY))

    # Extract both camera frames from the final stabilized observation
    img_np   = get_libero_image(obs)        # Agentview: (H, W, 3) uint8, 180°-flipped
    wrist_np = get_libero_wrist_image(obs)  # Wrist cam: (H, W, 3) uint8, 180°-flipped

    env.close()   # Release MuJoCo and OpenGL resources immediately after frame capture

    return img_np, wrist_np


# ─────────────────── Distance Utilities ───────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two embedding vectors.

    Cosine similarity measures the angle between two vectors in the embedding
    space, normalized by their magnitudes. It is bounded in [-1, 1]:
      - 1.0 : perfectly identical direction (same embedding up to scaling)
      - 0.0 : orthogonal (no linear correlation)
      - -1.0: perfectly opposite directions

    For visual embeddings from a CLIP vision tower, values above 0.95 indicate
    that the two embeddings are nearly indistinguishable, meaning the model
    encodes the two corresponding scenes with nearly the same visual representation.

    The formula is:

    \[
        \text{cos\_sim}(a, b) =
        \frac{a \cdot b}{\|a\|_2 \cdot \|b\|_2}
    \]

    Parameters
    ----------
    a : np.ndarray
        First embedding vector. Any shape, but typically `(hidden_dim,)` = `(2048,)`.
    b : np.ndarray
        Second embedding vector. Must have the same shape as `a`.

    Returns
    -------
    float
        Cosine similarity in [-1.0, 1.0].

    Notes
    -----
    A small epsilon (1e-12) is added to the norm before division to prevent
    division-by-zero for the zero vector. This is a standard numerical safeguard.
    """
    # L2-normalize both vectors: divide by their respective norms + epsilon guard
    a = a / (np.linalg.norm(a) + 1e-12)   # Unit vector in direction of a
    b = b / (np.linalg.norm(b) + 1e-12)   # Unit vector in direction of b

    # Dot product of two unit vectors equals cosine of the angle between them
    return float(np.dot(a, b))


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Euclidean (L2) distance between two embedding vectors.

    Euclidean distance measures the straight-line distance between two points
    in the embedding space:

    \[
        d(a, b) = \|a - b\|_2 = \sqrt{\sum_{i=1}^{d}(a_i - b_i)^2}
    \]

    Unlike cosine similarity, Euclidean distance is sensitive to the MAGNITUDE
    of the embedding vectors, not just their direction. For normalized embeddings,
    cosine similarity and Euclidean distance encode the same information, but
    for un-normalized vectors (as used here), they can differ.

    A low Euclidean distance between two visual embeddings confirms that the
    CLIP + projector pipeline produces similar activation patterns for both scenes,
    complementing the cosine similarity analysis.

    Parameters
    ----------
    a : np.ndarray
        First embedding vector, shape `(hidden_dim,)`.
    b : np.ndarray
        Second embedding vector, shape `(hidden_dim,)`.

    Returns
    -------
    float
        L2 distance ≥ 0. Values close to 0 indicate nearly identical embeddings.
    """
    return float(np.linalg.norm(a - b))   # Equivalent to np.sqrt(np.sum((a - b) ** 2))


# ─────────────────── Pretty Print Utilities ───────────────────────────────────

def short(name: str, n: int = 18) -> str:
    """
    Create a compact abbreviated display name from a task name string.

    Converts an underscore-separated task name into a space-separated abbreviation
    by taking the first 4 characters of each word. This produces short labels that
    fit in fixed-width matrix columns while remaining recognizable.

    Example
    -------
    `short("put_the_wine_bottle_on_top_of_the_cabinet")`
        → "put the wine bott on top of the cabi"[:18]
        → "put the wine bott "

    Parameters
    ----------
    name : str
        Task name string with words separated by underscores
        (e.g., "put_the_wine_bottle_on_top_of_the_cabinet").
    n : int, optional
        Maximum number of characters in the output string. Default: 18.

    Returns
    -------
    str
        Abbreviated display name, truncated to `n` characters.
    """
    words = name.replace("_", " ").split()      # Convert underscores to spaces, split into words
    abbr = " ".join(w[:4] for w in words)       # Take first 4 chars of each word, rejoin with spaces
    return abbr[:n]                             # Truncate to maximum display width


def print_matrix(
    matrix: np.ndarray,
    labels: List[str],
    title: str,
    fmt: str = "{:.4f}",
):
    """
    Print a symmetric similarity or distance matrix to stdout in a formatted table.

    Produces a readable fixed-width table with abbreviated row labels and numeric
    column indices. Designed for 10×10 matrices but works for any square matrix.

    Layout Example (n=3)
    --------------------
    ─────────────────────────────────────────────
      COSINE SIMILARITY between visual embeddings
    ─────────────────────────────────────────────
                               0          1          2
       0 put  the win    1.0000     0.9987     0.9991
       1 open the top    0.9987     1.0000     0.9993
       2 turn on th      0.9991     0.9993     1.0000

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix of shape `(n, n)` containing the values to display.
        Typically cosine similarities (float in [-1, 1]) or Euclidean distances
        (float ≥ 0).
    labels : List[str]
        List of `n` row/column label strings (task names). Each label is
        abbreviated to 18 characters via `short()` for column-width alignment.
    title : str
        Descriptive title string printed above the matrix in the header.
    fmt : str, optional
        Python format string for formatting each matrix cell value.
        Default: "{:.4f}" (4 decimal places, appropriate for cosine similarity).
        Use "{:.2f}" for Euclidean distances.
    """
    n = len(labels)
    col_w = 10   # Width of each numeric value column in characters
    lbl_w = 22   # Width of the row label column (index + abbreviated name)

    # Total table width: label column + n numeric columns + margins
    width = n * col_w + lbl_w + 4

    # Print table borders and title
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")

    # Print column header: right-aligned numeric indices (0..n-1)
    header = f"{'':>{lbl_w}}"          # Left-pad empty string for label column
    for i in range(n):
        header += f" {i:>{col_w - 1}}"  # Right-align index in (col_w-1) chars
    print(header)

    # Print each data row: index + abbreviated label + formatted values
    for i in range(n):
        # Row label: "  {index:2d} {abbreviated_name:<(lbl_w-4)}"
        row = f"  {i:2d} {short(labels[i]):<{lbl_w - 4}}"

        for j in range(n):
            # Right-align each formatted value in (col_w-1) character width
            row += f" {fmt.format(matrix[i, j]):>{col_w - 1}}"

        print(row)


# ─────────────────── Sanity Check ─────────────────────────────────────────────

def sanity_check_text_independence(
    policy,
    image_processor,
    img_np: np.ndarray,
    wrist_np: np.ndarray,
):
    """
    Verify that TinyVLA's visual encoder produces identical patch embeddings
    regardless of which text prompt is used — confirming that CLIP does NOT
    condition on language.

    This sanity check validates a fundamental architectural property of TinyVLA:
    unlike cross-attention-based VLMs (e.g., Flamingo), CLIP's vision tower
    processes images with no access to language tokens. Therefore, passing the
    EXACT same image pair through `extract_visual_embedding` multiple times —
    even with different associated language prompts — must produce bit-for-bit
    identical `patches_flat` tensors.

    Deviations from exact identity (above 1e-5 tolerance) would indicate one
    of the following problems:
      - A **dropout layer** not properly disabled by `model.eval()` is active
        inside the vision tower or mm_projector.
      - A **BatchNorm layer** is still updating running statistics (should be
        frozen in eval mode).
      - A **stochastic operation** (e.g., random noise injection) is present
        in the preprocessing pipeline.

    Why This Matters
    ----------------
    If visual patches were NOT identical across runs, then the visual embedding
    analysis would be confounded: apparent inter-task differences in visual
    similarity could partly reflect intra-run variance rather than true
    semantic scene differences. Confirming text independence and determinism
    validates the entire visual embedding analysis methodology.

    Parameters
    ----------
    policy : LlavaPythiaForCausalLM
        The loaded TinyVLA model. Must be in eval mode (policy.eval() already called).
    image_processor : CLIPImageProcessor
        CLIP image processor for preprocessing, passed to `extract_visual_embedding`.
    img_np : np.ndarray
        Primary camera frame to test, shape (H, W, 3) uint8.
        Task 0's first frame is used in the main function.
    wrist_np : np.ndarray
        Wrist camera frame to test, shape (H, W, 3) uint8.

    Notes
    -----
    The function runs `len(SANITY_PROMPTS)` = 4 independent forward passes through
    `extract_visual_embedding`. The text prompts in `SANITY_PROMPTS` are NOT passed
    to the visual encoder (CLIP has no text input path), but the variable name
    `run_idx` mirrors the convention of iterating over different prompts to make
    the test's intent clear to the reader.

    `np.allclose(ref, flat, atol=1e-5)`:
      Checks that all elements differ by less than 1e-5 in absolute value.
      This tolerance accounts for float16 → float32 conversion rounding
      and is sufficiently tight to detect any meaningful numerical divergence.
    """
    print("\n" + "=" * 80)
    print("SANITY CHECK: same image → identical visual embeddings for every run?")
    print("(CLIP encoder never receives text: result must be bit-for-bit identical)")
    print("=" * 80)

    results = []   # Accumulate patch arrays from each independent forward pass

    # Run the visual encoder once per sanity prompt (prompts are irrelevant to CLIP,
    # but the iteration mirrors the text-variation structure of the main analysis)
    for run_idx in range(len(SANITY_PROMPTS)):
        flat, _ = extract_visual_embedding(
            policy, image_processor, img_np, wrist_np
        )   # Returns (1152, 2048) numpy array
        results.append(flat)
        print(f"  run {run_idx + 1:2d}: patches shape = {flat.shape}")

    ref = results[0]        # Use the first run as the reference for comparison
    all_identical = True    # Track whether ALL subsequent runs matched the reference

    # Compare each subsequent run against the reference run
    for i, flat in enumerate(results[1:], 1):
        # Compute maximum absolute difference across all patch × dimension elements
        max_diff = float(np.abs(ref - flat).max())

        # Check element-wise equality within 1e-5 absolute tolerance
        identical = np.allclose(ref, flat, atol=1e-5)

        # Update global flag: one non-identical run fails the sanity check
        all_identical = all_identical and identical

        status = "✓ IDENTICAL" if identical else "✗ DIFFERENT"
        print(f"\n  Run 0 vs Run {i}: {status}  max|diff| = {max_diff:.2e}")

    # Print overall conclusion
    if all_identical:
        print(
            "\n  ✓ CONFIRMED: visual block is deterministic and text-independent"
        )
    else:
        print(
            "\n  ✗ WARNING: results are not identical across runs — "
            "check for active dropout or BatchNorm in eval mode"
        )


# ─────────────────── Main ─────────────────────────────────────────────────────

def main():
    """
    Execute the complete visual embedding analysis for TinyVLA on LIBERO Goal.

    Orchestrates the following pipeline:

    1. **Model Loading**: Load TinyVLA checkpoint-20000 with Pythia-1.3B base.
    2. **Task Suite Loading**: Instantiate LIBERO Goal benchmark (10 tasks).
    3. **First Frame Extraction**: For each task, initialize the environment
       and capture the first stabilized frame from both cameras.
    4. **Visual Embedding Extraction**: Pass each frame pair through CLIP +
       mm_projector and collect the mean-pooled (2048,) embedding.
    5. **Similarity Matrix Computation**: Compute all 10×10 pairwise cosine
       similarities and Euclidean distances.
    6. **Matrix Printing**: Display formatted tables via `print_matrix`.
    7. **Statistical Summary**: Compute and display statistics over off-diagonal
       pairs (excluding self-similarity on the diagonal).
    8. **Automated Interpretation**: Classify the inter-task visual similarity
       regime based on mean cosine similarity thresholds.
    9. **Sanity Check**: Verify that visual patches are text-independent and
       deterministic across repeated forward passes.

    All results are printed to stdout; no files are written to disk.

    Interpretation Thresholds
    --------------------------
    The function applies three interpretability regimes based on mean off-diagonal
    cosine similarity:

    - **> 0.99** (very high similarity):
      Visual embeddings are nearly indistinguishable across tasks. The model's
      GPT-NeoX backbone cannot use visual features to discriminate tasks —
      it must rely exclusively on the language prompt. Performance differences
      across L1/L2/L3 command levels are attributable to the linguistic component.

    - **0.95 – 0.99** (high similarity):
      Visual embeddings are similar but not identical. Language dominates
      but visual features may provide some supplementary task-discriminating signal.

    - **< 0.95** (moderate similarity):
      Visual embeddings meaningfully differ between tasks. Both visual and
      linguistic features contribute to task discrimination.
    """
    # ── Stage 1: Model Loading ──────────────────────────────────────────────
    policy, image_processor = load_model()   # Load TinyVLA model + CLIP image processor

    # ── Stage 2: Task Suite Loading ─────────────────────────────────────────
    benchmark_dict = benchmark.get_benchmark_dict()   # All LIBERO task suite factories
    task_suite     = benchmark_dict["libero_goal"]()  # Instantiate LIBERO Goal (10 tasks)
    n_tasks        = task_suite.n_tasks               # = 10

    # Print analysis header with model configuration summary
    print(f"\n{'=' * 80}")
    print(f"VISUAL EMBEDDING ANALYSIS — TinyVLA checkpoint 20000 — LIBERO Goal")
    print(f"{'=' * 80}")
    print(f"visual_concat  = {policy.config.concat}")         # Expected: "token_cat"
    print(f"CLIP hidden    = 1024  →  projector output (hidden_size) = {policy.config.hidden_size}")
    print(f"Num tasks      = {n_tasks}")
    print(f"{'=' * 80}\n")

    # ── Stage 3 & 4: Frame Extraction + Embedding Extraction ───────────────
    mean_embeddings: List[np.ndarray] = []  # List of (2048,) mean-pooled vectors, one per task
    flat_embeddings: List[np.ndarray] = []  # List of (1152, 2048) raw patch arrays, one per task
    task_labels:     List[str] = []         # Task name strings for matrix axis labels

    for task_id in range(n_tasks):
        task     = task_suite.get_task(task_id)   # Retrieve task object by index
        task_key = TASK_NAMES[task_id]            # Human-readable name from TASK_NAMES constant
        task_labels.append(task_key)              # Register for matrix printing

        print(f"[{task_id:2d}] {task_key}")

        # Initialize the environment, run stabilization, capture first frame
        img_np, wrist_np = get_first_frame(task, task_id, task_suite)

        # Extract visual embedding: CLIP → projector → token_cat → mean_pool
        flat, mean = extract_visual_embedding(
            policy, image_processor, img_np, wrist_np
        )

        flat_embeddings.append(flat)   # (1152, 2048) — raw patches for potential future use
        mean_embeddings.append(mean)   # (2048,)       — used for pairwise similarity

        print(f"     combined patches shape: {flat.shape}  |  mean shape: {mean.shape}")

    # ── Stage 5: Pairwise Similarity Matrix Construction ───────────────────
    n = n_tasks

    # Allocate square matrices for cosine similarity and Euclidean distance
    cos_matrix = np.zeros((n, n))   # Will be filled with values in [-1, 1]
    euc_matrix = np.zeros((n, n))   # Will be filled with values ≥ 0

    for i in range(n):
        for j in range(n):
            # Diagonal (i==j): cosine_sim = 1.0, euclidean_dist = 0.0 (self-similarity)
            cos_matrix[i, j] = cosine_sim(mean_embeddings[i], mean_embeddings[j])
            euc_matrix[i, j] = euclidean_dist(mean_embeddings[i], mean_embeddings[j])

    # ── Stage 6: Matrix Printing ─────────────────────────────────────────
    print_matrix(
        cos_matrix, task_labels,
        "COSINE SIMILARITY between visual embedding (mean-pool, token_cat) — 10 tasks",
        fmt="{:.4f}",   # 4 decimal places for cosine similarity
    )
    print_matrix(
        euc_matrix, task_labels,
        "EUCLIDEAN DISTANCE between visual embedding (mean-pool, token_cat) — 10 tasks",
        fmt="{:.2f}",   # 2 decimal places for distance (values may be large)
    )

    # ── Stage 7: Off-Diagonal Statistics ─────────────────────────────────
    # Build a boolean mask that excludes the diagonal (self-similarity entries)
    mask    = ~np.eye(n, dtype=bool)   # True everywhere EXCEPT the diagonal
    off_cos = cos_matrix[mask]          # Flattened array of n*(n-1) = 90 off-diagonal similarities
    off_euc = euc_matrix[mask]          # Flattened array of n*(n-1) = 90 off-diagonal distances

    print(f"\n{'─' * 80}")
    print(f"STATISTICS (off-diagonal pairs, N={n * (n - 1)})")

    # Cosine similarity statistics: values close to 1.0 → near-identical visual reps
    print(
        f"  Cosine Similarity:   "
        f"mean={off_cos.mean():.4f}  std={off_cos.std():.4f}  "
        f"min={off_cos.min():.4f}  max={off_cos.max():.4f}"
    )
    # Euclidean distance statistics: values close to 0.0 → near-identical visual reps
    print(
        f"  Euclidean Distance:  "
        f"mean={off_euc.mean():.2f}   std={off_euc.std():.2f}   "
        f"min={off_euc.min():.2f}   max={off_euc.max():.2f}"
    )

    # ── Stage 8: Automated Interpretation ────────────────────────────────
    mean_cos = off_cos.mean()   # Single scalar summarizing inter-task visual similarity

    print(f"\n{'─' * 80}")
    print("INTERPRETATION:")

    if mean_cos > 0.99:
        # Very high similarity: model must rely on language to distinguish tasks
        print("  ● Very high cosine similarity (>0.99): visual embeddings are nearly identical")
        print("    across all tasks. The model relies PRIMARILY on the text prompt.")
        print("    → Performance differences across L1/L2/L3 are attributable to the")
        print("      LINGUISTIC component, not the visual one.")
    elif mean_cos > 0.95:
        # High similarity: language dominates, but vision contributes marginally
        print("  ● High cosine similarity (>0.95): visual block produces similar but")
        print("    not identical representations. Language plays a dominant role.")
    else:
        # Moderate similarity: visual features are task-discriminative
        print("  ● Moderate cosine similarity: visual block differentiates between tasks.")
        print("    Both visual and linguistic components contribute to task discrimination.")

    # ── Stage 9: Sanity Check ─────────────────────────────────────────────
    # Use task 0's frame for the text-independence sanity check
    task0        = task_suite.get_task(0)
    img0, wrist0 = get_first_frame(task0, 0, task_suite)

    # Verify CLIP is deterministic and text-independent across multiple forward passes
    sanity_check_text_independence(policy, image_processor, img0, wrist0)

    print(f"\n{'=' * 80}")
    print("Analysis complete.")


# ─────────────────── Entry Point ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Script entry point when executed directly from the command line.

    This script has no CLI arguments — all configuration (model paths, task suite,
    constants) is hardcoded at the top of the file. Run as:

        cd TinyVLA/test/libero_test
        python verify_visual_embedding_tinyvla.py

    Expected Output:
        - Model loading confirmation with config attributes
        - Per-task progress log during frame extraction
        - 10×10 cosine similarity matrix (values expected > 0.99 for LIBERO Goal)
        - 10×10 Euclidean distance matrix
        - Off-diagonal statistics summary
        - Automated interpretation of visual similarity regime
        - Sanity check confirmation (✓ CONFIRMED for a correctly configured model)

    Prerequisites:
        - CUDA GPU with sufficient VRAM to load TinyVLA (≥ 8 GB recommended)
        - checkpoint-20000 and 1.3B base model at the hardcoded paths
        - LIBERO environment with MuJoCo renderer available
    """
    main()