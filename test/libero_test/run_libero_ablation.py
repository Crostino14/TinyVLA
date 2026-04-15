"""
run_libero_ablation.py  (TinyVLA version)
==========================================

Overview
--------
This script implements an ablation study that tests whether the TinyVLA
Vision-Language-Action (VLA) policy genuinely understands full natural language
task descriptions, or instead relies on superficial keyword shortcuts to complete
manipulation tasks.

The core idea is to replace the original task instruction (e.g.,
"Turn on the stove") with a deliberately shortened or re-ordered variant
(e.g., just "stove", or "bowl stove") encoded in a custom BDDL file, and then
measure whether the policy still succeeds. If success rates are comparable
across all keyword variants — including near-empty commands — it suggests the
model ignores language and acts as a vision-only policy. If success rates drop
with degraded language, it indicates genuine language grounding.

This kind of linguistic robustness probing is directly motivated by findings
in VLA robustness literature [web:16], where models like OpenVLA-OFT were shown
to behave as "Vision-Action" models that disregard language entirely in
short-horizon tasks.

Supported Tasks
---------------
The ablation is currently configured for two libero_goal tasks:
  - Task 8 (task_id=7): "Turn on the stove"
      → Variants: single keyword "stove", multi-keyword "bowl stove",
        "plate stove", action-only "Turn on"
  - Task 9 (task_id=8): "Put the bowl on the plate"
      → Variants: "bowl", "plate", "bowl plate", "Put plate", "plate bowl"

Each variant has a corresponding custom BDDL file that encodes the degraded
command as the task's language description.

Architecture
------------
The policy under test is `llava_pythia_act_policy`, wrapping LLaVA-Pythia
(a Pythia-backbone VLM with a diffusion or ACT action head). At each step, the
model receives:
  - Front-view camera image (agentview)
  - Wrist camera image (robot0_eye_in_hand)
  - A (possibly degraded) language instruction
  - Normalized robot proprioceptive state

Inference uses temporal aggregation: overlapping action chunks are averaged
with exponential recency weights for smooth execution.

Adapted From
------------
Originally derived from the OpenVLA evaluation pipeline and adapted to use
TinyVLA's LlavaPythia backbone and droid_diffusion action head.

Dependencies
------------
  - libero        : Robotic manipulation benchmark (tasks, environments)
  - llava_pythia  : LLaVA-Pythia multimodal VLA model
  - torch         : PyTorch deep learning framework
  - torchvision   : Image transforms
  - draccus       : Dataclass-based CLI argument parser
  - cv2 (OpenCV)  : Image preprocessing
  - imageio       : Video recording
  - tqdm          : Progress bars
  - wandb         : (Optional) Weights & Biases experiment tracking

Usage
-----
    python run_libero_ablation.py \
        --model_path /path/to/checkpoint \
        --model_base /path/to/base_model \
        --ablation_task_id 7 \
        --ablation_test_key stove1 \
        --num_trials_per_task 50 \
        --seed 42
"""

import sys
import os
import logging
import json
import time
import pickle
import cv2
import numpy as np
import torch
import tqdm

from collections import deque       # Double-ended queue (available for temporal buffering)
from dataclasses import dataclass   # Standard Python dataclass decorator
from enum import Enum               # Enumeration base class for task suite names
from pathlib import Path            # Object-oriented filesystem path utilities
from typing import Optional, Union  # PEP 484 type hints

from torchvision import transforms  # Image transformation pipeline (ToTensor, Resize, etc.)

# Disable HuggingFace tokenizer parallelism to prevent deadlocks in forked processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Force the compute device to CUDA (GPU)
os.environ["DEVICE"] = "cuda"
# Disable W&B at OS level (can be re-enabled via cfg.use_wandb)
os.environ["WANDB_DISABLED"] = "true"

# ── TinyVLA / LlavaPythia imports ─────────────────────────────────────────────
# Core model configuration class for LLaVA-Pythia
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig
# Conversation template registry (maps mode names like "pythia" to prompt formats)
from llava_pythia.conversation import conv_templates, SeparatorStyle
# Utility to load a pretrained LLaVA-Pythia checkpoint (handles LoRA and full models)
from llava_pythia.model.builder import load_pretrained_model
# Tokenization and utility helpers for the LLaVA-Pythia model
from llava_pythia.mm_utils import (
    tokenizer_image_token,          # Tokenizes a prompt, inserting image token placeholders
    get_model_name_from_path,       # Extracts a human-readable model name from a checkpoint path
    KeywordsStoppingCriteria,       # Stop generation when certain keywords appear in output
)
# Special token constants used to format prompts with image placeholders
from llava_pythia.constants import (
    IMAGE_TOKEN_INDEX,              # Integer index used as a placeholder for image embeddings
    DEFAULT_IMAGE_TOKEN,            # String token "<image>" marking image injection point
    DEFAULT_IM_START_TOKEN,         # Optional "<im_start>" wrapper for image token
    DEFAULT_IM_END_TOKEN,           # Optional "<im_end>" wrapper for image token
)
from llava_pythia.model import *    # Wildcard import of all model classes and utilities

# Custom PyTorch utilities (e.g., rot_6d_to_euler_angles for action conversion)
import models.TinyVLA.test.utils.torch_utils as TorchUtils

import draccus    # Dataclass-aware CLI parser (similar to simple-parsing / Hydra)
import wandb      # Weights & Biases for optional remote experiment tracking

# LIBERO benchmark registry (provides access to task suites and individual tasks)
from libero.libero import benchmark

# Custom LIBERO utility functions for this TinyVLA evaluation pipeline
from models.TinyVLA.test.utils.libero_utils import (
    get_libero_dummy_action,    # Returns a no-op action for physics stabilization
    get_libero_env,             # Creates a LIBERO environment (supports custom BDDL override)
    get_libero_image,           # Extracts the agentview camera image from obs dict
    get_libero_wrist_image,     # Extracts the wrist camera image from obs dict
    quat2axisangle,             # Converts quaternion orientation to axis-angle representation
    save_rollout_video,         # Saves a recorded episode as an MP4 video file
)
# Run timestamp (YYYY-MM-DD_HH-MM-SS string) and global seed setter
from models.TinyVLA.test.utils.robot_utils import DATE_TIME, set_seed_everywhere


# ─────────────────────────────────────────────────────────────────────────────

# Configure the root logger to emit INFO-level messages with timestamps to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],  # Output to standard output (console)
)
# Module-level logger; all log_message calls ultimately route through this
logger = logging.getLogger(__name__)


# ── Task suite constants ──────────────────────────────────────────────────────

class TaskSuite(str, Enum):
    """
    Enumeration of supported LIBERO task suite identifiers.

    Inheriting from both `str` and `Enum` allows direct string comparison
    and use as dictionary keys or CLI argument values without extra conversion.

    Members
    -------
    LIBERO_GOAL : "libero_goal"
        The libero_goal suite — tasks where the same objects appear in the same
        scene but with varying goal conditions (used for all ablation tasks here).
    """
    LIBERO_GOAL = "libero_goal"


# Maximum number of simulation steps allowed per episode, keyed by task suite.
# 300 steps for libero_goal matches the LIBERO benchmark's evaluation protocol.
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_GOAL: 300,
}


# ── Ablation task configurations ──────────────────────────────────────────────

def get_ablation_tasks(task_id: int) -> dict:
    """
    Retrieve the ablation configuration for a specific LIBERO task by its 0-based index.

    Each ablation configuration defines a set of "test variants" for one task.
    Every variant replaces the original full natural language instruction with a
    degraded or keyword-only version, encoded in a custom BDDL file. By running
    the same policy with each degraded command and measuring success rates, we can
    determine whether the model relies on specific keywords or on full language understanding.

    Currently supported task IDs:
      - 7 → Task 8: "Turn on the stove"
      - 8 → Task 9: "Put the bowl on the plate"

    Parameters
    ----------
    task_id : int
        0-based index of the task in the libero_goal suite.

    Returns
    -------
    dict
        Ablation configuration with the following structure:
        {
            "task_name": str,          # Human-readable name of the task
            "tests": {
                "<variant_key>": {
                    "bddl_file"       : str,  # Filename of the custom BDDL for this variant
                    "expected_command": str,  # The degraded keyword command used as language input
                },
                ...
            }
        }

    Raises
    ------
    ValueError
        If `task_id` is not a key in ABLATION_CONFIGS, listing all valid IDs.

    Notes
    -----
    The BDDL files define the same scene and success condition as the original task
    but embed the degraded language string in their task description field.
    This ensures the environment and goal are identical across all ablation variants;
    only the language input to the policy changes.

    Example
    -------
    >>> cfg = get_ablation_tasks(7)
    >>> cfg["task_name"]
    'Turn on the stove'
    >>> cfg["tests"]["stove1"]["expected_command"]
    'stove'
    """

    # Registry of all configured ablation tasks.
    # Each entry maps a 0-based task_id to its name and a dict of test variants.
    ABLATION_CONFIGS = {

        # Task 8 (0-indexed as 7): "Turn on the stove"
        # Tests range from a single-word command to a verb-phrase-only command.
        7: {
            "task_name": "Turn on the stove",
            "tests": {
                # Variant 1: only the target object name
                "stove1": {
                    "bddl_file": "turn_on_the_stove_ablation_stove1.bddl",
                    "expected_command": "stove",
                },
                # Variant 2: a distractor object + target object
                "stove2": {
                    "bddl_file": "turn_on_the_stove_ablation_stove2.bddl",
                    "expected_command": "bowl stove",
                },
                # Variant 3: another distractor + target object
                "stove3": {
                    "bddl_file": "turn_on_the_stove_ablation_stove3.bddl",
                    "expected_command": "plate stove",
                },
                # Variant 4: only the action phrase (no object specified)
                "stove4": {
                    "bddl_file": "turn_on_the_stove_ablation_stove4.bddl",
                    "expected_command": "Turn on",
                },
            },
        },

        # Task 9 (0-indexed as 8): "Put the bowl on the plate"
        8: {
            "task_name": "Put the bowl on the plate",
            "tests": {
                "bowl_plate1": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate1.bddl",
                    "expected_command": "bowl",
                },
                "bowl_plate2": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate2.bddl",
                    "expected_command": "plate",
                },
                "bowl_plate3": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate3.bddl",
                    "expected_command": "bowl plate",
                },
                "bowl_plate4": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate4.bddl",
                    "expected_command": "Put plate",
                },
                "bowl_plate5": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate5.bddl",
                    "expected_command": "plate bowl",
                },
            },
        },
    }

    # Raise a descriptive error if the requested task_id has no ablation config
    if task_id not in ABLATION_CONFIGS:
        raise ValueError(
            f"No ablation configuration found for task_id={task_id}. "
            f"Available task IDs: {list(ABLATION_CONFIGS.keys())}"
        )

    return ABLATION_CONFIGS[task_id]


# ── Configuration dataclass ───────────────────────────────────────────────────

@dataclass
class GenerateConfig:
    # fmt: off

    ###########################################################################
    # Ablation-specific parameters
    ###########################################################################
    ablation_task_id: int = 7       # 0-indexed task ID to ablate (default: Task 8)
    ablation_test_key: str = ""     # If non-empty, run only this variant key; empty = run all

    ###########################################################################
    # Model-specific parameters (TinyVLA)
    ###########################################################################
    model_path: str = ""            # Path to the TinyVLA LoRA checkpoint; REQUIRED
    model_base: str = ""            # Path to the base LLaVA-Pythia model; REQUIRED
    model_family: str = "tiny_vla"  # Model family ID for dummy action selection

    ###########################################################################
    # LIBERO environment-specific parameters
    ###########################################################################
    task_suite_name: str = TaskSuite.LIBERO_GOAL  # Only libero_goal is supported here
    num_steps_wait: int = 10        # Physics stabilization steps at episode start
    num_trials_per_task: int = 50   # Number of rollout episodes per ablation variant
    initial_states_path: str = "DEFAULT"  # Path to custom initial states JSON, or "DEFAULT"
    env_img_res: int = 256          # Camera render resolution in pixels

    ###########################################################################
    # Utils
    ###########################################################################
    run_id_note: Optional[str] = None           # Optional human-readable run label
    local_log_dir: str = "/mnt/beegfs/a.cardamone7/outputs/logs"  # Log directory

    use_wandb: bool = False                     # Enable W&B remote tracking
    wandb_entity: str = "your-wandb-entity"     # W&B entity (username or team)
    wandb_project: str = "your-wandb-project"   # W&B project name

    seed: int = 42                  # Global random seed for reproducibility
    run_number: int = 0             # Run version number for output directory naming

    debug: bool = False             # Activate debugpy remote debugger if True
    # fmt: on


# ── Validation ────────────────────────────────────────────────────────────────

def validate_config(cfg: GenerateConfig) -> None:
    """
    Validate the runtime configuration before executing the ablation study.

    Performs the following checks:
      1. `model_path` is non-empty (required for loading the checkpoint).
      2. `model_base` is non-empty (required when using LoRA adapters).
      3. `task_suite_name` is set to "libero_goal" (the only supported suite).
      4. `ablation_task_id` corresponds to a valid entry in ABLATION_CONFIGS.

    Parameters
    ----------
    cfg : GenerateConfig
        The runtime configuration populated from CLI arguments.

    Raises
    ------
    AssertionError
        If any of the first three checks fail.
    ValueError
        If `ablation_task_id` has no ablation configuration (raised inside
        `get_ablation_tasks`).

    Notes
    -----
    This function is intentionally called early in `eval_ablation()` to
    surface configuration errors before any expensive model loading occurs.
    """
    assert cfg.model_path, "model_path must not be empty!"
    assert cfg.model_base, "model_base must not be empty!"
    assert cfg.task_suite_name == TaskSuite.LIBERO_GOAL, (
        "Ablation only works with libero_goal!"
    )
    # This call will raise ValueError if task_id is not registered
    get_ablation_tasks(cfg.ablation_task_id)


# ── Policy class ───────────────────────────────────────────────────────────────

class llava_pythia_act_policy:
    """
    Wrapper class for the TinyVLA LLaVA-Pythia Vision-Language-Action policy.

    This class encapsulates the full inference pipeline for TinyVLA, including:
      - Loading pretrained weights with optional LoRA adapter support
      - Constructing multimodal input batches from dual-camera observations,
        robot proprioceptive state, and natural language task descriptions
      - Formatting prompts using LLaVA-Pythia conversation templates
      - Preprocessing images to the square canvas format expected by the visual encoder

    This is functionally identical to the policy class in `run_libero_eval.py`
    and `run_libero_eval_task_comp.py`, reproduced here for standalone operation.

    Attributes
    ----------
    policy_config : dict
        Stores the configuration dict passed at initialization.
    data_args : any
        Optional supplementary data arguments (currently unused).
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for encoding language prompts into token IDs.
    policy : LlavaPythiaForCausalLM
        The loaded VLA model (placed on GPU).
    image_processor : transformers.CLIPImageProcessor (or equivalent)
        Image normalizer and tensor converter.
    context_len : int
        Maximum token sequence length supported by the model.
    config : LlavaPythiaConfig
        Model configuration containing action_dim, chunk_size, mm_use_im_start_end, etc.
    conv : Conversation
        Conversation template instance for prompt formatting (created per forward pass).
    """

    def __init__(self, policy_config, data_args=None):
        """
        Initialize the LLaVA-Pythia policy wrapper.

        Parameters
        ----------
        policy_config : dict
            Configuration dictionary with required keys:
              - "model_path"  (str) : Path to the LoRA checkpoint (or full model) directory.
              - "model_base"  (str) : Path to the frozen base model (used when LoRA is enabled).
              - "enable_lora" (bool): Whether to load LoRA adapter weights.
              - "conv_mode"   (str) : Key for the conversation template (e.g., "pythia").
              - "action_head" (str) : Action head type: "act" or "droid_diffusion".
        data_args : any, optional
            Supplementary data arguments reserved for future use. Default: None.
        """
        super(llava_pythia_act_policy).__init__()  # Call object.__init__ (no-op)
        self.load_policy(policy_config)             # Load weights, tokenizer, config
        self.data_args = data_args                  # Store for potential downstream use

    def load_policy(self, policy_config):
        """
        Load pretrained LLaVA-Pythia model weights and associated components.

        When LoRA is enabled, the base model provides the frozen backbone and
        the checkpoint directory contains the adapter delta weights. The config
        is loaded from the parent directory of the checkpoint path, since LoRA
        saves only the adapter weights while the base config lives one level up.

        Sets:
          - self.tokenizer       : Tokenizer for language prompt encoding
          - self.policy          : Loaded VLA model (on CUDA)
          - self.image_processor : Image normalizer/resizer
          - self.context_len     : Maximum sequence length
          - self.config          : LlavaPythiaConfig from parent directory

        Parameters
        ----------
        policy_config : dict
            See `__init__` for required keys and descriptions.
        """
        self.policy_config = policy_config

        # When using LoRA: provide the base model path; otherwise load full model directly
        model_base = policy_config["model_base"] if policy_config["enable_lora"] else None

        # Extract a human-readable model name from the last component(s) of the path
        model_name = get_model_name_from_path(policy_config["model_path"])
        model_path = policy_config["model_path"]

        # Load the full model stack: tokenizer, model, image processor, context length
        # The two False flags disable device_map="auto" and load_in_8bit respectively
        self.tokenizer, self.policy, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name, False, False
        )

        # Load model configuration from the parent directory of the checkpoint
        # (LoRA checkpoints store only adapter weights; base config is one level up)
        self.config = LlavaPythiaConfig.from_pretrained(
            "/".join(model_path.split("/")[:-1]), trust_remote_code=True
        )

    def process_batch_to_llava(self, curr_image, robo_state, raw_lang):
        """
        Construct a model-ready input batch from raw observations and a task description.

        Handles the complete multimodal preprocessing pipeline:
          1. Splits the stacked dual-image tensor into agentview and wrist components.
          2. Pads each image to a square canvas (letterboxing/pillarboxing).
          3. Normalizes images using the model's image processor.
          4. Builds the conversation prompt with image tokens and language instruction.
          5. Tokenizes the prompt, replacing "<image>" with IMAGE_TOKEN_INDEX.
          6. Packs all tensors into a model-compatible input dictionary.

        Parameters
        ----------
        curr_image : torch.Tensor
            Image tensor of shape (2, C, H, W) or (1, 2, C, H, W) with stacked
            agentview (front camera) and wrist camera frames.
        robo_state : torch.Tensor
            1D float tensor of shape (state_dim,) representing the normalized
            robot proprioceptive state: [eef_xyz, axis_angle, gripper_qpos].
        raw_lang : str
            Task description string passed to the policy, which may be a full
            sentence (e.g., "Turn on the stove") or a degraded keyword command
            (e.g., "stove") for ablation purposes.

        Returns
        -------
        data_dict : dict
            Input batch for `policy.forward()` with keys:
              - "input_ids"      : (1, seq_len) LongTensor of token IDs on CUDA
              - "attention_mask" : (1, seq_len) BoolTensor (False for pad tokens)
              - "images"         : Preprocessed agentview image tensor on GPU
              - "images_r"       : Preprocessed wrist camera tensor on GPU
              - "states"         : (1, state_dim) float tensor on GPU

        Notes
        -----
        The key design choice here is that `raw_lang` is passed unmodified to the
        model, regardless of whether it is a complete sentence or a degraded keyword
        string. This is intentional for the ablation study — the model receives the
        same visual input but a linguistically reduced command.
        """
        # Fresh conversation template for each forward pass (no state leakage)
        self.conv = conv_templates[self.policy_config["conv_mode"]].copy()

        # Remove extra batch dimension if input has 5 dims: (1, 2, C, H, W) → (2, C, H, W)
        if len(curr_image.shape) == 5:
            curr_image = curr_image.squeeze(0)

        # Split stacked images along the camera dimension:
        #   image   → agentview (front camera), shape (1, C, H, W)
        #   image_r → wrist camera,             shape (1, C, H, W)
        image, image_r = torch.chunk(curr_image, 2, dim=0)

        # --- Agentview image preprocessing ---
        # Pad to square canvas filled with mean color (avoids distortion)
        image = self.expand2square(image, tuple(x for x in self.image_processor.image_mean))
        # Normalize to model's expected pixel distribution (do_rescale=False: already [0,1])
        image_tensor = self.image_processor.preprocess(
            image, return_tensors="pt",
            do_normalize=True, do_rescale=False, do_center_crop=False
        )["pixel_values"]
        image_tensor = image_tensor.to(self.policy.device, dtype=self.policy.dtype)  # Move to GPU

        # --- Wrist camera preprocessing (identical pipeline) ---
        image_r = self.expand2square(image_r, tuple(x for x in self.image_processor.image_mean))
        image_tensor_r = self.image_processor.preprocess(
            image_r, return_tensors="pt",
            do_normalize=True, do_rescale=False, do_center_crop=False
        )["pixel_values"]
        image_tensor_r = image_tensor_r.to(self.policy.device, dtype=self.policy.dtype)

        inp = raw_lang  # The (possibly degraded) language instruction for this ablation variant
        assert image is not None, "image must be provided."  # Defensive check

        # Prepend the image token to the prompt.
        # If mm_use_im_start_end=True, wrap with explicit boundary markers.
        if self.policy.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + inp  # Simple format: <image>\n<instruction>

        # Add user message (human role) to the conversation template
        self.conv.append_message(self.conv.roles[0], inp)
        image = None  # Release reference; data is captured in image_tensor

        # Add empty assistant turn to trigger generation at this position
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()   # Render the full conversation string
        prompt += " <|endoftext|>"        # Append EOS token to close the context

        # Tokenize the prompt: <image> → IMAGE_TOKEN_INDEX placeholder in token IDs
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()            # Add batch dimension; move to CUDA

        # Attention mask: 1 for real tokens, 0 for padding tokens
        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Move robot state to the model's device/dtype and add batch dimension
        states = robo_state.to(self.policy.device, dtype=self.policy.dtype)

        # Pack the full input batch into a dict for the model forward call
        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attn_mask,
            images=image_tensor,
            images_r=image_tensor_r,
            states=states.unsqueeze(0),  # Shape: (1, state_dim)
        )
        return data_dict

    def expand2square(self, pil_imgs, background_color):
        """
        Pad a batch of images to a square canvas by centering along the shorter axis.

        Creates a square background canvas of size max(H, W) × max(H, W) filled with
        a uniform color, and places the original image at the center. This prevents
        aspect-ratio distortion when feeding non-square images to CLIP-style encoders.

        Parameters
        ----------
        pil_imgs : torch.Tensor
            Image batch with shape (B, C, H, W), values in [0, 1] (float32).
        background_color : tuple of float
            Per-channel fill values for the background canvas (typically the
            image processor's `image_mean`, e.g., (0.485, 0.456, 0.406)).

        Returns
        -------
        torch.Tensor
            Square image batch of shape (B, max_dim, max_dim, C) as a torch.Tensor
            on the same device and dtype as the input. Note: channels are in the
            LAST dimension (B, H, W, C) for compatibility with the image processor.

        Notes
        -----
        - For square inputs (H == W): only a dimension permutation is applied.
        - For portrait (H > W): image is centered horizontally (pillarbox padding).
        - For landscape (H < W): image is centered vertically (letterbox padding).
        """
        batch_size, channels, height, width = pil_imgs.shape

        # Square canvas size is the larger of the two spatial dimensions
        max_dim = max(height, width)

        # Initialize canvas with background fill; shape: (B, max_dim, max_dim, C)
        expanded_imgs = np.full(
            (batch_size, max_dim, max_dim, channels), background_color, dtype=np.float32
        )

        if height == width:
            # Already square: permute (B, C, H, W) → (B, H, W, C) for NumPy layout
            expanded_imgs = pil_imgs.permute(0, 2, 3, 1).cpu().numpy()
        elif height > width:
            # Portrait: pad left and right (pillarbox)
            offset = (max_dim - width) // 2  # Number of background pixels on each horizontal side
            expanded_imgs[:, :height, offset: offset + width, :] = (
                pil_imgs.permute(0, 2, 3, 1).cpu().numpy()
            )
        else:
            # Landscape: pad top and bottom (letterbox)
            offset = (max_dim - height) // 2  # Number of background pixels on each vertical side
            expanded_imgs[:, offset: offset + height, :width, :] = (
                pil_imgs.permute(0, 2, 3, 1).cpu().numpy()
            )

        # Convert back to torch.Tensor on the same device and dtype as the input
        expanded_imgs = torch.tensor(expanded_imgs).to(
            dtype=pil_imgs.dtype, device=pil_imgs.device
        )
        return expanded_imgs


# ── Observation helpers ───────────────────────────────────────────────────────

def convert_actions(pred_action):
    """
    Convert a raw 10-DOF policy output from 6D rotation format to a 7-DOF robot command.

    The VLA model predicts actions as:
      pred_action[0:3]  → XYZ end-effector displacement
      pred_action[3:9]  → Rotation in 6D continuous representation (Zhou et al., CVPR 2019)
      pred_action[9]    → Gripper open/close scalar

    This function converts the 6D rotation to Euler angles (XYZ convention), yielding
    the standard 7-DOF robot action: [x, y, z, roll, pitch, yaw, gripper].

    Parameters
    ----------
    pred_action : np.ndarray
        Raw action vector of shape (10,).

    Returns
    -------
    np.ndarray
        Converted 7-DOF action array: [x, y, z, euler_x, euler_y, euler_z, gripper].

    Notes
    -----
    The 6D rotation representation avoids the gimbal lock and discontinuities of
    Euler angles or quaternions in neural network output space.
    See: "On the Continuity of Rotation Representations in Neural Networks",
    Zhou et al., CVPR 2019.
    """
    cur_xyz = pred_action[:3]                           # XYZ translation: shape (3,)
    cur_rot6d = pred_action[3:9]                        # 6D rotation vector: shape (6,)
    cur_gripper = np.expand_dims(pred_action[-1], axis=0)  # Gripper scalar → shape (1,)

    # Convert numpy array to PyTorch tensor and add batch dimension: (1, 6)
    cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)

    # Convert 6D rotation → Euler angles (XYZ convention), then squeeze to (3,)
    cur_euler = TorchUtils.rot_6d_to_euler_angles(
        rot_6d=cur_rot6d, convention="XYZ"
    ).squeeze().numpy()

    # Concatenate all components into the final 7-DOF action vector
    pred_action = np.concatenate((cur_xyz, cur_euler, cur_gripper))
    return pred_action


def get_obs(obs, stats):
    """
    Extract and preprocess visual and proprioceptive observations from the LIBERO environment.

    Processes the raw observation dictionary returned by `env.step()` or `env.reset()`
    into two tensors consumed by the policy:
      1. **images**: A (2, 180, 320, 3) float32 NumPy array of preprocessed camera frames.
      2. **states**: A 1D NumPy array of Z-score normalized proprioceptive state.

    Parameters
    ----------
    obs : dict
        Raw environment observation dictionary. Expected keys:
          - "agentview_image"           : (H, W, 3) uint8 front-view RGB
          - "robot0_eye_in_hand_image"  : (H, W, 3) uint8 wrist-view RGB
          - "robot0_eef_pos"            : (3,) float32 end-effector XYZ position
          - "robot0_eef_quat"           : (4,) float32 end-effector quaternion (w, x, y, z)
          - "robot0_gripper_qpos"       : (2,) float32 gripper joint positions
    stats : dict
        Dataset normalization statistics loaded from `dataset_stats.pkl`:
          - "qpos_mean" : (state_dim,) mean of proprioceptive states in training data
          - "qpos_std"  : (state_dim,) std deviation of proprioceptive states

    Returns
    -------
    images : np.ndarray
        Array of shape (2, 180, 320, 3) with preprocessed camera images.
        Index 0: agentview (front camera). Index 1: wrist camera.
    states : np.ndarray
        Normalized proprioceptive state array of shape (state_dim,) = (8,):
        [eef_x, eef_y, eef_z, axis_x, axis_y, axis_z, gripper_l, gripper_r].

    Notes
    -----
    - `[::-1, ::-1]` flips the image both vertically and horizontally to correct
      MuJoCo's inverted camera output (cameras render upside-down and mirrored
      by default in the robosuite/LIBERO simulation backend).
    - Resizing to (320, 180) matches the resolution expected by the model's image encoder
      before the `expand2square` padding step.
    - The proprioceptive state is z-score normalized: (x - μ) / σ, ensuring that
      the model receives inputs from the same distribution as during training.
    """
    # Build the image batch: flip both axes to correct MuJoCo's inverted camera output,
    # then resize to 320×180 for the model's image encoder
    images = np.array([
        cv2.resize(obs["agentview_image"][::-1, ::-1], (320, 180)),           # Front camera
        cv2.resize(obs["robot0_eye_in_hand_image"][::-1, ::-1], (320, 180)),  # Wrist camera
    ])

    # Construct proprioceptive state by concatenating:
    #   end-effector XYZ position      (3,)
    #   axis-angle orientation         (3,) — converted from quaternion
    #   gripper joint positions        (2,)
    states = np.concatenate((
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),  # Quaternion → axis-angle for compactness
        obs["robot0_gripper_qpos"],
    ))

    # Z-score normalization using training statistics: (x - mean) / std
    states = (states - stats["qpos_mean"]) / stats["qpos_std"]

    return images, states


# ── Logging helpers ───────────────────────────────────────────────────────────

def log_message(message: str, log_file=None):
    """
    Emit a message to both the console logger and an optional text log file.

    Provides a single unified call site for all runtime logging throughout the
    ablation study, ensuring that every informational message is simultaneously
    printed to stdout and persisted to the run's log file.

    Parameters
    ----------
    message : str
        The text string to log.
    log_file : file-like object or None, optional
        An open writable file object. If provided, the message is appended
        to the file followed by a newline, and the file is flushed immediately
        to prevent data loss on unexpected termination. Default: None.

    Returns
    -------
    None
    """
    logger.info(message)        # Emit at INFO level to stdout via the configured logger
    if log_file:
        log_file.write(message + "\n")  # Append message with newline separator
        log_file.flush()                # Force write to disk immediately


def setup_logging(cfg: GenerateConfig, task_name: str):
    """
    Initialize run-specific logging: construct a unique run ID, open a log file,
    and optionally start a Weights & Biases tracking run.

    The run ID encodes the key identifying parameters:
        ABLATION-Task{N}-{task_name}-{model_family}-{DATE_TIME}[--{run_id_note}]

    Parameters
    ----------
    cfg : GenerateConfig
        Runtime configuration (provides model_family, run_id_note, etc.).
    task_name : str
        Human-readable task name (e.g., "Turn on the stove"), used in the run ID.
        Spaces are replaced with underscores and converted to lowercase.

    Returns
    -------
    log_file : file object
        Writable file object for the run's text log. Caller must close it.
    local_log_filepath : str
        Absolute path to the created log file.
    run_id : str
        The unique run identifier string (used for W&B run naming and file naming).

    Side Effects
    ------------
    - Creates `cfg.local_log_dir` (and all parent directories) if it does not exist.
    - Opens a new text file for writing (not appending).
    - Optionally calls `wandb.init()` to start a new W&B run.
    """
    # Sanitize task name for use in the run ID (no spaces, lowercase)
    safe_task_name = task_name.replace(" ", "_").lower()

    # Compose the unique run ID string using task, model, and timestamp components
    run_id = (
        f"ABLATION-Task{cfg.ablation_task_id + 1}-"
        f"{safe_task_name}-{cfg.model_family}-{DATE_TIME}"
    )

    # Append optional user note if provided (for human-readable differentiation)
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Ensure the log directory exists before creating the file
    os.makedirs(cfg.local_log_dir, exist_ok=True)

    # Open the log file for writing; creates a new file at each run
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Optionally initialize a new W&B run for remote tracking
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


# ── Initial-states loading ────────────────────────────────────────────────────

def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """
    Load the initial environment states for a given task from the LIBERO benchmark
    or from a user-specified JSON file of custom initial states.

    Initial states define the exact starting configuration of the simulation:
    object positions, robot joint angles, and gripper state. Using fixed initial
    states ensures reproducibility across ablation variants — each variant sees
    the exact same scene at the start of every episode.

    Parameters
    ----------
    cfg : GenerateConfig
        Runtime configuration. Key field: `initial_states_path`.
    task_suite : libero.libero.benchmark.TaskSuite
        Instantiated LIBERO task suite object providing access to task init states.
    task_id : int
        0-based index of the task within the suite.
    log_file : file object or None, optional
        Open log file for emitting status messages.

    Returns
    -------
    initial_states : list or array-like
        Default initial states from the LIBERO benchmark for `task_id`.
        Indexed by episode index (0 to num_trials - 1).
    all_initial_states : dict or None
        If `cfg.initial_states_path != "DEFAULT"`, a nested dict loaded from
        the JSON file, structured as:
            {task_name_str: {demo_0: {success: bool, initial_state: list}, ...}}
        Returns None when using default states.

    Notes
    -----
    - The default LIBERO initial states are typically 50 pre-sampled configurations
      stored in `.pruned_init` files.
    - Custom initial states from JSON are used when evaluating on a specific subset
      of episodes (e.g., only episodes where the expert demo succeeded), enabling
      more controlled ablation comparisons.
    """
    # Always load the default LIBERO initial states for this task
    initial_states = task_suite.get_task_init_states(task_id)

    if cfg.initial_states_path != "DEFAULT":
        # Load custom initial states from a user-provided JSON file
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        # Use benchmark default initial states
        log_message("Using default initial states", log_file)
        return initial_states, None


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    policy,
    policy_config,
    initial_state=None,
    log_file=None,
):
    """
    Execute a single evaluation rollout episode in the LIBERO simulation environment.

    This function runs the complete policy-environment interaction loop:
      1. Reset the environment (and optionally restore a specific initial state).
      2. Stabilize the simulation physics with no-op actions.
      3. At each timestep: observe → preprocess → query policy → execute action.
      4. Return whether the task was completed and the full trajectory record.

    A key feature is **temporal aggregation**: the policy predicts a chunk of
    `num_queries` future actions per query. These overlapping predictions are
    accumulated in a rolling buffer (`all_time_actions`) and combined at each
    step via exponentially-decaying weights, reducing action jitter.

    Special behavior for basket scenes: if the observation contains a "basket_1_pos"
    key (a known TinyVLA-specific environment artifact), the basket is repositioned
    to a fixed canonical location to avoid unstable initial configurations.

    Parameters
    ----------
    cfg : GenerateConfig
        Evaluation configuration (num_steps_wait, model_family, task_suite_name).
    env : OffScreenRenderEnv
        Active LIBERO simulation environment (already instantiated).
    task_description : str
        Language instruction passed to the policy. This may be a full sentence or
        a degraded keyword command for the ablation study.
    policy : llava_pythia_act_policy
        Loaded TinyVLA policy instance.
    policy_config : dict
        Policy configuration (used to select action head and post-processing).
    initial_state : object or None, optional
        If provided, used to restore a specific scene configuration via
        `env.set_init_state()`. If None, `env.reset()` uses default placement.
    log_file : file object or None, optional
        Open log file for emitting episode-level error messages.

    Returns
    -------
    success : bool
        True if `done=True` was returned by the environment before `max_timesteps`,
        indicating the success criterion was met.
    replay_traj : dict
        Recorded episode data:
          - "images"        : list of (256, 256, 3) uint8 NumPy arrays (agentview frames)
          - "task_command"  : str, the language instruction used in this episode
          - "states"        : list of normalized proprioceptive state arrays
          - "actions"       : list of (7,) float32 NumPy arrays (executed robot actions)

    Notes
    -----
    Temporal Aggregation (from ACT — Action Chunking with Transformers):
    -------------------------------------------------------------------
    Each policy query returns a chunk of `num_queries` future actions.
    `all_time_actions[t, t:t+num_queries]` stores the chunk predicted at step t.
    At execution step t, all chunks that include step t are gathered, zero-initialized
    slots are filtered out, and the remaining predictions are combined as:

        a_t = Σ exp(-k · i) · a_i  /  Σ exp(-k · i)

    where i is the prediction's age (0 = most recent) and k=0.01 gives mild
    recency bias. This weighted ensemble reduces action inconsistencies.

    Model Warm-up:
    --------------
    At t=0, ten dummy forward passes are executed and discarded. This forces
    CUDA kernel compilation and GPU memory allocation to occur before timing-sensitive
    real inference begins.
    """
    env.reset()                           # Reset environment to clear prior episode state
    to_tensor = transforms.ToTensor()     # Converts HWC uint8 numpy → CHW float32 tensor in [0,1]

    # Set the initial state if specified (for reproducible evaluation across ablation variants)
    if initial_state is not None:
        obs = env.set_init_state(initial_state)   # Restore exact object/robot configuration
    else:
        obs = env.reset()
        # TinyVLA-specific workaround: if a basket is present in the scene,
        # reposition it to a known stable location to avoid physics instability
        if "basket_1_pos" in obs.keys():
            basket_pos = [0.005, 0.261, 0.035]          # Canonical basket XYZ position
            basket_quat = [0.000, 0.000, 0.000, 1.000]  # Identity quaternion (no rotation)
            # Directly write the basket pose into the MuJoCo simulation data
            env.sim.data.set_joint_qpos(
                env.env.objects_dict["basket_1"].joints[0],
                np.concatenate((basket_pos, basket_quat)),
            )
            # Run stabilization steps after repositioning the basket
            t = 0
            while t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1

    # ── Policy configuration ──────────────────────────────────────────────
    if policy_config["action_head"] == "act":
        rand_crop_resize = False   # ACT does not use random crop at test time
        temporal_agg = True        # ACT uses temporal aggregation
    else:
        rand_crop_resize = True    # Diffusion head applies random crop for consistency with training
        temporal_agg = True

    action_dim = policy.config.action_dim   # Raw action output dimensionality (e.g., 10)
    policy.policy.eval()                    # Evaluation mode: disable dropout, batch norm tracking

    # Load dataset normalization statistics from the pickle file next to the checkpoint
    stats_path = os.path.join(
        "/".join(policy_config["model_path"].split("/")[:-1]), "dataset_stats.pkl"
    )
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)  # Contains: action_mean, action_std, action_min, action_max, qpos_mean, qpos_std

    # Define post-processing function to un-normalize model outputs:
    #   ACT head: z-score normalized → reverse with a * std + mean
    #   Diffusion heads: normalized to [-1, 1] → reverse with linear map to [min, max]
    if policy_config["action_head"] == "act":
        post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
    elif policy_config["action_head"] in ("transformer_diffusion", "droid_diffusion"):
        post_process = (
            lambda a: ((a + 1) / 2) * (stats["action_max"] - stats["action_min"])
            + stats["action_min"]
        )
    else:
        raise NotImplementedError(f"Unknown action_head: {policy_config['action_head']}")

    # Determine query frequency (how often to call the policy):
    #   With temporal aggregation: query at every step (frequency=1)
    #   Without: query every chunk_size/2 steps
    query_frequency = policy.config.chunk_size / 2
    if temporal_agg:
        query_frequency = 1
        num_queries = policy.config.chunk_size  # Number of future steps per prediction chunk

    # Maximum episode length from the task suite constants
    max_timesteps = int(TASK_MAX_STEPS[cfg.task_suite_name])

    # Pre-allocate temporal aggregation buffer:
    # Shape: (max_timesteps, max_timesteps + num_queries, action_dim)
    # all_time_actions[t, s] = prediction for step s made at query time t
    if temporal_agg:
        all_time_actions = torch.zeros(
            [max_timesteps, max_timesteps + num_queries, action_dim],
            dtype=torch.float32,
        ).cuda()

    # ── Episode loop initialization ───────────────────────────────────────
    t = 0                       # Simulation timestep counter
    replay_traj = dict()        # Trajectory record for video and .npy export
    image_list = []             # Agentview frames for video recording
    robot_state_list = []       # Proprioceptive state history
    target_action_list = []     # Executed action history
    success = False             # Episode outcome (True if task solved)

    with torch.inference_mode():  # Disable autograd for all operations inside
        try:
            # --- Physics Stabilization ---
            # Execute no-op actions to allow the simulation to settle from reset
            while t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
            t = 0  # Reset step counter after stabilization phase

            # --- Main Control Loop ---
            while t < max_timesteps:
                # Extract preprocessed images and normalized robot state
                traj_rgb_np, robot_state = get_obs(obs=obs, stats=stats)

                # Record agentview frame resized to 256×256 for video consistency
                image_list.append(cv2.resize(traj_rgb_np[0], (256, 256)))
                robot_state_list.append(robot_state)

                # Convert robot state to a CUDA float tensor for policy input
                robot_state = torch.from_numpy(robot_state).float().cuda()

                # --- Image tensor preparation (at policy query frequency) ---
                if t % query_frequency == 0:
                    curr_image = []
                    for img in traj_rgb_np:
                        # Convert HWC numpy → CHW float32 tensor [0,1] on CUDA
                        curr_image.append(to_tensor(img).float().cuda())
                    curr_image = torch.stack(curr_image, dim=0)  # Shape: (2, C, H, W)

                    if rand_crop_resize:
                        # Center crop to 95% of original dimensions, then resize back.
                        # This mimics the random crop augmentation used during training.
                        original_size = curr_image.shape[-2:]  # (H, W)
                        ratio = 0.95                           # Retain 95% of spatial extent
                        curr_image = curr_image[
                            :, :,
                            int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                            int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2),
                        ]
                        curr_image = curr_image.squeeze(0)               # Remove batch dim
                        resize_transform = transforms.Resize(original_size, antialias=True)
                        curr_image = resize_transform(curr_image)        # Resize back to original size
                        curr_image = curr_image.unsqueeze(0)             # Restore batch dimension

                # --- Network Warm-up (first step only) ---
                # Execute 10 dummy forward passes at t=0 to initialize CUDA kernels
                # and stabilize GPU memory. Outputs are discarded.
                if t == 0:
                    for _ in range(10):
                        batch = policy.process_batch_to_llava(curr_image, robot_state, task_description)
                        policy.policy(**batch, eval=True)

                # --- Policy Inference ---
                if policy_config["action_head"] in ("act", "droid_diffusion"):
                    if t % query_frequency == 0:
                        # Build the multimodal input batch and query the policy
                        batch = policy.process_batch_to_llava(curr_image, robot_state, task_description)
                        # all_actions: shape (1, chunk_size, action_dim)
                        all_actions = policy.policy(**batch, eval=True)

                    if temporal_agg:
                        # Store predictions for future steps t through t+num_queries-1
                        all_time_actions[[t], t: t + num_queries] = all_actions

                        # Collect all predictions covering the current step t
                        actions_for_curr_step = all_time_actions[:, t]  # (max_timesteps, action_dim)

                        # Exclude never-written (all-zero) buffer slots
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]

                        # Compute exponential recency weights (older = lower weight)
                        k = 0.01  # Decay rate: k=0.01 gives mild preference for recent predictions
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()   # Normalize to sum=1
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)

                        # Weighted sum across all available predictions → (1, action_dim)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        # No temporal aggregation: use the t-th action in the current chunk
                        raw_action = all_actions[:, t % query_frequency]
                else:
                    raise NotImplementedError  # Only ACT and droid_diffusion heads supported

                # --- Action Post-processing ---
                raw_action = raw_action.squeeze(0).cpu().numpy()  # Remove batch dim; move to CPU
                action = post_process(raw_action)                  # Un-normalize to original scale
                action = convert_actions(action)                   # Convert 10-DOF → 7-DOF command

                # --- Environment Step ---
                obs, reward, done, info = env.step(action.tolist())  # Execute action in simulation
                target_action_list.append(action)                    # Record executed action

                # If the environment signals success, terminate the episode immediately
                if done:
                    success = True
                    break
                t += 1  # Increment timestep counter

        except Exception as e:
            # Catch any exception (MuJoCo instability, CUDA OOM, etc.) and mark as failure
            log_message(f"Episode error: {e}", log_file)
            success = False

    # Pack trajectory record for video and .npy export
    replay_traj["images"] = image_list
    replay_traj["task_command"] = task_description
    replay_traj["states"] = robot_state_list
    replay_traj["actions"] = target_action_list

    return success, replay_traj


# ── Single ablation variant runner ───────────────────────────────────────────

def run_ablation_task(
    cfg: GenerateConfig,
    task_key: str,
    task_info: dict,
    task_suite,
    task,
    policy,
    policy_config,
    log_file=None,
):
    """
    Execute the full multi-trial evaluation for a single ablation language variant.

    For each of the `cfg.num_trials_per_task` episodes:
      1. Selects the appropriate initial state (default LIBERO or custom JSON).
      2. Optionally skips episodes where the expert demonstration failed
         (only when using custom initial states with success metadata).
      3. Runs the policy with the degraded ablation command.
      4. Saves a rollout video via `save_rollout_video`.
      5. Logs per-episode and cumulative statistics.

    Parameters
    ----------
    cfg : GenerateConfig
        Runtime configuration (num_trials_per_task, initial_states_path, etc.).
    task_key : str
        Short identifier for this ablation variant (e.g., "stove1", "bowl_plate3").
    task_info : dict
        Variant configuration dict from ABLATION_CONFIGS["tests"][task_key]:
          - "bddl_file"       : str — custom BDDL filename for this variant
          - "expected_command": str — the degraded keyword command used as language input
    task_suite : libero.libero.benchmark.TaskSuite
        Instantiated LIBERO benchmark task suite (provides task and init states).
    task : libero Task
        The LIBERO Task object for `cfg.ablation_task_id` (provides `task.language`).
    policy : llava_pythia_act_policy
        Loaded TinyVLA policy instance.
    policy_config : dict
        Policy configuration dictionary.
    log_file : file object or None, optional
        Open text log file.

    Returns
    -------
    task_success_rate : float
        Success rate for this variant: task_successes / task_episodes ∈ [0.0, 1.0].
    task_episodes : int
        Total number of episodes completed for this variant.
    task_successes : int
        Number of successful episodes.

    Notes
    -----
    - The `get_libero_env` function is called with `ablation_bddl_file` to load
      the custom BDDL that encodes the degraded language description. The returned
      `task_description` is the ablated command, while `original_description` is
      the full original sentence (used for logging and initial state key lookup).
    - When using custom initial states from JSON, episodes are identified by
      `"demo_{episode_idx}"` and are skipped if `success=False` in the metadata.
      This ensures the ablation is evaluated only on configurations where the full
      task is solvable.
    """
    task_id = cfg.ablation_task_id

    # Log variant header for visibility in the log file
    log_message("=" * 80, log_file)
    log_message(f"ABLATION TASK: {task_key.upper()}", log_file)
    log_message(f"BDDL File: {task_info['bddl_file']}", log_file)
    log_message("=" * 80, log_file)

    # Load initial states for this task
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Create the LIBERO environment using the ablation-specific BDDL file.
    # `get_libero_env` returns:
    #   env                  : OffScreenRenderEnv with the ablation BDDL loaded
    #   task_description     : degraded command parsed from the ablation BDDL
    #   original_description : original full task command (from the standard task)
    env, task_description, original_description = get_libero_env(
        task,
        cfg.model_family,
        ablation_bddl_file=task_info["bddl_file"],  # Custom BDDL encoding the degraded command
        resolution=cfg.env_img_res,
    )

    ablation_command = task_description  # The degraded keyword string used as policy input

    log_message(f"Original Task {task_id + 1} Command: {original_description}", log_file)
    log_message(f"Ablation Command (from BDDL): '{ablation_command}'", log_file)

    task_episodes, task_successes = 0, 0  # Variant-local counters

    # --- Trial Loop ---
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"Ablation {task_key}"):

        # --- Initial State Selection ---
        if cfg.initial_states_path == "DEFAULT":
            # Use the pre-packaged LIBERO benchmark initial state at this index
            initial_state = initial_states[episode_idx]
        else:
            # Use custom initial states from JSON, keyed by task description
            initial_states_task_key = original_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"  # Key format in the custom JSON

            # Skip episodes where the expert demonstration failed (per metadata)
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(
                    f"Skipping episode {episode_idx} (failed expert demo)", log_file
                )
                continue  # Do not count this episode in the totals

            # Load the initial state as a NumPy array from the JSON list
            initial_state = np.array(
                all_initial_states[initial_states_task_key][episode_key]["initial_state"]
            )

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # --- Run Episode with Degraded Ablation Command ---
        success, replay_traj = run_episode(
            cfg,
            env,
            ablation_command,  # The degraded keyword command fed to the policy
            policy,
            policy_config,
            initial_state,
            log_file,
        )

        task_episodes += 1          # Increment total episode count for this variant
        if success:
            task_successes += 1     # Increment success count

        # --- Save Rollout Video ---
        # `save_rollout_video` writes an MP4 file to the configured output directory.
        # The `change_command=True` flag signals that an ablated (non-standard) command was used.
        save_rollout_video(
            replay_traj,
            task_episodes,
            success=success,
            task_description=(
                f"ablation_task{task_id + 1}_{task_key}_"
                f"{ablation_command.replace(' ', '_')}"
            ),
            log_file=log_file,
            dataset_name=cfg.task_suite_name,
            run=cfg.run_number,
            change_command=True,        # Flag: language was modified from standard
            command_level="ablation",   # Sub-directory label for output organization
        )

        # Log per-episode statistics
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {task_episodes}", log_file)
        log_message(
            f"# successes: {task_successes} ({task_successes / task_episodes * 100:.1f}%)",
            log_file,
        )

    # Compute variant-level success rate (guard against division by zero)
    task_success_rate = (
        float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
    )
    log_message(f"Current task success rate: {task_success_rate:.4f}", log_file)

    # Optionally log this variant's success rate to W&B
    if cfg.use_wandb:
        wandb.log({
            f"success_rate/ablation_{task_key}": task_success_rate,
            f"num_episodes/ablation_{task_key}": task_episodes,
        })

    return task_success_rate, task_episodes, task_successes


# ── Summary printer ───────────────────────────────────────────────────────────

def print_ablation_results(results: dict, task_name: str, ablation_tests: dict):
    """
    Print a formatted summary table of ablation study results to stdout.

    The table displays per-variant performance alongside the degraded command
    used, enabling quick visual comparison of how different keyword subsets
    affect task success rates.

    Table format:
        Test (20 chars) | Command (20 chars) | Success Rate | Episodes
        ...
        AVERAGE         |                    | avg_rate     | total/total

    Parameters
    ----------
    results : dict
        Maps variant key (str) → dict with:
          - "success_rate" : float ∈ [0, 1]
          - "episodes"     : int (total episodes run)
          - "successes"    : int (number of successful episodes)
    task_name : str
        Human-readable name of the ablated task (printed in the title).
    ablation_tests : dict
        The original ablation tests configuration dict (used to retrieve
        the `expected_command` for each variant key).

    Returns
    -------
    None

    Notes
    -----
    - The average success rate is computed as the arithmetic mean across all
      variants (not weighted by episode count).
    - Total successes and episodes are the true sums, not derived from the rate.
    """
    print("\n" + "=" * 100)
    print(f"ABLATION STUDY RESULTS - Task: {task_name}")
    print("=" * 100)
    # Column header row with fixed-width formatting
    print(f"{'Test':<20} | {'Command':<20} | {'Success Rate':>12} | {'Episodes':>15}")
    print("-" * 100)

    # Print one row per ablation variant
    for task_key, result in results.items():
        cmd = ablation_tests[task_key]["expected_command"]  # Degraded command for this variant
        sr = result["success_rate"]       # Fractional success rate
        succ = result["successes"]        # Absolute success count
        total = result["episodes"]        # Total episode count
        # Format: variant key | command | XX.X% | N/M
        print(f"{task_key:<20} | {cmd:<20} | {sr:>11.1%} | {succ:>6}/{total:<7}")

    print("=" * 100)

    # Compute aggregate statistics across all variants
    avg_sr = sum(r["success_rate"] for r in results.values()) / len(results)  # Arithmetic mean
    total_succ = sum(r["successes"] for r in results.values())
    total_eps = sum(r["episodes"] for r in results.values())
    print(f"{'AVERAGE':<20} | {'':<20} | {avg_sr:>11.1%} | {total_succ:>6}/{total_eps:<7}")
    print("=" * 100)


# ── Main entry point ──────────────────────────────────────────────────────────

@draccus.wrap()
def eval_ablation(cfg: GenerateConfig) -> float:
    """
    Main entry point for the TinyVLA ablation study on LIBERO tasks.

    Orchestrates the complete ablation study pipeline:
      1. Validates configuration and seeds RNG for reproducibility.
      2. Loads the ablation task configuration for the target task.
      3. Instantiates the TinyVLA LLaVA-Pythia policy.
      4. Loads the LIBERO task suite and task object.
      5. Sets up logging (text file + optional W&B).
      6. Runs evaluation for each ablation variant (or a single one if specified).
      7. Prints and logs a formatted results table.
      8. Returns the average success rate across all variants.

    The `@draccus.wrap()` decorator automatically parses CLI arguments into the
    `GenerateConfig` dataclass, enabling configuration via command-line flags
    (e.g., `--model_path`, `--ablation_task_id`, `--num_trials_per_task`).

    Parameters
    ----------
    cfg : GenerateConfig
        Runtime configuration populated by `draccus` from command-line arguments.

    Returns
    -------
    float
        Average success rate across all ablation variants evaluated, in [0.0, 1.0].

    Side Effects
    ------------
    - Creates log directory and writes a text log file.
    - Saves MP4 rollout videos to the configured output directory.
    - Optionally logs metrics to Weights & Biases and uploads the log file.
    - Starts a debugpy server if `cfg.debug=True`.

    Notes
    -----
    - The `ablation_test_key` filter allows running a single variant in isolation,
      which is useful for parallelizing the study across multiple jobs or nodes.
    - W&B metrics include per-variant success rates and the overall average.
    """
    # --- Optional: Remote Debugger Attachment ---
    if cfg.debug:
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))       # Start debugpy server on all interfaces, port 5678
        print("Waiting for debugger attach")
        debugpy.wait_for_client()                # Block until a debugger connects

    # Validate all configuration fields before expensive model loading
    validate_config(cfg)

    # Set global random seeds for Python, NumPy, and PyTorch for reproducibility
    set_seed_everywhere(cfg.seed)

    # --- Load Ablation Configuration ---
    ablation_config = get_ablation_tasks(cfg.ablation_task_id)
    task_name = ablation_config["task_name"]          # e.g., "Turn on the stove"
    ablation_tests = ablation_config["tests"]         # Dict of variant key → {bddl_file, expected_command}

    # --- Instantiate TinyVLA Policy ---
    action_head = "droid_diffusion"   # Use diffusion-based action head for this evaluation
    policy_config = {
        "model_path": cfg.model_path,
        "model_base": cfg.model_base,
        "enable_lora": True,          # Load LoRA adapter weights from model_path
        "conv_mode": "pythia",
        "action_head": action_head,
    }
    policy = llava_pythia_act_policy(policy_config)

    # ── Init LIBERO benchmark ─────────────────────────────────────────────
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.ablation_task_id)

    # ── Logging ───────────────────────────────────────────────────────────
    log_file, local_log_filepath, run_id = setup_logging(cfg, task_name)

    log_message("=" * 80, log_file)
    log_message(
        f"ABLATION STUDY: Task {cfg.ablation_task_id + 1} Keyword Shortcut Analysis",
        log_file,
    )
    log_message("=" * 80, log_file)
    log_message(f"Task Name:       {task_name}", log_file)
    log_message(f"Base Command:    {task.language}", log_file)
    log_message(f"Model:           {cfg.model_path}", log_file)
    log_message(f"Seed:            {cfg.seed}", log_file)
    log_message(f"Trials/ablation: {cfg.num_trials_per_task}", log_file)
    log_message(f"Ablation tests:  {list(ablation_tests.keys())}", log_file)
    log_message("=" * 80, log_file)

    # ── Filter to a single variant if ablation_test_key is provided ────────
    if cfg.ablation_test_key:
        if cfg.ablation_test_key not in ablation_tests:
            raise ValueError(
                f"Unknown ablation_test_key '{cfg.ablation_test_key}'. "
                f"Available keys for task {cfg.ablation_task_id}: {list(ablation_tests.keys())}"
            )
        ablation_tests = {cfg.ablation_test_key: ablation_tests[cfg.ablation_test_key]}
        log_message(f"Running single variant: {cfg.ablation_test_key}", log_file)
    else:
        log_message(f"Running all variants: {list(ablation_tests.keys())}", log_file)

    # ── Run all ablation variants ─────────────────────────────────────────
    results = {}
    for task_key, task_info in ablation_tests.items():
        sr, episodes, successes = run_ablation_task(
            cfg,
            task_key,
            task_info,
            task_suite,
            task,
            policy,
            policy_config,
            log_file,
        )
        results[task_key] = {
            "success_rate": sr,
            "episodes": episodes,
            "successes": successes,
        }

    # ── Summary ───────────────────────────────────────────────────────────
    print_ablation_results(results, task_name, ablation_tests)

    log_message("\n" + "=" * 80, log_file)
    log_message("FINAL RESULTS", log_file)
    log_message("=" * 80, log_file)
    for task_key, result in results.items():
        log_message(
            f"  {task_key}: {result['success_rate']:.1%} "
            f"({result['successes']}/{result['episodes']})",
            log_file,
        )
    avg_sr = sum(r["success_rate"] for r in results.values()) / len(results)
    log_message(f"\nAVERAGE: {avg_sr:.1%}", log_file)
    log_message("=" * 80, log_file)

    if cfg.use_wandb:
        wandb.log({"success_rate/ablation_average": avg_sr})
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    return avg_sr


if __name__ == "__main__":
    eval_ablation()
