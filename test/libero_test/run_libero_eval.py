"""
eval_libero.py
==============
Evaluation script for TinyVLA (Tiny Vision-Language-Action) models on LIBERO robotic
manipulation benchmark tasks.

This module orchestrates end-to-end evaluation of a LLaVA-Pythia-based action policy
across multiple LIBERO task suites. It supports:
    - Single or multi-task evaluation across configurable task ranges.
    - Temporal action aggregation (ACT-style exponential weighting).
    - Command-level variation (L1/L2/L3 linguistic paraphrases of task goals).
    - Optional Weights & Biases (wandb) experiment tracking.
    - Rollout video saving and structured JSON summary output.

Dependencies:
    - llava_pythia: Custom LLaVA model built on top of EleutherAI's Pythia LLM.
    - libero: LIBERO simulation benchmark (MuJoCo-based environments).
    - torch / torchvision: Deep learning framework.
    - draccus: Dataclass-based CLI argument parsing.
    - einops: Tensor rearrangement utilities.
    - wandb: Experiment tracking (optional).

Typical usage:
    python eval_libero.py --model_path /path/to/checkpoint --task_suite_name libero_goal

Author: A. Cardamone
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard library imports
# ──────────────────────────────────────────────────────────────────────────────
import sys          # System-specific parameters and functions (e.g., sys.exit)
import os           # OS-level operations: file paths, environment variables
import logging      # Structured logging to stdout and log files
import glob         # Unix-style pathname pattern expansion (used elsewhere)

# ──────────────────────────────────────────────────────────────────────────────
# Environment variable configuration
# Must be set BEFORE importing HuggingFace tokenizers or CUDA libraries,
# because they are read at import time.
# ──────────────────────────────────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # Prevents HuggingFace tokenizer
                                                  # deadlocks in multi-threaded contexts
os.environ['DEVICE'] = "cuda"                     # Target device for inference
os.environ["WANDB_DISABLED"] = "true"             # Globally disable wandb auto-logging;
                                                  # manual wandb.init() calls still work

# ──────────────────────────────────────────────────────────────────────────────
# LLaVA-Pythia model imports
# ──────────────────────────────────────────────────────────────────────────────
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig
# LlavaPythiaConfig: HuggingFace PretrainedConfig subclass that stores all
# model hyperparameters (vision tower, action head, chunk size, action_dim, etc.)

from llava_pythia.conversation import conv_templates, SeparatorStyle
# conv_templates: dict mapping conversation-mode names (e.g., "pythia") to
#   Conversation objects that manage role-based chat formatting.
# SeparatorStyle: Enum controlling how human/assistant turns are delimited.

from llava_pythia.model.builder import load_pretrained_model
# load_pretrained_model: Factory that loads tokenizer, vision tower, LM backbone,
#   and optional LoRA adapters from a checkpoint directory.

from llava_pythia.mm_utils import (
    tokenizer_image_token,          # Tokenizes text with IMAGE_TOKEN_INDEX placeholders
    get_model_name_from_path,       # Extracts model name from checkpoint path string
    KeywordsStoppingCriteria,       # StoppingCriteria subclass for keyword-based EOS
)

from llava_pythia.constants import (
    IMAGE_TOKEN_INDEX,      # Integer sentinel token ID representing an image in the sequence
    DEFAULT_IMAGE_TOKEN,    # String placeholder "<image>" inserted into prompts
    DEFAULT_IM_START_TOKEN, # "<im_start>" boundary token (used when mm_use_im_start_end=True)
    DEFAULT_IM_END_TOKEN,   # "<im_end>"   boundary token (used when mm_use_im_start_end=True)
)

from llava_pythia.model import *   # Registers all custom model classes in HuggingFace's
                                   # AutoModel registry (side-effect import)

# ──────────────────────────────────────────────────────────────────────────────
# Deep learning / numerical imports
# ──────────────────────────────────────────────────────────────────────────────
import torch                         # Core PyTorch tensor operations and autograd
from torchvision import transforms   # Image preprocessing pipelines (ToTensor, Resize)
import cv2                           # OpenCV: image I/O and resizing (BGR by default)
from copy import deepcopy            # Deep object copying (used for conversation templates)
from itertools import repeat         # Creates infinite iterators (utility, not used directly here)
import numpy as np                   # Numerical array operations
import time                          # Wall-clock timing utilities
from einops import rearrange         # Concise tensor shape manipulation (e.g., "b c h w -> b h w c")

# ──────────────────────────────────────────────────────────────────────────────
# Project-local utilities
# ──────────────────────────────────────────────────────────────────────────────
import models.TinyVLA.test.utils.torch_utils as TorchUtils
# TorchUtils: Custom rotation conversion utilities.
# Used for rot_6d_to_euler_angles (6D → XYZ Euler angles).

import matplotlib.pyplot as plt      # Plotting (available for debugging/visualization)
import argparse                      # CLI argument parsing (superseded by draccus here)

from models.TinyVLA.test.utils.robot_utils import set_seed_everywhere
# set_seed_everywhere: Sets random seeds in Python, NumPy, and PyTorch (including CUDA)
#   for fully deterministic evaluation runs.

# ──────────────────────────────────────────────────────────────────────────────
# Python standard library: typing and dataclasses
# ──────────────────────────────────────────────────────────────────────────────
from enum import Enum                # Base class for symbolic enumerations
from dataclasses import dataclass    # Decorator that auto-generates __init__, __repr__, etc.
import tqdm                          # Progress bars for loops
from typing import Optional, Union   # Type hints for optional and union types

# ──────────────────────────────────────────────────────────────────────────────
# External framework imports
# ──────────────────────────────────────────────────────────────────────────────
import draccus          # Dataclass-based argument parser (reads CLI args into dataclasses)
import transformers     # HuggingFace Transformers (tokenizers, model loading, etc.)

# ──────────────────────────────────────────────────────────────────────────────
# LIBERO benchmark imports
# ──────────────────────────────────────────────────────────────────────────────
from libero.libero import benchmark
# benchmark: Module exposing get_benchmark_dict(), which returns a dict mapping
#   task-suite names to lazy-constructable TaskSuite classes.

from models.TinyVLA.test.utils.libero_utils import (
    get_libero_dummy_action,    # Returns a zero/no-op action for the given model family
    get_libero_env,             # Constructs a LIBERO MuJoCo gym environment for a task
    get_libero_image,           # Extracts the agent-view RGB image from an observation dict
    get_libero_wrist_image,     # Extracts the wrist-camera RGB image from an observation dict
    quat2axisangle,             # Converts quaternion [w,x,y,z] to axis-angle representation
    save_rollout_video,         # Encodes and saves an episode's image list as an MP4 video
)

from PIL import Image                            # Pillow: image manipulation (used by image_processor)
from models.TinyVLA.test.utils.robot_utils import DATE_TIME
# DATE_TIME: A pre-formatted timestamp string (e.g., "20240115-143022") used for
#   unique run identifiers and log file names.

import wandb            # Weights & Biases experiment tracking client
import json             # JSON serialization/deserialization for config and result files
from collections import deque   # Double-ended queue (available for sliding-window buffers)
import pickle           # Binary serialization for loading dataset statistics (pkl files)

# ──────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,                          # Show INFO and above (DEBUG is suppressed)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Timestamp + level + message
    handlers=[logging.StreamHandler()],          # Write log records to sys.stderr
)
logger = logging.getLogger(__name__)
# Module-level logger. Using __name__ namespaces log records (e.g., "eval_libero")
# so they can be filtered independently in multi-module setups.


# ──────────────────────────────────────────────────────────────────────────────
# Task Suite Enumeration
# ──────────────────────────────────────────────────────────────────────────────
class TaskSuite(str, Enum):
    """
    Enumeration of supported LIBERO task suites.

    Inheriting from both `str` and `Enum` allows values to be used directly as
    strings (e.g., in dict keys, CLI args) without calling `.value`.

    Members:
        LIBERO_SPATIAL: 10 tasks testing spatial reasoning (object placement).
        LIBERO_OBJECT:  10 tasks testing object manipulation diversity.
        LIBERO_GOAL:    10 tasks with longer-horizon goal-conditioned behaviour.
        LIBERO_10:      10 complex tasks with up to 520 timesteps.
        LIBERO_90:      90 diverse tasks for large-scale training/evaluation.
    """
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT  = "libero_object"
    LIBERO_GOAL    = "libero_goal"
    LIBERO_10      = "libero_10"
    LIBERO_90      = "libero_90"


# ──────────────────────────────────────────────────────────────────────────────
# Maximum episode length per task suite
# ──────────────────────────────────────────────────────────────────────────────
TASK_MAX_STEPS = {
    # Dict[TaskSuite, int] → maximum number of environment steps allowed per episode.
    # These values were determined empirically to balance evaluation thoroughness
    # with simulation runtime.
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT:  280,
    TaskSuite.LIBERO_GOAL:    300,
    TaskSuite.LIBERO_10:      520,
    TaskSuite.LIBERO_90:      400,
}


# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def log_message(message: str, log_file=None):
    """
    Log a message to both the console (via the module logger) and optionally
    to an open text file.

    This dual-sink logging ensures that both live monitoring (terminal) and
    post-hoc analysis (log file) capture the same information without duplicating
    logging configuration.

    Args:
        message (str): The text to log. Newlines are preserved as-is.
        log_file (file-like object, optional): An open writable file handle.
            If provided, the message is appended followed by a newline and the
            buffer is flushed immediately (important for crash-safe logging).
            Defaults to None (console only).

    Returns:
        None

    Example:
        >>> log_message("Episode 1 started", log_file=open("run.txt", "w"))
    """
    logger.info(message)       # Write to stderr via the configured logging handler
    if log_file:
        log_file.write(message + "\n")   # Append message with trailing newline
        log_file.flush()                 # Force OS buffer flush → survives crashes


def convert_actions(pred_action: np.ndarray) -> np.ndarray:
    """
    Convert a raw predicted action from the policy's output representation
    to the robot's native Euler-angle control space.

    The policy outputs actions in a hybrid representation:
        - XYZ end-effector position delta    (indices 0:3)
        - 6D continuous rotation             (indices 3:9)  ← rotation-6D (Zhou et al. 2019)
        - Gripper open/close command         (index -1)     ← scalar in [0, 1]

    6D rotation is the preferred differentiable rotation representation during
    training (avoids gimbal lock and discontinuities of Euler angles). At inference
    time, it must be decoded back to XYZ Euler angles for the simulator's
    `env.step()` API.

    Args:
        pred_action (np.ndarray): Raw action vector of shape (10,):
            [dx, dy, dz, r0, r1, r2, r3, r4, r5, gripper]

    Returns:
        np.ndarray: Converted action vector of shape (7,):
            [dx, dy, dz, roll, pitch, yaw, gripper]
            where roll/pitch/yaw are XYZ Euler angles in radians.

    Note:
        The conversion is performed on CPU after wrapping in a PyTorch tensor,
        because `TorchUtils.rot_6d_to_euler_angles` is a torch-based function.
    """
    cur_xyz     = pred_action[:3]           # End-effector position delta (3D Cartesian)
    cur_rot6d   = pred_action[3:9]          # 6D rotation representation (6 floats)
    cur_gripper = np.expand_dims(pred_action[-1], axis=0)  # Gripper scalar → shape (1,)

    # Convert 6D rotation to XYZ Euler angles using PyTorch utility
    cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)   # Shape: (1, 6)
    cur_euler = (
        TorchUtils.rot_6d_to_euler_angles(rot_6d=cur_rot6d, convention="XYZ")
        .squeeze()      # Remove batch dim → shape (3,)
        .numpy()        # Back to NumPy for env.step()
    )

    # Concatenate position, euler angles, and gripper into final 7D action
    pred_action = np.concatenate((cur_xyz, cur_euler, cur_gripper))

    return pred_action


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize the gripper dimension of an action from [0, 1] to [-1, 1] and
    optionally binarize to {-1, +1}.

    LIBERO environments use gripper values in [0, 1] (0=closed, 1=open), but
    many policy training pipelines represent gripper commands in [-1, 1] for
    symmetry with other action dimensions. This function rescales accordingly.

    The linear mapping applied is:
        normalized = 2 * (x - orig_low) / (orig_high - orig_low) - 1
                   = 2 * x - 1     (since orig_low=0, orig_high=1)

    Args:
        action (np.ndarray): Action array of any shape (…, action_dim).
            The last element (index -1) is treated as the gripper command.
        binarize (bool): If True, apply np.sign() after normalization, snapping
            continuous values to exactly {-1, 0, +1}. Defaults to True.

    Returns:
        np.ndarray: A copy of `action` with the gripper dimension re-scaled
            (and optionally binarized). The original array is not modified.

    Example:
        >>> normalize_gripper_action(np.array([0.1, 0.2, 0.3, 1.0]))
        array([ 0.1,  0.2,  0.3,  1.0])  # gripper 1.0 → 2*1-1=1.0 → sign → +1
    """
    normalized_action = action.copy()           # Avoid mutating the caller's array
    orig_low, orig_high = 0.0, 1.0             # Source range for gripper values

    # Linear rescaling: maps [0, 1] → [-1, 1]
    normalized_action[..., -1] = (
        2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    )

    if binarize:
        # Snap to {-1, 0, +1}; in practice eliminates ambiguous mid-range values
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Invert the gripper command sign in an action vector.

    Some data sources define gripper polarity differently (e.g., +1 = close vs.
    +1 = open). This function flips the sign of the last action dimension to
    reconcile mismatched conventions between dataset and environment.

    Args:
        action (np.ndarray): Action array of any shape (…, action_dim).
            The last element (index -1) is the gripper command.

    Returns:
        np.ndarray: A copy of `action` with the gripper dimension negated.
            All other dimensions are unchanged.

    Example:
        >>> invert_gripper_action(np.array([0.0, 0.0, 0.0, -1.0]))
        array([ 0.,  0.,  0.,  1.])
    """
    inverted_action = action.copy()        # Defensive copy; original is unchanged
    inverted_action[..., -1] *= -1.0      # Negate gripper dimension
    return inverted_action


# ──────────────────────────────────────────────────────────────────────────────
# Policy Class
# ──────────────────────────────────────────────────────────────────────────────

class llava_pythia_act_policy:
    """
    Inference wrapper for the LLaVA-Pythia Vision-Language-Action (VLA) policy.

    This class handles:
        1. Loading a pretrained LLaVA-Pythia checkpoint (with optional LoRA adapters).
        2. Preprocessing raw sensor observations (stereo camera images + robot joint state)
           into the token sequence format expected by the multimodal LLM.
        3. Providing a `process_batch_to_llava` method that returns a dict ready to pass
           directly to the model's forward pass.

    The policy does NOT manage its own inference loop; that is handled externally by
    `run_episode()`.

    Attributes:
        tokenizer: HuggingFace tokenizer for the Pythia LM backbone.
        policy: The full LlavaPythia model (LM + vision tower + action head).
        image_processor: CLIPImageProcessor (or similar) for image normalization.
        context_len (int): Maximum token sequence length the model supports.
        config (LlavaPythiaConfig): Full model configuration loaded from checkpoint dir.
        data_args: Optional dataset arguments (currently unused but reserved for future use).
        conv: Current conversation object (reset on each call to process_batch_to_llava).
    """

    def __init__(self, policy_config: dict, data_args=None):
        """
        Initialize the policy by loading the pretrained model.

        Args:
            policy_config (dict): Configuration dictionary containing:
                - "model_path" (str): Path to the checkpoint directory.
                - "model_base" (str): Path to the base model (used when LoRA is enabled).
                - "enable_lora" (bool): Whether LoRA adapters should be loaded.
                - "conv_mode" (str): Conversation template key (e.g., "pythia").
                - "action_head" (str): Action head type (e.g., "act", "droid_diffusion").
            data_args (optional): Dataset arguments passed from training configs.
                Currently unused but included for interface compatibility. Defaults to None.
        """
        super(llava_pythia_act_policy).__init__()   # Explicit MRO call (no-op for object)
        self.load_policy(policy_config)             # Delegate model loading
        self.data_args = data_args                  # Store dataset args for future use

    def load_policy(self, policy_config: dict):
        """
        Load a pretrained LLaVA-Pythia model from disk.

        This method sets:
            self.policy_config, self.tokenizer, self.policy,
            self.image_processor, self.context_len, self.config

        The model is loaded in evaluation mode with all weights frozen (unless LoRA
        layers are present, which are loaded as part of the checkpoint).

        Args:
            policy_config (dict): Same structure as described in __init__.

        Side effects:
            - Allocates model weights on GPU (via load_pretrained_model internals).
            - Reads LlavaPythiaConfig from the parent directory of model_path.
        """
        self.policy_config = policy_config

        # When LoRA is enabled, model_base is the frozen backbone path and
        # model_path contains only the delta adapter weights.
        model_base = policy_config["model_base"] if policy_config['enable_lora'] else None

        # Extract a human-readable model name from the directory path string
        model_name = get_model_name_from_path(policy_config['model_path'])
        model_path = policy_config["model_path"]

        # Load tokenizer, model, image_processor, and context window length
        # load_model_8bit=False, load_model_4bit=False → full-precision (fp16 via dtype)
        self.tokenizer, self.policy, self.image_processor, self.context_len = (
            load_pretrained_model(model_path, model_base, model_name, False, False)
        )

        # Load the model config from the checkpoint's parent folder
        # (config.json lives one directory above the checkpoint step folder)
        self.config = LlavaPythiaConfig.from_pretrained(
            '/'.join(model_path.split('/')[:-1]),   # Parent directory of checkpoint
            trust_remote_code=True                   # Allow custom config code
        )

    def process_batch_to_llava(
        self,
        curr_image: torch.Tensor,
        robo_state: torch.Tensor,
        raw_lang: str,
    ) -> dict:
        """
        Prepare a single-step inference batch for the LLaVA-Pythia model.

        This method handles the full preprocessing pipeline:
            1. Reset conversation history and construct the chat prompt.
            2. Split the stereo image tensor into left (agent-view) and right (wrist) frames.
            3. Pad both frames to square aspect ratio using the image processor's mean color.
            4. Normalize and convert frames to GPU tensors.
            5. Tokenize the prompt with image token placeholders.
            6. Assemble and return a batch dict compatible with the model's forward signature.

        Args:
            curr_image (torch.Tensor): Stereo image batch of shape (2, C, H, W) or
                (1, 2, C, H, W). The first chunk is the agent (global) view;
                the second is the wrist (ego) view.
            robo_state (torch.Tensor): Proprioceptive state vector of shape (state_dim,).
                Contains normalized EEF position, orientation (axis-angle), and gripper.
            raw_lang (str): Natural language task description / instruction.

        Returns:
            dict: Batch dictionary with keys:
                - "input_ids"      (torch.Tensor): Token IDs on CUDA, shape (1, seq_len).
                - "attention_mask" (torch.Tensor): Binary mask (non-pad tokens), same shape.
                - "images"         (torch.Tensor): Preprocessed agent-view image, (1, C, H', W').
                - "images_r"       (torch.Tensor): Preprocessed wrist-view image, (1, C, H', W').
                - "states"         (torch.Tensor): Robot state unsqueezed to (1, state_dim).
        """
        # Reset conversation template for this new inference step
        self.conv = conv_templates[self.policy_config['conv_mode']].copy()

        # Handle optional leading batch dimension (B=1) in the image tensor
        if len(curr_image.shape) == 5:
            curr_image = curr_image.squeeze(0)   # (1, 2, C, H, W) → (2, C, H, W)

        # Split stereo pair: first half = agent view, second half = wrist camera
        image, image_r = torch.chunk(curr_image, 2, dim=0)

        # ── Agent-view image preprocessing ──────────────────────────────────
        # Pad to square canvas (to avoid distortion from direct resize)
        image = self.expand2square(image, tuple(x for x in self.image_processor.image_mean))

        # Normalize: do_normalize=True applies ImageNet mean/std subtraction
        #            do_rescale=False  → values already in [0, 1] from ToTensor()
        #            do_center_crop=False → crop was handled by expand2square
        image_tensor = self.image_processor.preprocess(
            image,
            return_tensors='pt',
            do_normalize=True,
            do_rescale=False,
            do_center_crop=False
        )['pixel_values']
        image_tensor = image_tensor.to(self.policy.device, dtype=self.policy.dtype)

        # ── Wrist-view image preprocessing (identical pipeline) ─────────────
        image_r = self.expand2square(image_r, tuple(x for x in self.image_processor.image_mean))
        image_tensor_r = self.image_processor.preprocess(
            image_r,
            return_tensors='pt',
            do_normalize=True,
            do_rescale=False,
            do_center_crop=False
        )['pixel_values']
        image_tensor_r = image_tensor_r.to(self.policy.device, dtype=self.policy.dtype)

        # ── Prompt construction ──────────────────────────────────────────────
        inp = raw_lang   # Start with the raw language instruction

        assert image is not None, 'image must be provided.'

        # Wrap image token with optional start/end boundary tokens
        if self.policy.config.mm_use_im_start_end:
            # "<im_start><image><im_end>\n" + instruction
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            # "<image>\n" + instruction  (simpler, more common)
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

        # Add human turn to conversation and get the assistant response slot
        self.conv.append_message(self.conv.roles[0], inp)    # roles[0] = "USER" / "Human"
        image = None                                           # Image passed separately via tensor
        self.conv.append_message(self.conv.roles[1], None)    # roles[1] = "ASSISTANT" / "GPT"

        # Render full conversation to string (includes system prompt, delimiters, etc.)
        prompt = self.conv.get_prompt()
        prompt += " <|endoftext|>"    # Append EOS token to signal generation end

        # Tokenize with IMAGE_TOKEN_INDEX placeholders for vision tokens
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            )
            .unsqueeze(0)   # Add batch dimension: (seq_len,) → (1, seq_len)
            .cuda()         # Move to GPU
        )

        # Build attention mask: 1 for real tokens, 0 for padding tokens
        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Move robot state to policy device with matching dtype
        states = robo_state.to(self.policy.device, dtype=self.policy.dtype)

        # Assemble the batch dict expected by the model's forward() signature
        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attn_mask,
            images=image_tensor,
            images_r=image_tensor_r,
            states=states.unsqueeze(0)   # (state_dim,) → (1, state_dim)
        )

        return data_dict

    def expand2square(
        self,
        pil_imgs: torch.Tensor,
        background_color: tuple,
    ) -> torch.Tensor:
        """
        Pad a batch of images to a square canvas without distortion.

        For rectangular images, the shorter dimension is centered on a canvas
        filled with `background_color`. This preserves aspect ratio while
        satisfying square-input requirements of most vision encoders (ViT/CLIP).

        Args:
            pil_imgs (torch.Tensor): Batch of images, shape (B, C, H, W).
                Expected to be float tensors with values in [0, 1].
            background_color (tuple): RGB fill values, one per channel.
                Typically the image processor's channel-wise mean (e.g., (0.48, 0.46, 0.41)).

        Returns:
            torch.Tensor: Square-padded batch of shape (B, max_dim, max_dim, C),
                temporarily in HWC layout, then converted back to the input's
                dtype and device. Note: the channel ordering is preserved.

        Implementation note:
            The function uses NumPy for the canvas allocation (np.full) and
            converts back to torch at the end for compatibility with the
            image_processor's preprocess() method, which accepts torch tensors.
        """
        batch_size, channels, height, width = pil_imgs.shape
        max_dim = max(height, width)   # Target canvas side length

        # Allocate (B, max_dim, max_dim, C) float32 canvas filled with background color
        # np.full broadcasts background_color across the channel dimension
        expanded_imgs = np.full(
            (batch_size, max_dim, max_dim, channels),
            background_color,
            dtype=np.float32
        )

        # Convert from PyTorch (B, C, H, W) to NumPy (B, H, W, C) for array slicing
        if height == width:
            # Already square: no padding needed, just rearrange dims
            expanded_imgs = pil_imgs.permute(0, 2, 3, 1).cpu().numpy()
        elif height > width:
            # Tall image: pad left and right sides symmetrically
            offset = (max_dim - width) // 2   # Pixels of padding on each side
            expanded_imgs[:, :height, offset:offset + width, :] = (
                pil_imgs.permute(0, 2, 3, 1).cpu().numpy()
            )
        else:
            # Wide image: pad top and bottom symmetrically
            offset = (max_dim - height) // 2
            expanded_imgs[:, offset:offset + height, :width, :] = (
                pil_imgs.permute(0, 2, 3, 1).cpu().numpy()
            )

        # Convert back to torch tensor on the original device with original dtype
        expanded_imgs = torch.tensor(expanded_imgs).to(
            dtype=pil_imgs.dtype,
            device=pil_imgs.device
        )
        return expanded_imgs


# ──────────────────────────────────────────────────────────────────────────────
# Configuration Dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerateConfig:
    """
    Evaluation run configuration.

    All fields have defaults that constitute a sensible baseline configuration.
    Values can be overridden via CLI arguments (parsed by draccus) or programmatically.

    The `@draccus.wrap()` decorator on `run_libero_eval` reads these fields from
    `sys.argv` using the field names as flags (e.g., --model_path /path/to/ckpt).

    Sections:
        Model:       Paths and family identifier for the policy checkpoint.
        LIBERO:      Task suite, rollout count, image resolution, and task range.
        Utils:       Logging, wandb, random seed, and debug toggle.
        Command:     Linguistic command variation settings for ablation studies.
    """

    # ── Model parameters ─────────────────────────────────────────────────────
    model_path: str = (
        "/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/"
        "checkpoints_saving_folder/tinyvla/"
        "tiny_vla_llava_pythia_lora_libero_goal_no_noops_lora_r_64/checkpoint-54000"
    )
    """Absolute path to the specific checkpoint directory (e.g., checkpoint-54000/)."""

    model_base: str = (
        "/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/"
        "checkpoints_saving_folder/tinyvla/llava_pythia_libero_goal_no_noops_64/1.3B"
    )
    """Path to the base model used as the LoRA backbone (1.3B parameter Pythia variant)."""

    model_family: str = "tiny_vla"
    """Identifier for the model family; used by libero_utils for action formatting."""

    # ── LIBERO environment parameters ─────────────────────────────────────────
    task_suite_name: str = TaskSuite.LIBERO_GOAL
    """Name of the LIBERO task suite to evaluate. See TaskSuite enum for valid values."""

    num_steps_wait: int = 10
    """Number of no-op steps at episode start to allow physics to stabilize."""

    num_trials_per_task: int = 50
    """Number of independent rollout episodes per task (and per command level)."""

    initial_states_path: str = "DEFAULT"
    """
    Path to a JSON file of pre-recorded initial states, or "DEFAULT" to use
    the task suite's built-in initial states. Custom states enable fair comparison
    across methods by fixing starting configurations.
    """

    env_img_res: int = 256
    """
    Resolution (pixels, square) for environment rendering. Note: this is the
    render resolution, NOT the policy input resolution (which is determined by
    the image processor's expected input size).
    """

    task_range: str = "0-9"
    """
    Inclusive range of task indices to evaluate, formatted as "start-end".
    For example, "0-9" evaluates tasks 0 through 9 (10 tasks total).
    """

    # ── Utility parameters ────────────────────────────────────────────────────
    run_id_note: Optional[str] = None
    """Optional suffix appended to the auto-generated run ID for disambiguation."""

    local_log_dir: str = "./experiments/logs"
    """Directory where per-run text log files are written."""

    summary_file: Optional[str] = None
    """
    If set, path to a JSON file where final evaluation results are written after
    each command level completes. Enables partial result recovery if the run crashes.
    """

    checkpoint_size: int = 20000
    """Reserved for potential checkpoint-step-based filtering (currently informational)."""

    use_wandb: bool = False
    """Whether to log metrics and artifacts to Weights & Biases."""

    wandb_entity: str = "your-wandb-entity"
    """WandB workspace (entity) to log runs into. Override with your team/username."""

    wandb_project: str = "your-wandb-project"
    """WandB project name under which runs are grouped."""

    seed: int = 7
    """Global random seed for Python, NumPy, and PyTorch. Ensures reproducibility."""

    run_number: int = 0
    """Integer run identifier appended to video filenames for deduplication."""

    debug: bool = False
    """If True, attach a debugpy server on port 5678 and wait for a debugger connection."""

    local_rank: int = 0
    """Local process rank for potential future distributed evaluation (currently unused)."""

    # ── Command variation parameters ──────────────────────────────────────────
    change_command: bool = False
    """
    If True, replace the task's default BDDL language goal with a linguistic
    paraphrase (L1/L2/L3). The paraphrase files must exist alongside the
    original BDDL files.
    """

    command_level: Optional[str] = None
    """
    Which paraphrase level to use:
        - None or "default": Use the original task description.
        - "l1": Level-1 paraphrase (near-synonym substitution).
        - "l2": Level-2 paraphrase (moderate rephrasing).
        - "l3": Level-3 paraphrase (high-level goal description).
        - "all": Evaluate all four (None, l1, l2, l3) sequentially.
        - "all_no_default": Evaluate l1, l2, l3 (skip default).
        - "l1,l2": Comma-separated subset of levels.
    """

    selected_version: Optional[int] = None
    """
    If set, pin evaluation to a single paraphrase version number (e.g., 1, 2, 3).
    If None, all available version files matching the level pattern are tested.
    """


# ──────────────────────────────────────────────────────────────────────────────
# Observation Processing
# ──────────────────────────────────────────────────────────────────────────────

def get_obs(obs: dict, stats: dict) -> tuple:
    """
    Extract and preprocess visual and proprioceptive observations from an
    environment observation dictionary.

    The function:
        1. Resizes both camera views to (320, 180) for consistent downstream handling.
        2. Flips images vertically and horizontally (`[::-1, ::-1]`) to correct for
           LIBERO's coordinate system (images are stored upside-down in the sim buffer).
        3. Constructs a 9-dimensional proprioceptive state from EEF position (3),
           orientation in axis-angle (3), and gripper joint positions (2, though treated
           as one value in downstream code).
        4. Normalizes the state using pre-computed dataset statistics (z-score normalization).

    Args:
        obs (dict): Observation dictionary returned by `env.step()` or `env.reset()`.
            Expected keys:
                - "agentview_image":          np.ndarray (H, W, 3), agent camera RGB.
                - "robot0_eye_in_hand_image":  np.ndarray (H, W, 3), wrist camera RGB.
                - "robot0_eef_pos":            np.ndarray (3,), end-effector XYZ position.
                - "robot0_eef_quat":           np.ndarray (4,), end-effector quaternion (w,x,y,z).
                - "robot0_gripper_qpos":       np.ndarray (2,), gripper joint positions.
        stats (dict): Dataset statistics loaded from `dataset_stats.pkl`.
            Expected keys:
                - "qpos_mean": np.ndarray (state_dim,), per-dimension mean.
                - "qpos_std":  np.ndarray (state_dim,), per-dimension standard deviation.

    Returns:
        tuple:
            - images (np.ndarray): Stacked camera views, shape (2, 180, 320, 3),
              dtype uint8. Index 0 = agent view, index 1 = wrist view.
            - states (np.ndarray): Normalized proprioceptive state, shape (state_dim,),
              dtype float32.
    """
    # Collect and resize both camera views
    # [::-1, ::-1] flips both axes to correct LIBERO's inverted image buffer
    images = np.array([
        cv2.resize(obs['agentview_image'][::-1, ::-1], (320, 180)),          # Agent view
        cv2.resize(obs['robot0_eye_in_hand_image'][::-1, ::-1], (320, 180))  # Wrist view
    ])

    # Build proprioceptive state from EEF pose and gripper reading
    states = np.concatenate((
        obs["robot0_eef_pos"],                              # XYZ position (3,)
        quat2axisangle(obs["robot0_eef_quat"]),             # Orientation: quat → axis-angle (3,)
        obs["robot0_gripper_qpos"]                          # Gripper joint positions (2,)
    ))

    # Z-score normalize: bring state into the distribution seen during training
    states = (states - stats["qpos_mean"]) / stats["qpos_std"]

    return images, states


# ──────────────────────────────────────────────────────────────────────────────
# Command Level Parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_command_levels(cfg: GenerateConfig) -> list:
    """
    Resolve the `command_level` configuration option into a list of concrete
    level identifiers to evaluate sequentially.

    Supports preset aliases ("all", "all_no_default", "default"), comma-separated
    lists, and single level strings. Always returns a list, even for a single level.

    Args:
        cfg (GenerateConfig): Evaluation configuration. Relevant fields:
            - `change_command` (bool): If False, always returns [None] (use default command).
            - `command_level` (Optional[str]): Level specification (see GenerateConfig docstring).

    Returns:
        list: Ordered list of level identifiers to evaluate. Each element is either:
            - None  → use the original task description (no paraphrase).
            - "l1"  → Level-1 linguistic paraphrase.
            - "l2"  → Level-2 linguistic paraphrase.
            - "l3"  → Level-3 linguistic paraphrase.

    Examples:
        >>> parse_command_levels(cfg_with_change_command=False)
        [None]
        >>> parse_command_levels(cfg_with_command_level="all")
        [None, "l1", "l2", "l3"]
        >>> parse_command_levels(cfg_with_command_level="l1,l3")
        ["l1", "l3"]
        >>> parse_command_levels(cfg_with_command_level="l2")
        ["l2"]
    """
    # If command variation is disabled, always use the original description
    if not cfg.change_command or cfg.command_level is None:
        return [None]

    # Preset multi-level aliases
    presets = {
        "all":            [None, "l1", "l2", "l3"],    # All levels including default
        "all_no_default": ["l1", "l2", "l3"],           # Paraphrase levels only
        "default":        [None],                        # Explicitly request default
    }
    if cfg.command_level in presets:
        return presets[cfg.command_level]

    # Comma-separated list: "l1,l2" → ["l1", "l2"]
    if "," in cfg.command_level:
        return [l.strip() for l in cfg.command_level.split(",")]

    # Single level string: "l2" → ["l2"]
    return [cfg.command_level]


# ──────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(cfg: GenerateConfig) -> tuple:
    """
    Initialize per-run logging infrastructure: create a text log file and
    optionally initialize a Weights & Biases run.

    The run ID is constructed from task suite, model family, and timestamp to
    ensure uniqueness across concurrent or repeated runs.

    Args:
        cfg (GenerateConfig): Evaluation configuration. Relevant fields:
            - `task_suite_name`:  Included in the run ID string.
            - `model_family`:     Included in the run ID string.
            - `run_id_note`:      Optional suffix for manual disambiguation.
            - `change_command`:   If True, command_level is appended to run ID.
            - `command_level`:    Level label appended to run ID when change_command=True.
            - `local_log_dir`:    Directory in which the log file is created.
            - `use_wandb`:        If True, initialize wandb with entity/project/name.
            - `wandb_entity`:     WandB workspace name.
            - `wandb_project`:    WandB project name.

    Returns:
        tuple:
            - log_file (file):    Open text file handle for appending log messages.
            - local_log_filepath (str): Absolute path to the log file on disk.
            - run_id (str):       The constructed unique run identifier string.

    Side effects:
        - Creates `cfg.local_log_dir` if it does not exist.
        - Opens a new `.txt` file (will overwrite if the run ID collides).
        - Calls `wandb.init()` if `cfg.use_wandb` is True.
    """
    # Build unique run identifier: EVAL-<suite>-<family>-<timestamp>
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"          # Append user note (e.g., "noaug")

    # Append command level to distinguish level-specific log files
    if cfg.change_command and cfg.command_level:
        run_id += f"--{cfg.command_level}"

    # Ensure output directory exists (creates parent dirs as needed)
    os.makedirs(cfg.local_log_dir, exist_ok=True)

    # Open log file for writing (mode "w" truncates any existing file)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Optionally initialize wandb experiment tracking
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,                 # Human-readable run name in the dashboard
        )

    return log_file, local_log_filepath, run_id


# ──────────────────────────────────────────────────────────────────────────────
# Initial State Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_initial_states(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    log_file=None,
) -> tuple:
    """
    Load the set of initial environment states to use for a given task.

    Two modes are supported:
        1. DEFAULT: Use the task suite's built-in initial states (one per trial index).
        2. CUSTOM:  Load states from a JSON file at `cfg.initial_states_path`, which may
                    contain filtered states from successful expert demonstrations (useful
                    for fair comparison studies where failed demos are excluded).

    Args:
        cfg (GenerateConfig): Evaluation config. Relevant field: `initial_states_path`.
        task_suite: LIBERO TaskSuite object (e.g., LiberoGoal). Provides
            `get_task_init_states(task_id)` which returns a list of state dicts.
        task_id (int): Zero-based index of the task within the suite.
        log_file (file-like, optional): Open log file for dual-sink logging.

    Returns:
        tuple:
            - initial_states (list): List of state dicts from the task suite's built-in
              collection. Always returned regardless of mode.
            - all_initial_states (dict or None): Parsed JSON dict from the custom file,
              or None if using DEFAULT mode.
    """
    # Always load the default initial states as the primary/fallback source
    initial_states = task_suite.get_task_init_states(task_id)

    if cfg.initial_states_path != "DEFAULT":
        # Load custom states from JSON (e.g., filtered successful expert demos)
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


# ──────────────────────────────────────────────────────────────────────────────
# Episode Execution
# ──────────────────────────────────────────────────────────────────────────────

def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    policy: llava_pythia_act_policy,
    policy_config: dict,
    resize_size: int,
    initial_state=None,
    log_file=None,
) -> tuple:
    """
    Execute a single evaluation episode and return success status and trajectory data.

    The episode lifecycle:
        1. Reset environment and optionally set a specific initial state.
        2. Wait `cfg.num_steps_wait` steps for physics stabilization (no-ops).
        3. At each timestep:
            a. Extract observations (images + robot state).
            b. Query policy at every `query_frequency` steps.
            c. Apply temporal action aggregation (exponential weighting over past predictions).
            d. Post-process raw action (de-normalize or de-scale).
            e. Convert action from training representation to env's Euler-angle space.
            f. Step the environment and check for task completion.
        4. Return success flag and trajectory dict (for video saving and debugging).

    Temporal Aggregation:
        When `temporal_agg=True`, the policy produces a chunk of `num_queries` future
        actions at each query step. These are stored in an `all_time_actions` buffer of
        shape (max_timesteps, max_timesteps + num_queries, action_dim). At each step t,
        actions predicted for step t from all past query steps are retrieved and combined
        using exponentially-decaying weights (more recent predictions receive more weight):
            weight_i = exp(-k * i),  k = 0.01,  i = 0 (most recent) → N-1 (oldest)
        This improves temporal consistency and reduces jitter.

    Args:
        cfg (GenerateConfig): Evaluation configuration.
        env: LIBERO gym-style environment with step(), reset(), set_init_state() APIs.
        task_description (str): Natural language goal for this episode (given to policy).
        policy (llava_pythia_act_policy): Loaded VLA policy wrapper.
        policy_config (dict): Policy configuration dict (action_head, model_path, etc.).
        resize_size (int): Target square resolution for policy image inputs (typically 224).
        initial_state (np.ndarray or None): Serialized MuJoCo state to restore.
            If None, the environment is reset randomly.
        log_file (file-like, optional): Log file for error reporting.

    Returns:
        tuple:
            - success (bool): True if the environment returned done=True before max steps.
            - replay_traj (dict): Trajectory data dictionary with keys:
                - "images" (list of np.ndarray): Per-step agent-view frames at 256×256.
                - "task_command" (str): The task description used in this episode.
                - "states" (list of np.ndarray): Per-step proprioceptive states.
                - "actions" (list of np.ndarray): Per-step executed actions (7D).

    Raises:
        Any exception during environment interaction is caught internally; on exception,
        success is set to False and the partial trajectory is still returned.
    """
    env.reset()
    to_tensor = transforms.ToTensor()   # Converts HWC uint8 [0,255] → CHW float [0,1]

    # ── Initial state setup ───────────────────────────────────────────────────
    if initial_state is not None:
        # Restore a specific MuJoCo state (deterministic evaluation)
        obs = env.set_init_state(initial_state)
    else:
        obs = env.reset()
        # TinyVLA-specific fix: the basket object sometimes falls through the table
        # on reset; manually set its position and wait for stabilization.
        if 'basket_1_pos' in obs.keys():
            basket_pos  = [0.005, 0.261, 0.035]                # Target XYZ position
            basket_quat = [0.000, 0.000, 0.000, 1.000]         # Unit quaternion (no rotation)
            env.sim.data.set_joint_qpos(
                env.env.objects_dict['basket_1'].joints[0],
                np.concatenate((basket_pos, basket_quat))      # 7D joint state
            )
            t = 0
            while t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1

    # ── Policy-specific configuration ─────────────────────────────────────────
    if policy_config["action_head"] == 'act':
        rand_crop_resize = False     # ACT head: no random crop at inference
        temporal_agg = True          # Use temporal aggregation
    else:
        rand_crop_resize = True      # Diffusion head: random crop-resize for robustness
        temporal_agg = True

    action_dim = policy.config.action_dim   # Dimensionality of the action space (e.g., 7)
    policy.policy.eval()                     # Disable dropout, batch-norm tracking, etc.

    # ── Load dataset normalization statistics ──────────────────────────────────
    # stats.pkl is expected one directory above the checkpoint folder
    stats_path = os.path.join(
        "/".join(policy_config['model_path'].split('/')[:-1]),
        'dataset_stats.pkl'
    )
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)    # Contains qpos_mean, qpos_std, action_mean, action_std, etc.

    # ── Post-processing function for raw policy output ─────────────────────────
    if policy_config["action_head"] == 'act':
        # ACT: actions are z-scored; reverse normalization to get physical values
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif policy_config["action_head"] == 'transformer_diffusion':
        # Diffusion: actions are in [-1, 1]; map back to [action_min, action_max]
        post_process = lambda a: (
            ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
        )
    elif policy_config["action_head"] == 'droid_diffusion':
        # Same linear rescaling as transformer_diffusion
        post_process = lambda a: (
            ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
        )

    # ── Temporal aggregation parameters ───────────────────────────────────────
    query_frequency = policy.config.chunk_size / 2    # Default: query every half-chunk
    if temporal_agg:
        query_frequency = 1                            # Query every step when aggregating
        num_queries = policy.config.chunk_size         # Number of future steps predicted

    max_timesteps = int(200)    # Hard cap on episode length (policy-side; env may differ)

    # Pre-allocate circular action buffer for temporal aggregation
    if temporal_agg:
        all_time_actions = torch.zeros(
            [max_timesteps, max_timesteps + num_queries, action_dim],
            dtype=torch.float32
        ).cuda()
        # Shape: [T_query, T_target, action_dim]
        # all_time_actions[t_q, t_q:t_q+chunk] = actions predicted at query step t_q

    # ── Episode trajectory storage ─────────────────────────────────────────────
    t = 0
    replay_traj       = dict()
    image_list        = []          # List of 256×256 agent-view frames (for video)
    robot_state_list  = []          # List of raw (unnormalized) proprioceptive states
    target_action_list = []         # List of executed 7D actions
    success           = False

    with torch.inference_mode():    # Disable gradient computation for efficiency
        try:
            # ── Stabilization warm-up (no-op steps) ───────────────────────────
            while t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1

            t = 0   # Reset timestep counter after warm-up

            # ── Main control loop ──────────────────────────────────────────────
            while t < max_timesteps:
                # Extract visual and proprioceptive observations
                traj_rgb_np, robot_state = get_obs(obs=obs, stats=stats)

                # Store agent-view frame at 256×256 for video rendering
                image_list.append(cv2.resize(traj_rgb_np[0], (256, 256)))
                robot_state_list.append(robot_state)    # Store normalized state

                # Convert robot state to torch tensor on GPU
                robot_state = torch.from_numpy(robot_state).float().cuda()

                # ── Image preprocessing (at query frequency) ──────────────────
                if t % query_frequency == 0:
                    curr_image = []
                    for img in traj_rgb_np:
                        # ToTensor: (H, W, C) uint8 [0,255] → (C, H, W) float [0,1]
                        curr_image.append(to_tensor(img).float().cuda())
                    curr_image = torch.stack(curr_image, dim=0)   # (2, C, H, W)

                    # Optional: random crop-resize for data augmentation at test time
                    if rand_crop_resize:
                        original_size = curr_image.shape[-2:]    # (H, W)
                        ratio = 0.95     # Keep 95% of the image (5% cropped from edges)
                        # Compute crop indices: center-crop to ratio * size
                        curr_image = curr_image[
                            :, :,
                            int(original_size[0] * (1 - ratio) / 2)  # top
                            : int(original_size[0] * (1 + ratio) / 2),  # bottom
                            int(original_size[1] * (1 - ratio) / 2)  # left
                            : int(original_size[1] * (1 + ratio) / 2),  # right
                        ]
                        curr_image = curr_image.squeeze(0)
                        resize_transform = transforms.Resize(original_size, antialias=True)
                        curr_image = resize_transform(curr_image)  # Restore original size
                        curr_image = curr_image.unsqueeze(0)

                # ── Policy warm-up (first step only) ──────────────────────────
                if t == 0:
                    # Run 10 forward passes to warm up CUDA kernels and caches.
                    # This prevents the first real inference from being slow/timed.
                    for _ in range(10):
                        batch = policy.process_batch_to_llava(curr_image, robot_state, task_description)
                        policy.policy(**batch, eval=True)

                # ── Policy query ───────────────────────────────────────────────
                if policy_config['action_head'] in ["act", "droid_diffusion"]:
                    if t % query_frequency == 0:
                        # Prepare batch and run model forward pass
                        batch = policy.process_batch_to_llava(curr_image, robot_state, task_description)
                        all_actions = policy.policy(**batch, eval=True)
                        # all_actions shape: (1, chunk_size, action_dim)

                    if temporal_agg:
                        # Store this query's predictions in the action buffer
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        # Retrieve all past predictions targeting the current step t
                        actions_for_curr_step = all_time_actions[:, t]  # (max_T, action_dim)
                        # Filter out time steps that haven't been queried yet (still zero)
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        # Compute exponentially decaying weights (most recent = highest weight)
                        k = 0.01    # Decay rate: smaller k → slower decay → more history
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()   # Normalize to sum=1
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        # Weighted sum of past predictions
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        # No aggregation: take the action at the current position in the chunk
                        raw_action = all_actions[:, t % query_frequency]
                else:
                    raise NotImplementedError(
                        f"Action head '{policy_config['action_head']}' not supported."
                    )

                # ── Action post-processing and execution ──────────────────────
                raw_action = raw_action.squeeze(0).cpu().numpy()  # GPU tensor → NumPy (action_dim,)
                action = post_process(raw_action)                  # De-normalize to physical units
                action = convert_actions(action)                   # 6D rotation → Euler angles

                # Step the environment with the 7D action (list format required by LIBERO)
                obs, reward, done, info = env.step(action.tolist())
                target_action_list.append(action)   # Log executed action

                if done:
                    success = True
                    break    # Task completed successfully; end episode early
                t += 1

        except Exception as e:
            # Catch all exceptions (e.g., physics divergence, MuJoCo errors)
            log_message(f"Episode error: {e}", log_file)
            success = False

    # ── Package trajectory for video saving and analysis ──────────────────────
    replay_traj['images']       = image_list
    replay_traj['task_command'] = task_description
    replay_traj['states']       = robot_state_list
    replay_traj['actions']      = target_action_list

    return success, replay_traj


# ──────────────────────────────────────────────────────────────────────────────
# Task-Level Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    policy: llava_pythia_act_policy,
    policy_config: dict,
    log_file,
    total_episodes: int = 0,
    total_successes: int = 0,
) -> tuple:
    """
    Run evaluation for a single task across all configured paraphrase versions.

    For each requested paraphrase version (or the default command):
        1. Locate the corresponding BDDL file (if using command variation).
        2. Instantiate the LIBERO environment with the appropriate language goal.
        3. Run `cfg.num_trials_per_task` episodes using `run_episode()`.
        4. Save rollout videos and log results.
        5. Aggregate version-level and task-level success metrics.

    When `cfg.selected_version is None` and multiple version files exist, all versions
    are evaluated sequentially and their results are reported both individually and in
    aggregate.

    Args:
        cfg (GenerateConfig): Evaluation configuration.
        task_suite: LIBERO TaskSuite object.
        task_id (int): Zero-based task index within the suite.
        policy (llava_pythia_act_policy): Loaded VLA policy.
        policy_config (dict): Policy configuration dict.
        log_file (file-like): Open log file handle.
        total_episodes (int): Running total of episodes across all previous tasks.
            Used for computing global success rate in log messages.
        total_successes (int): Running total of successes across all previous tasks.

    Returns:
        tuple:
            - total_episodes (int): Updated total after this task's episodes.
            - total_successes (int): Updated total successes after this task.
            - task_description (str): The task description used in the last version tested
              (for use as a dict key in `run_libero_eval`).
            - task_success_rate (float): Overall success rate for this task across all
              versions and episodes.
            - total_task_episodes (int): Total episodes run for this task (all versions).
    """
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # ── Find paraphrase version files ─────────────────────────────────────────
    available_versions = []
    if cfg.change_command and cfg.command_level is not None:
        # Extract the base BDDL filename without extension
        base_name = os.path.splitext(os.path.basename(task.bddl_file))[0]

        # Locate the folder containing BDDL files for this task
        try:
            from libero.libero import get_libero_path
            bddl_folder = os.path.join(get_libero_path("bddl_files"), task.problem_folder)
        except Exception:
            bddl_folder = os.path.dirname(task.bddl_file)    # Fallback: same directory

        # Pattern for paraphrase files: <base>_syn_<level>_v<N>.bddl
        pattern = f"{base_name}_syn_{cfg.command_level}_v"

        try:
            import re
            for filename in os.listdir(bddl_folder):
                # Match files that contain the pattern and end in .bddl
                if pattern.lower() in filename.lower() and filename.endswith('.bddl'):
                    match = re.search(r'_v(\d+)', filename, re.IGNORECASE)
                    if match:
                        version_num = int(match.group(1))    # Extract version number
                        available_versions.append((version_num, filename))
        except Exception as e:
            log_message(f"Warning: Could not list version files: {e}", log_file)

        available_versions.sort()    # Sort by version number for deterministic ordering

    # ── Determine which versions to test ──────────────────────────────────────
    if cfg.selected_version is not None:
        versions_to_test = [cfg.selected_version]         # User pinned a specific version
    elif available_versions:
        versions_to_test = [v[0] for v in available_versions]  # Test all found versions
    else:
        versions_to_test = [None]                          # No versions found; use default

    log_message("=" * 80, log_file)
    log_message(f"TASK {task_id + 1}/{task_suite.n_tasks}", log_file)
    log_message(f"Versions to test: {versions_to_test}", log_file)
    log_message("=" * 80, log_file)

    task_results_per_version = {}    # Accumulates results per version label

    # ── Loop over versions ─────────────────────────────────────────────────────
    for version_to_test in versions_to_test:

        # Resolve the BDDL file path for this specific version
        ablation_bddl_file = None
        if available_versions and version_to_test is not None:
            selected_files = [v[1] for v in available_versions if v[0] == version_to_test]
            if selected_files:
                ablation_bddl_file = os.path.join(bddl_folder, selected_files[0])

        # Instantiate environment with correct language goal and BDDL file
        env, task_description, original_description = get_libero_env(
            task,
            cfg.model_family,
            change_command=cfg.change_command,
            command_level=cfg.command_level,
            ablation_bddl_file=ablation_bddl_file,
            resolution=cfg.env_img_res
        )

        # Human-readable version label for logging
        version_label = f"v{version_to_test}" if version_to_test is not None else "default"
        log_message("=" * 80, log_file)
        log_message(f"Testing VERSION: {version_label}", log_file)
        log_message(f"Original Command: {original_description}", log_file)
        log_message(f"Variation Command: {task_description}", log_file)
        log_message("=" * 80, log_file)

        task_episodes  = 0
        task_successes = 0

        # ── Episode loop for this version ──────────────────────────────────────
        for episode_idx in tqdm.tqdm(
            range(cfg.num_trials_per_task),
            desc=f"Version {version_label}"
        ):
            log_message(
                f"\n[{version_label}] Episode {episode_idx + 1}/{cfg.num_trials_per_task}",
                log_file
            )

            # Determine initial state for this episode
            if cfg.initial_states_path == "DEFAULT":
                initial_state = initial_states[episode_idx]   # Built-in state by index
            else:
                # Custom state file: look up by task description and episode key
                initial_states_task_key = task_description.replace(" ", "_")
                episode_key = f"demo_{episode_idx}"

                # Skip episodes where the expert demo itself failed
                if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                    log_message(f"Skipping episode {episode_idx} (failed expert demo)", log_file)
                    continue

                initial_state = np.array(
                    all_initial_states[initial_states_task_key][episode_key]["initial_state"]
                )

            log_message(f"Starting episode {task_episodes + 1}...", log_file)

            # Run a single episode and collect results
            success, replay_traj = run_episode(
                cfg, env, task_description, policy, policy_config, 224, initial_state, log_file
            )

            # Update episode and success counters
            task_episodes   += 1
            total_episodes  += 1
            if success:
                task_successes  += 1
                total_successes += 1

            # Save episode video to disk
            save_rollout_video(
                replay_traj,
                total_episodes,
                success=success,
                task_description=task_description,
                log_file=log_file,
                dataset_name=cfg.task_suite_name,
                run=cfg.run_number,
                change_command=cfg.change_command,
                command_level=cfg.command_level
            )

            # Log running statistics after each episode
            log_message(f"Success: {success}", log_file)
            log_message(f"Total episodes so far: {total_episodes}", log_file)
            log_message(
                f"Total successes: {total_successes} "
                f"({total_successes / total_episodes * 100:.1f}%)",
                log_file
            )

        # ── Version-level summary ──────────────────────────────────────────────
        version_success_rate = (
            float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
        )
        log_message(f"\n{'=' * 80}", log_file)
        log_message(f"VERSION {version_label} RESULTS:", log_file)
        log_message(f"  Episodes: {task_episodes}", log_file)
        log_message(f"  Successes: {task_successes}", log_file)
        log_message(f"  Success Rate: {version_success_rate:.1%}", log_file)
        log_message(f"{'=' * 80}\n", log_file)

        task_results_per_version[version_label] = {
            'success_rate': version_success_rate,
            'episodes':     task_episodes,
            'successes':    task_successes
        }

        # Clean up environment resources
        try:
            env.close()
        except Exception:
            pass    # Silently ignore cleanup errors (MuJoCo sometimes raises on close)

    # ── Task-level aggregation across all versions ─────────────────────────────
    total_task_episodes  = sum(r['episodes']  for r in task_results_per_version.values())
    total_task_successes = sum(r['successes'] for r in task_results_per_version.values())
    task_success_rate = (
        float(total_task_successes) / float(total_task_episodes)
        if total_task_episodes > 0 else 0.0
    )

    # Print version breakdown only when multiple versions were tested
    if len(versions_to_test) > 1:
        log_message("=" * 80, log_file)
        log_message("SUMMARY BY VERSION:", log_file)
        log_message("-" * 80, log_file)
        for version_label, results in task_results_per_version.items():
            sr   = results['success_rate']
            succ = results['successes']
            eps  = results['episodes']
            log_message(f"  {version_label:>10}: {sr:.1%} ({succ}/{eps} episodes)", log_file)
        log_message("-" * 80, log_file)
        log_message(
            f"  {'OVERALL':>10}: {task_success_rate:.1%} "
            f"({total_task_successes}/{total_task_episodes} episodes)",
            log_file
        )
        log_message("=" * 80, log_file)
    else:
        log_message("=" * 80, log_file)
        log_message(
            f"TASK SUCCESS RATE: {task_success_rate:.1%} "
            f"({total_successes}/{total_episodes} episodes)",
            log_file
        )
        log_message("=" * 80, log_file)

    return total_episodes, total_successes, task_description, task_success_rate, total_task_episodes


# ──────────────────────────────────────────────────────────────────────────────
# Results Display
# ──────────────────────────────────────────────────────────────────────────────

def print_results_table(
    task_results: dict,
    command_levels: list,
    all_results: dict,
) -> None:
    """
    Print a formatted ASCII summary table of evaluation results to stdout.

    Supports two display modes:
        1. Single level: Shows task name, success rate with raw counts, and episode count.
        2. Multiple levels: Shows a matrix of tasks × command levels with per-cell counts.

    Both modes include an "OVERALL" footer row and a summary by command level.

    Args:
        task_results (dict): Nested dict: {level_name: {task_name: {success_rate, episodes}}}.
            Task names are the natural language descriptions of each task.
        command_levels (list): Ordered list of command level identifiers as originally
            passed to parse_command_levels() (may include None for the default level).
        all_results (dict): Dict: {level_name: {success_rate, total_episodes, total_successes}}.
            Contains aggregated statistics across all tasks for each command level.

    Returns:
        None (prints to stdout).
    """
    print("\n" + "=" * 100)
    print("DETAILED RESULTS TABLE")
    print("=" * 100)

    # Normalize None → "default" for display consistency
    first_level  = command_levels[0]
    level_name   = first_level if first_level is not None else "default"
    task_names   = list(task_results[level_name].keys())
    level_names  = [l if l is not None else "default" for l in command_levels]

    if len(level_names) == 1:
        # ── Single-level table: task | success rate (X/Y) | episodes ──────────
        print(f"{'Task':<50} | {'Success Rate':>20} | {'Episodes':>8}")
        print("-" * 100)

        for task_name in task_names:
            result   = task_results[level_names[0]][task_name]
            sr       = result['success_rate']
            eps      = result['episodes']
            successes = int(sr * eps)       # Recover integer success count from rate
            print(f"{task_name:<50} | {sr:>11.1%} ({successes:>2}/{eps:<2}) | {eps:>8}")

        print("-" * 100)

        # Overall row
        overall_sr   = all_results[level_names[0]]['success_rate']
        overall_succ = all_results[level_names[0]]['total_successes']
        overall_eps  = all_results[level_names[0]]['total_episodes']
        print(f"{'OVERALL':<50} | {overall_sr:>11.1%} ({overall_succ:>3}/{overall_eps:<3}) | {overall_eps:>8}")

    else:
        # ── Multi-level matrix table: task | L1 | L2 | L3 | … ────────────────
        header = f"{'Task':<40}"
        for lname in level_names:
            header += f" | {lname.upper():>20}"
        print(header)
        print("-" * (41 + len(level_names) * 24))

        for task_name in task_names:
            row = f"{task_name:<40}"
            for lname in level_names:
                if task_name in task_results[lname]:
                    result    = task_results[lname][task_name]
                    sr        = result['success_rate']
                    eps       = result['episodes']
                    successes = int(sr * eps)
                    row += f" | {sr:>6.1%} ({successes:>2}/{eps:<2})"
                else:
                    row += f" | {'N/A':>20}"    # Task was not evaluated for this level
            print(row)

        print("-" * (41 + len(level_names) * 24))

        # Overall row across all tasks for each level
        overall_row = f"{'OVERALL':<40}"
        for lname in level_names:
            result = all_results[lname]
            sr     = result['success_rate']
            succ   = result['total_successes']
            total  = result['total_episodes']
            overall_row += f" | {sr:>6.1%} ({succ:>3}/{total:<3})"
        print(overall_row)

    print("=" * 100)

    # ── Summary by command level ───────────────────────────────────────────────
    print("\nSUMMARY BY COMMAND LEVEL:")
    print("-" * 60)
    for lname in level_names:
        result = all_results[lname]
        sr     = result['success_rate']
        succ   = result['total_successes']
        total  = result['total_episodes']
        print(f"  {lname.upper():>15}: {sr:.1%} ({succ}/{total} episodes)")
    print("=" * 100)


# ──────────────────────────────────────────────────────────────────────────────
# Main Evaluation Entry Point
# ──────────────────────────────────────────────────────────────────────────────

@draccus.wrap()
def run_libero_eval(cfg: GenerateConfig) -> float:
    """
    Main evaluation orchestrator for TinyVLA on LIBERO benchmarks.

    This function is the top-level entry point, decorated with `@draccus.wrap()` which
    automatically populates `cfg` from command-line arguments matching GenerateConfig fields.

    Execution flow:
        1. (Optional) Attach debugpy for remote debugging.
        2. Instantiate the LLaVA-Pythia policy from the specified checkpoint.
        3. Load the LIBERO task suite and determine the evaluation task range.
        4. For each command level (default / l1 / l2 / l3):
            a. Set up logging (text file + optional wandb).
            b. Iterate over tasks in the specified range.
            c. For each task, call `run_task()` which runs N episodes per version.
            d. Aggregate and log success metrics.
            e. Optionally save a JSON summary file.
        5. Print the final results table.
        6. Return the overall success rate.

    Args:
        cfg (GenerateConfig): Populated by draccus from sys.argv. See GenerateConfig
            for the full list of configurable fields.

    Returns:
        float: Mean success rate across command levels (when multiple levels are tested),
            or the single-level success rate (when only one level is evaluated).

    Side effects:
        - Writes text log files to `cfg.local_log_dir`.
        - Writes rollout videos to the directory managed by `save_rollout_video`.
        - Optionally writes a JSON summary to `cfg.summary_file`.
        - Optionally logs metrics and uploads log files to WandB.
    """
    # ── Optional debugpy remote debugging ─────────────────────────────────────
    if cfg.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))    # Listen on all interfaces, port 5678
        print("Waiting for debugger attach")
        debugpy.wait_for_client()             # Block until VS Code / PyCharm connects

    # ── Policy instantiation ───────────────────────────────────────────────────
    action_head = 'droid_diffusion'    # Hard-coded action head type for this evaluation
    policy_config = {
        "model_path":   f"{cfg.model_path}",
        "model_base":   f"{cfg.model_base}",
        "enable_lora":  True,                 # Always use LoRA adapters in this setup
        "conv_mode":    "pythia",             # Conversation template key
        "action_head":  action_head,
    }

    set_seed_everywhere(cfg.seed)    # Set all random seeds for reproducibility
    policy = llava_pythia_act_policy(policy_config)    # Load model onto GPU

    # ── Task suite initialization ──────────────────────────────────────────────
    benchmark_dict = benchmark.get_benchmark_dict()    # All registered LIBERO task suites
    task_suite = benchmark_dict[cfg.task_suite_name]() # Instantiate selected suite
    num_tasks  = task_suite.n_tasks
    print(f"Evaluating {num_tasks} tasks in {cfg.task_suite_name}")

    # Parse task range "start-end" string into integer bounds
    task_start, task_end = map(int, cfg.task_range.split('-'))
    print(f"Task range: {task_start}-{task_end}")

    # ── Command level resolution ───────────────────────────────────────────────
    command_levels = parse_command_levels(cfg)    # e.g., [None, "l1", "l2", "l3"]

    all_results  = {}    # {level_name: {success_rate, total_episodes, total_successes}}
    task_results = {}    # {level_name: {task_name: {success_rate, episodes}}}

    # ── Main evaluation loop: iterate over command levels ─────────────────────
    for level in command_levels:
        current_level_name = level if level is not None else "default"
        cfg.command_level  = level
        cfg.change_command = (level is not None)    # Disable command variation for default

        # Re-seed for each level to make levels independently reproducible
        set_seed_everywhere(cfg.seed)

        # Initialize log file and optional wandb run for this level
        log_file, local_log_filepath, run_id = setup_logging(cfg)

        log_message("=" * 80, log_file)
        log_message(f"EVALUATING: {current_level_name.upper()}", log_file)
        log_message("=" * 80, log_file)

        task_results[current_level_name] = {}    # Results dict for this level

        total_episodes = 0
        total_successes = 0

        # ── Task loop ──────────────────────────────────────────────────────────
        for task_id in tqdm.tqdm(
            range(task_start, min(task_end + 1, num_tasks)),
            desc=f"Level {current_level_name}"
        ):
            # Run all episodes for this task and collect metrics
            total_episodes, total_successes, task_name, task_sr, task_eps = run_task(
                cfg, task_suite, task_id, policy, policy_config,
                log_file, total_episodes, total_successes
            )

            # Store per-task results for this level
            task_results[current_level_name][task_name] = {
                'success_rate': task_sr,
                'episodes':     task_eps
            }

        # ── Level-level summary ────────────────────────────────────────────────
        final_success_rate = (
            float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
        )
        all_results[current_level_name] = {
            'success_rate':    final_success_rate,
            'total_episodes':  total_episodes,
            'total_successes': total_successes
        }

        log_message("=" * 80, log_file)
        log_message(f"RESULTS FOR {current_level_name.upper()}:", log_file)
        log_message(
            f"Success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)",
            log_file
        )
        log_message("=" * 80, log_file)

        # ── Optional partial JSON summary ──────────────────────────────────────
        if cfg.summary_file:
            summary_data = {
                'task_range':    f"{task_start}-{task_end}",
                'task_results':  task_results[current_level_name],
                'overall_results': all_results[current_level_name]
            }
            with open(cfg.summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)    # Pretty-print JSON
            print(f"PARTIAL RESULTS SAVED: {cfg.summary_file}")

        # ── Optional wandb logging ─────────────────────────────────────────────
        if cfg.use_wandb:
            wandb.log({
                f"success_rate/{current_level_name}": final_success_rate,
                f"num_episodes/{current_level_name}": total_episodes,
            })
            wandb.save(local_log_filepath)    # Upload log file as artifact

        # Close log file before moving to next level
        if log_file:
            log_file.close()

    # ── Final results display ──────────────────────────────────────────────────
    print_results_table(task_results, command_levels, all_results)

    # Return scalar success rate for downstream use (e.g., hyperparameter search)
    if len(command_levels) > 1:
        # Average success rate across all command levels
        return sum(r['success_rate'] for r in all_results.values()) / len(all_results)
    else:
        return final_success_rate


# ──────────────────────────────────────────────────────────────────────────────
# Script entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # draccus.wrap() intercepts sys.argv and injects parsed values into cfg.
    # This block is skipped when the module is imported (e.g., in tests).
    run_libero_eval()