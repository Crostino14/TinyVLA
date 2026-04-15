"""
run_libero_eval_task_comp.py
============================

Purpose
-------
Evaluates a TinyVLA pre-trained policy on custom LIBERO task composition
scenarios (task_comp_l1 and task_comp_l2 suites).

Scientific Objective
--------------------
Tests *task-level generalization*: the model must apply known motor primitives
(pick-place, open drawer, push) to new object-target combinations NEVER seen
during training. This kind of evaluation is fundamental in robotic manipulation
to distinguish genuine compositional capability from overfitting to the
object-target pairs present in the training set.

Custom Tasks (all share the ``libero_goal`` scene):
  L1 - Object-target generalization:
    1. Put the plate on the top of the cabinet
    2. Put the plate on the stove
    3. Put the cream cheese on the top of the cabinet
    4. Put the cream cheese on the plate
    5. Open the top layer of the drawer and put the cream cheese inside

  L2 - Structural generalization (longer sequences):
    1. Open the middle drawer of the cabinet
    2. Put the bowl on the stove
    3. Put the cream cheese in the bowl
    4. Push the plate to the front of the stove
    5. Put the bowl on top of the cabinet

General Architecture
--------------------
The file follows a pipeline pattern:
  1. Configuration  (draccus + GenerateConfig dataclass)
  2. Policy loading (LLaVA-Pythia + LoRA)
  3. Custom task loading (BDDL + init_states)
  4. Evaluation loop per task (N trials x M tasks)
  5. Result collection + text/video/wandb logging

Key Dependencies
----------------
- TinyVLA / LLaVA-Pythia  : multimodal VLA backbone (vision-language-action)
- LIBERO                  : MuJoCo/Robosuite-based robotic simulator
- PyTorch / torchvision   : GPU inference
- draccus                 : CLI configuration management (hydra alternative)
- einops                  : readable tensor operations
- imageio                 : rollout video saving (.mp4)

Usage
-----
Run from CLI with draccus::

    python run_libero_eval_task_comp.py \\
        --model_path /path/to/checkpoint/last \\
        --model_base /path/to/base/model \\
        --comp_level l1 \\
        --num_trials_per_task 50

Implementation Notes
--------------------
- ``TOKENIZERS_PARALLELISM=false`` prevents deadlocks in HuggingFace tokenizer
  workers when used with multiprocessing DataLoaders.
- ``DEVICE=cuda`` forces execution on NVIDIA GPU.
- ``WANDB_DISABLED=true`` disables W&B unless explicitly activated via
  ``use_wandb=True`` in GenerateConfig.
- Gripper normalization follows the [-1, +1] convention (open/closed).
- Rotation actions are produced as rot-6D vectors and converted to XYZ Euler
  angles before being sent to the simulator controller.
"""
import sys
import os
import logging
import gc  # Python garbage collector, used to free GPU/CPU memory after each task

# Disable HuggingFace tokenizer parallelism warnings (avoids deadlocks in forked processes)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set the compute device to CUDA (GPU)
os.environ['DEVICE'] = "cuda"
# Disable Weights & Biases logging at the OS environment level (can be overridden in cfg)
os.environ["WANDB_DISABLED"] = "true"

# LLaVA-Pythia model components
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig
from llava_pythia.conversation import conv_templates, SeparatorStyle
from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import (
    tokenizer_image_token,           # Tokenizes a prompt, inserting image token placeholders
    get_model_name_from_path,        # Extracts the model name from a directory path string
    KeywordsStoppingCriteria,        # Stopping criterion for generation based on keyword matching
)
from llava_pythia.constants import (
    IMAGE_TOKEN_INDEX,               # Integer index representing the image placeholder in token sequences
    DEFAULT_IMAGE_TOKEN,             # String token representing a single image: "<image>"
    DEFAULT_IM_START_TOKEN,          # Optional image start token: "<im_start>"
    DEFAULT_IM_END_TOKEN,            # Optional image end token: "<im_end>"
)

import torch
from torchvision import transforms   # torchvision image transforms (e.g., ToTensor, Resize)
import cv2                           # OpenCV for image resizing and color manipulation
from copy import deepcopy            # Deep copy for avoiding in-place mutation of mutable objects
import numpy as np                   # NumPy for array math and robot state computation
import time                          # Standard time utilities (used implicitly in logging)
from llava_pythia.model import *     # Import all model components from llava_pythia
from einops import rearrange         # Tensor reshape utility with named dimension syntax
import models.TinyVLA.test.utils.torch_utils as TorchUtils  # Custom torch utility (e.g., rot_6d_to_euler_angles)
import pickle                        # Serialization for loading dataset stats (.pkl files)

import draccus                       # Dataclass-based CLI config parser (similar to simple-parsing)
from dataclasses import dataclass    # Python standard dataclass decorator
from typing import Optional, Union   # Type hints
from enum import Enum                # Enumeration base class for TaskSuite definition
import tqdm                          # Progress bar for loops
import json                          # JSON serialization (available but not used directly here)
from collections import deque        # Double-ended queue (available for potential temporal buffering)

# LIBERO environment and task management
from libero.libero import get_libero_path           # Returns absolute paths to LIBERO data directories
from libero.libero.benchmark import Task            # NamedTuple-like container for task metadata
from libero.libero.envs import OffScreenRenderEnv   # Headless MuJoCo-based simulation environment

# Custom utility functions for the TinyVLA LIBERO evaluation pipeline
from models.TinyVLA.test.utils.libero_utils import (
    get_libero_dummy_action,         # Returns a zero/no-op action vector for environment stabilization
    get_libero_image,                # Extracts the main agentview camera image from observations
    get_libero_wrist_image,          # Extracts the wrist camera image from observations
    quat2axisangle,                  # Converts quaternion rotation to axis-angle representation
    extract_command_from_bddl,       # Parses a BDDL file to extract the natural language task description
)
from models.TinyVLA.test.utils.robot_utils import (
    DATE_TIME,                       # Timestamp string (format: YYYY-MM-DD_HH-MM-SS) for unique file naming
    set_seed_everywhere,             # Sets random seeds in Python, NumPy, and PyTorch for reproducibility
)


# ============================================================================
# Task Composition L1 - Custom Task Definitions
# ============================================================================

# Each entry defines:
#   "bddl_file"        : The BDDL specification file for the custom compositional task.
#                        BDDL (Behavior Description and Definition Language) encodes the
#                        task goal, objects, initial conditions, and success criteria.
#   "init_states_from" : The name of the TRAINING task whose .pruned_init file is reused
#                        as the initial state distribution. This ensures the scene starts
#                        in a familiar configuration even though the goal is novel.
TASK_COMP_L1_TASKS = [
    {
        # Training: push plate (push_plate_to_stove) + put on cabinet (bowl/wine_bottle→cabinet)
        # Composition: pick-place plate → cabinet (new object-target pair)
        "bddl_file": "put_the_plate_on_top_of_the_cabinet_task_comp_l1.bddl",
        "init_states_from": "push_the_plate_to_the_front_of_the_stove",
    },
    {
        # Training: push plate (push_plate_to_stove) + put on stove (bowl→stove)
        # Composition: pick-place plate → stove (new object-target pair)
        "bddl_file": "put_the_plate_on_the_stove_task_comp_l1.bddl",
        "init_states_from": "push_the_plate_to_the_front_of_the_stove",
    },
    {
        # Training: cream_cheese→bowl + put on cabinet (bowl/wine_bottle→cabinet)
        # Composition: pick-place cream_cheese → cabinet (new object-target pair)
        "bddl_file": "put_the_cream_cheese_on_top_of_the_cabinet_task_comp_l1.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
    {
        # Training: cream_cheese→bowl + bowl→plate
        # Composition: pick-place cream_cheese → plate (new object-target pair)
        "bddl_file": "put_the_cream_cheese_on_the_plate_task_comp_l1.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
    {
        # Training: open drawer + bowl inside
        # Composition: open drawer + cream_cheese inside (swaps object, same primitive)
        "bddl_file": "open_the_top_drawer_and_put_the_cream_cheese_inside_task_comp_l1.bddl",
        "init_states_from": "open_the_top_drawer_and_put_the_bowl_inside",
    },
]

# ============================================================================
# Task Composition L2 - Custom Task Definitions
# ============================================================================

# L2 tasks are harder than L1: they introduce longer-horizon chains
# (e.g., open → pick-place → manipulation) or more distant generalization gaps.
TASK_COMP_L2_TASKS = [
    {
        # L2: open MIDDLE drawer + put bowl inside (chain: open + pick-place)
        "bddl_file": "open_the_middle_drawer_of_the_cabinet_task_comp_l2.bddl",
        "init_states_from": "open_the_middle_drawer_of_the_cabinet",
    },
    {
        # L2: put bowl on stove + turn on stove (chain: pick-place + manipulation)
        "bddl_file": "put_the_bowl_on_the_stove_task_comp_l2.bddl",
        "init_states_from": "put_the_bowl_on_the_stove",
    },
    {
        # L2: put cream cheese in bowl + put bowl on plate (chain: 2 pick-place)
        "bddl_file": "put_the_cream_cheese_in_the_bowl_task_comp_l2.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
    {
        # L2: push plate to stove front + put bowl on plate (chain: push + pick-place)
        "bddl_file": "push_the_plate_to_the_front_of_the_stove_task_comp_l2.bddl",
        "init_states_from": "push_the_plate_to_the_front_of_the_stove",
    },
    {
        # L2: put cream cheese in bowl + put bowl on top of cabinet (chain: 2 pick-place)
        "bddl_file": "put_the_bowl_on_top_of_the_cabinet_task_comp_l2.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
]

# Unified registry mapping composition level identifiers to their task lists.
# Access via: TASK_COMP_REGISTRY["l1"] or TASK_COMP_REGISTRY["l2"]
TASK_COMP_REGISTRY = {
    "l1": TASK_COMP_L1_TASKS,
    "l2": TASK_COMP_L2_TASKS,
}

# Maximum number of simulation steps per episode (consistent with libero_goal benchmark)
TASK_MAX_STEPS = 500 


# Set up the root logger with timestamp, level, and message format.
# Outputs to standard output (console).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def log_message(message: str, log_file=None):
    """
    Log a message to both the console and an optional file.

    This function serves as a unified logging interface throughout the evaluation,
    ensuring that all runtime messages (task progress, success rates, errors) are
    simultaneously written to the terminal (via the Python logger) and to a
    persistent text file for post-hoc analysis.

    Parameters
    ----------
    message : str
        The string message to log.
    log_file : file-like object or None, optional
        An open file object (writable) to which the message is also written.
        If None, the message is only sent to the logger. After writing, the
        file is flushed immediately to prevent buffered data loss on crashes.

    Returns
    -------
    None

    Example
    -------
    >>> log_message("Episode 1 started", log_file=open("run.txt", "w"))
    2025-06-01 12:00:00 [INFO] Episode 1 started
    """
    logger.info(message)       # Emit message at INFO level to stdout via the configured logger
    if log_file:
        log_file.write(message + "\n")   # Append message with newline to the log file
        log_file.flush()                 # Force flush to disk to avoid data loss on crash


def convert_actions(pred_action):
    """
    Convert a raw predicted action from the model's 10-DOF output format to a
    7-DOF robot command format.

    The policy model outputs actions in the following layout:
      - Indices 0:3   → XYZ end-effector displacement (translation)
      - Indices 3:9   → Rotation in 6D continuous representation (Zhou et al., 2019)
      - Index   9 (last) → Gripper open/close command (scalar)

    The 6D rotation representation is converted to Euler angles (XYZ convention)
    using TorchUtils, resulting in the standard 7-DOF robot action:
      [x, y, z, roll, pitch, yaw, gripper]

    Parameters
    ----------
    pred_action : np.ndarray
        Raw action array of shape (10,) output by the policy network.

    Returns
    -------
    np.ndarray
        Converted action array of shape (7,) with layout:
        [x, y, z, euler_x, euler_y, euler_z, gripper].

    Notes
    -----
    The 6D rotation representation avoids the discontinuities of Euler angles
    or quaternions in neural network outputs. See:
    "On the Continuity of Rotation Representations in Neural Networks",
    Zhou et al., CVPR 2019.
    """
    cur_xyz = pred_action[:3]        # Extract XYZ end-effector translation (3D vector)
    cur_rot6d = pred_action[3:9]     # Extract 6D rotation representation (6D vector)
    cur_gripper = np.expand_dims(pred_action[-1], axis=0)  # Expand gripper scalar to shape (1,)

    # Convert numpy 6D rotation to a PyTorch tensor with batch dimension: shape (1, 6)
    cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)

    # Convert 6D rotation to Euler angles (XYZ convention); output shape: (3,)
    cur_euler = TorchUtils.rot_6d_to_euler_angles(
        rot_6d=cur_rot6d, convention="XYZ"
    ).squeeze().numpy()

    # Concatenate translation + euler angles + gripper → shape (7,)
    return np.concatenate((cur_xyz, cur_euler, cur_gripper))


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize the gripper dimension of a robot action from [0, 1] to [-1, 1],
    with optional binarization to {-1, +1}.

    The LIBERO simulation accepts gripper actions in the range [-1, 1], where:
      -1 → fully closed gripper
      +1 → fully open gripper

    Raw model outputs are typically in [0, 1]. This function applies a linear
    mapping: normalized = 2 * (x - min) / (max - min) - 1, with min=0, max=1.

    Parameters
    ----------
    action : np.ndarray
        Action array of arbitrary leading shape (..., action_dim). The last
        dimension's final element is treated as the gripper command.
    binarize : bool, optional (default=True)
        If True, snaps the normalized gripper value to its sign: +1 or -1.
        This mimics a binary open/close gripper without fractional positions.

    Returns
    -------
    np.ndarray
        A copy of the input action with the gripper dimension normalized (and
        optionally binarized). The original array is not modified in-place.

    Notes
    -----
    The formula used is: normalized = 2 * value / 1.0 - 1 = 2 * value - 1,
    since min=0.0 and max=1.0 are hardcoded.
    """
    normalized_action = action.copy()  # Avoid mutating the original action array

    # Linear normalization from [0.0, 1.0] → [-1.0, 1.0] applied only to gripper dimension
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - 0.0) / (1.0 - 0.0) - 1

    if binarize:
        # Snap to binary values: positive → +1, negative → -1, zero → 0
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Invert the sign of the gripper action dimension.

    Some policy training datasets use the opposite gripper convention relative
    to the LIBERO simulator. This function corrects for that mismatch by
    negating only the last element of the action vector, effectively swapping
    "open" and "close" commands.

    Parameters
    ----------
    action : np.ndarray
        Action array of arbitrary leading shape (..., action_dim).

    Returns
    -------
    np.ndarray
        A copy of the input with the gripper dimension (last element) negated.
        Does not modify the input in-place.

    Example
    -------
    >>> invert_gripper_action(np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    """
    inverted_action = action.copy()   # Work on a copy to preserve the original
    inverted_action[..., -1] *= -1.0  # Negate only the gripper (last) dimension
    return inverted_action


def get_obs(obs, stats):
    """
    Extract and preprocess robot observations from a LIBERO environment step.

    This function processes the raw observation dictionary returned by the
    LIBERO environment into two components:
      1. **images**: A (2, H, W, C) NumPy array containing the resized
         agentview (front camera) and wrist camera images. Images are
         vertically and horizontally flipped to correct for camera orientation.
      2. **states**: A normalized 1D NumPy array encoding the robot's
         proprioceptive state (end-effector position + orientation + gripper),
         standardized using pre-computed dataset statistics.

    Parameters
    ----------
    obs : dict
        Raw observation dictionary from `env.step()` or `env.reset()`.
        Expected keys:
          - "agentview_image"         : (H, W, 3) uint8 front-view RGB image
          - "robot0_eye_in_hand_image": (H, W, 3) uint8 wrist-view RGB image
          - "robot0_eef_pos"          : (3,) float32 end-effector XYZ position
          - "robot0_eef_quat"         : (4,) float32 end-effector quaternion
          - "robot0_gripper_qpos"     : (2,) float32 gripper joint positions
    stats : dict
        Dictionary of dataset normalization statistics with keys:
          - "qpos_mean" : (state_dim,) mean of proprioceptive states in training data
          - "qpos_std"  : (state_dim,) std deviation of proprioceptive states

    Returns
    -------
    images : np.ndarray
        Array of shape (2, 180, 320, 3) containing the preprocessed agentview
        and wrist camera images (float32).
    states : np.ndarray
        Normalized 1D array of shape (state_dim,) = (8,) encoding:
        [eef_x, eef_y, eef_z, axis_x, axis_y, axis_z, gripper_l, gripper_r]

    Notes
    -----
    - Images are flipped with `[::-1, ::-1]` to correct upside-down and
      mirrored camera output from MuJoCo's rendering engine.
    - The proprioceptive state is normalized as: (x - mean) / std, which
      ensures the model receives inputs in the same distribution as training.
    - Quaternions are converted to axis-angle via `quat2axisangle` for
      compatibility with the policy's state encoder.
    """
    # Stack both camera images into shape (2, 180, 320, 3):
    #   [::-1, ::-1] flips both axes to correct MuJoCo's inverted camera output
    images = np.array([
        cv2.resize(obs['agentview_image'][::-1, ::-1], (320, 180)),          # Front camera, resized to 320×180
        cv2.resize(obs['robot0_eye_in_hand_image'][::-1, ::-1], (320, 180))  # Wrist camera, resized to 320×180
    ])

    # Build proprioceptive state vector by concatenating:
    #   - end-effector XYZ position        : shape (3,)
    #   - axis-angle orientation (from quat): shape (3,)
    #   - gripper joint positions           : shape (2,)
    # Total state_dim = 8
    states = np.concatenate((
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),  # Convert quaternion → axis-angle for compactness
        obs["robot0_gripper_qpos"]
    ))

    # Z-score normalization using training dataset statistics: (x - μ) / σ
    states = (states - stats["qpos_mean"]) / stats["qpos_std"]

    return images, states

# ============================================================================
# Policy Class (from TinyVLA)
# ============================================================================


class llava_pythia_act_policy:
    """
    Wrapper class for the LLaVA-Pythia Vision-Language-Action policy.

    This class encapsulates the entire inference pipeline for the TinyVLA model,
    handling:
      - Loading the pretrained model weights (with optional LoRA adapters)
      - Preprocessing multimodal inputs (dual-camera images + language + robot state)
      - Running forward passes to generate robot action chunks

    The underlying model is LLaVA-Pythia, a Vision-Language Model (VLM) built on
    the Pythia language backbone and extended with:
      - A visual encoder (CLIP-style) for processing camera images
      - An action head (ACT or diffusion-based) for robot control output
      - Support for dual image inputs (agentview + wrist camera)

    Attributes
    ----------
    policy_config : dict
        Configuration dictionary containing model paths and inference settings.
    data_args : any
        Optional data arguments forwarded from higher-level config (unused here).
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for converting language prompts to token IDs.
    policy : LlavaPythiaForCausalLM
        The main VLA model, loaded onto GPU.
    image_processor : transformers.CLIPImageProcessor (or equivalent)
        Handles image normalization and tensor conversion.
    context_len : int
        Maximum context length supported by the model.
    config : LlavaPythiaConfig
        Model configuration including action_dim, chunk_size, mm_use_im_start_end, etc.
    conv : Conversation
        Conversation template instance used to format multi-turn prompts.
    """

    def __init__(self, policy_config, data_args=None):
        """
        Initialize the policy wrapper.

        Parameters
        ----------
        policy_config : dict
            Dictionary with required keys:
              - "model_path"   (str): Path to the checkpoint directory (LoRA-merged or full)
              - "model_base"   (str): Path to the base model (used when LoRA is enabled)
              - "enable_lora"  (bool): Whether to load the model with LoRA adapter weights
              - "conv_mode"    (str): Conversation template key (e.g., "pythia")
              - "action_head"  (str): Action prediction head type ("act" or "droid_diffusion")
        data_args : any, optional
            Supplementary data arguments (currently unused, reserved for future use).
        """
        super(llava_pythia_act_policy).__init__() # Call parent __init__ (no-op for object)
        self.load_policy(policy_config) # Load the model and tokenizer based on the provided configuration
        self.data_args = data_args # Store data args for potential future use

    def load_policy(self, policy_config):
        """
        Load the pretrained LLaVA-Pythia model and associated components.

        Handles both full-model checkpoints and LoRA-augmented checkpoints.
        When LoRA is enabled, `model_base` provides the frozen backbone and
        `model_path` provides the adapter weights.

        Sets the following instance attributes upon success:
          - self.tokenizer       : Tokenizer for language prompt encoding
          - self.policy          : The loaded VLA model (on GPU)
          - self.image_processor : Preprocessor for image tensors
          - self.context_len     : Maximum supported sequence length
          - self.config          : LlavaPythiaConfig loaded from parent directory

        Parameters
        ----------
        policy_config : dict
            See `__init__` for expected keys.

        Notes
        -----
        The config is loaded from the parent directory of `model_path` (one level up),
        since LoRA checkpoints store only adapter deltas while the base config resides
        in the parent directory.
        """
        self.policy_config = policy_config
        
        # When LoRA is enabled, the base model must be provided separately;
        # otherwise, the full model is loaded directly from model_path.
        model_base = policy_config["model_base"] if policy_config['enable_lora'] else None
        
        # Extract a human-readable model name from the directory path string
        model_name = get_model_name_from_path(policy_config['model_path'])
        model_path = policy_config["model_path"]

        # Load the model, tokenizer, and image processor from the checkpoint directory.
        # The final two False flags disable device_map and load_in_8bit respectively.
        self.tokenizer, self.policy, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name, False, False
        )
        
        # Load the model configuration from the parent directory (one level up from checkpoint).
        # trust_remote_code=True allows loading custom model classes defined outside HuggingFace.
        self.config = LlavaPythiaConfig.from_pretrained(
            '/'.join(model_path.split('/')[:-1]), trust_remote_code=True
        )

    def process_batch_to_llava(self, curr_image, robo_state, raw_lang):
        """
        Prepare a forward-pass input batch from raw observations and a language command.

        This method assembles the full multimodal input dictionary required by the
        LLaVA-Pythia policy. It handles:
          1. Splitting dual-image tensors into agentview and wrist components
          2. Padding each image to a square canvas (via expand2square)
          3. Running image preprocessing (normalization to model's expected range)
          4. Constructing the language prompt using the conversation template
          5. Tokenizing the prompt with image token placeholders
          6. Packaging everything into a dict for `policy.forward()`

        Parameters
        ----------
        curr_image : torch.Tensor
            Image tensor of shape (2, C, H, W) or (1, 2, C, H, W) containing
            stacked agentview and wrist camera frames.
        robo_state : torch.Tensor
            1D float tensor of shape (state_dim,) containing normalized
            proprioceptive state (eef_pos + orientation + gripper).
        raw_lang : str
            Raw natural language task description string, e.g.,
            "put the plate on top of the cabinet".

        Returns
        -------
        data_dict : dict
            Input batch dictionary with keys:
              - "input_ids"      : (1, seq_len) token IDs tensor on CUDA
              - "attention_mask" : (1, seq_len) binary mask (0 for padding tokens)
              - "images"         : Preprocessed agentview image tensor on GPU
              - "images_r"       : Preprocessed wrist image tensor on GPU
              - "states"         : (1, state_dim) robot state tensor on GPU

        Notes
        -----
        - The conversation template (e.g., "pythia") formats the prompt with
          system/user/assistant role markers expected by the model.
        - The image token DEFAULT_IMAGE_TOKEN ("<image>") is inserted into the
          prompt string; `tokenizer_image_token` replaces it with IMAGE_TOKEN_INDEX
          so the model knows where to inject visual embeddings.
        - "<|endoftext|>" is appended to signal generation termination.
        """
        
        # Initialize a fresh conversation instance from the registered templates
        self.conv = conv_templates[self.policy_config['conv_mode']].copy()

        # Remove spurious batch dimension if input has 5 dims: (1, 2, C, H, W) → (2, C, H, W)
        if len(curr_image.shape) == 5:
            curr_image = curr_image.squeeze(0)

        # Split the stacked dual-image tensor into agentview and wrist components:
        #   image   → agentview (front camera), shape (1, C, H, W)
        #   image_r → wrist camera,             shape (1, C, H, W)
        image, image_r = torch.chunk(curr_image, 2, dim=0)

        # --- Agentview image preprocessing ---
        # Pad to square canvas using the mean color as background fill
        image = self.expand2square(image, tuple(x for x in self.image_processor.image_mean))
        # Normalize and convert to the model's expected tensor format (no rescale/crop since already done)
        image_tensor = self.image_processor.preprocess(
            image, return_tensors='pt', do_normalize=True, do_rescale=False, do_center_crop=False
        )['pixel_values']
        # Move to the policy's device and cast to the model's dtype (e.g., float16)
        image_tensor = image_tensor.to(self.policy.device, dtype=self.policy.dtype)
        
        # --- Wrist camera image preprocessing (same pipeline as agentview) ---
        image_r = self.expand2square(image_r, tuple(x for x in self.image_processor.image_mean))
        image_tensor_r = self.image_processor.preprocess(
            image_r, return_tensors='pt', do_normalize=True, do_rescale=False, do_center_crop=False
        )['pixel_values']
        
        image_tensor_r = image_tensor_r.to(self.policy.device, dtype=self.policy.dtype)

        # Start constructing the language prompt
        inp = raw_lang
        assert image is not None, 'image must be provided.'  # Sanity check: image cannot be None

        # Prepend the image token to the language input.
        # If the model uses explicit start/end tokens (mm_use_im_start_end=True),
        # wrap the image token with <im_start> and <im_end> markers.
        if self.policy.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp  # Simpler format: <image>\n<lang instruction>

        # Add user message (human turn) to the conversation
        self.conv.append_message(self.conv.roles[0], inp)
        image = None  # Free reference; image data is already in image_tensor

        # Add empty assistant turn to prompt the model to generate a response
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()        # Render the full conversation as a string
        prompt += " <|endoftext|>"             # Append EOS token to signal end of context

        # Tokenize the prompt; IMAGE_TOKEN_INDEX replaces "<image>" in the token ID sequence
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()                  # Add batch dimension and move to GPU

        # Build attention mask: 1 for real tokens, 0 for padding tokens
        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Move robot state to model's device and dtype, then add batch dimension
        states = robo_state.to(self.policy.device, dtype=self.policy.dtype)

        # Pack all inputs into a dict for the model's forward() method
        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attn_mask,
            images=image_tensor,
            images_r=image_tensor_r,
            states=states.unsqueeze(0)  # Shape: (1, state_dim)
        )
        return data_dict

    def expand2square(self, pil_imgs, background_color):
        """
        Pad a batch of images to a square canvas by centering along the shorter axis.

        Many vision encoders (e.g., CLIP) expect square input images. This function
        creates a square canvas of size max(H, W) × max(H, W), filled with a uniform
        background color, and places the original image in the center along the
        shorter axis (letterboxing for portrait images, pillarboxing for landscape).

        Parameters
        ----------
        pil_imgs : torch.Tensor
            Batch of image tensors with shape (B, C, H, W). Values should be in
            the range [0, 1] (float32) as they are placed into a NumPy float32 array.
        background_color : tuple of float
            RGB background fill values (one per channel), typically the image
            processor's `image_mean` (e.g., (0.485, 0.456, 0.406) for CLIP).

        Returns
        -------
        torch.Tensor
            Square image batch of shape (B, max_dim, max_dim, C), converted back
            to a torch.Tensor with the same dtype and device as the input.

        Notes
        -----
        - The output tensor has channels in the LAST dimension (B, H, W, C),
          unlike the input which uses (B, C, H, W). The image_processor.preprocess
          method downstream handles this layout correctly.
        - For square inputs (H == W), no padding is applied; the image is simply
          permuted to (B, H, W, C) format.
        """
        batch_size, channels, height, width = pil_imgs.shape

        # Determine the square canvas size as the larger of the two spatial dims
        max_dim = max(height, width)

        # Allocate a background-filled float32 canvas: (B, max_dim, max_dim, C)
        expanded_imgs = np.full(
            (batch_size, max_dim, max_dim, channels),
            background_color, dtype=np.float32
        )

        if height == width:
            # Already square: simply reorder dimensions from (B, C, H, W) → (B, H, W, C)
            expanded_imgs = pil_imgs.permute(0, 2, 3, 1).cpu().numpy()
        elif height > width:
            # Portrait (tall) image: pad horizontally (pillarbox)
            offset = (max_dim - width) // 2          # Center the image horizontally
            expanded_imgs[:, :height, offset:offset + width, :] = (
                pil_imgs.permute(0, 2, 3, 1).cpu().numpy()
            )
        else:
            # Landscape (wide) image: pad vertically (letterbox)
            offset = (max_dim - height) // 2         # Center the image vertically
            expanded_imgs[:, offset:offset + height, :width, :] = (
                pil_imgs.permute(0, 2, 3, 1).cpu().numpy()
            )

        # Convert back to a torch.Tensor on the same device and with the same dtype as input
        expanded_imgs = torch.tensor(expanded_imgs).to(
            dtype=pil_imgs.dtype, device=pil_imgs.device
        )
        return expanded_imgs


# ============================================================================
# Configuration
# ============================================================================

class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


@dataclass
class GenerateConfig:
    # fmt: off

    # Model
    model_path: str = ""
    model_base: str = ""
    model_family: str = "tiny_vla"

    # LIBERO environment
    task_suite_name: str = TaskSuite.LIBERO_GOAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    env_img_res: int = 256

    # Utils
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    checkpoint_size: int = 20000

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"

    seed: int = 7
    run_number: int = 0
    debug: bool = False
    local_rank: int = 0
    comp_level: str = "l2"

    # Task subset (for splitting across nodes)
    task_start: int = 0
    task_end: int = -1  # -1 means all tasks
    # fmt: on


# ============================================================================
# Custom Task Loading
# ============================================================================

def load_custom_tasks(comp_level: str = "l2"):
    """
    Load and validate all custom task composition tasks for a given difficulty level.

    For each task defined in TASK_COMP_REGISTRY[comp_level], this function:
      1. Resolves the absolute path to the BDDL file (in the libero_goal scene folder)
      2. Parses the BDDL file to extract the natural language task description
      3. Constructs a LIBERO `Task` NamedTuple with all required metadata
      4. Loads the associated initial state distribution (.pruned_init file) from
         the corresponding training task's precomputed states

    This design allows compositional tasks to reuse training-distribution initial
    scenes (same object placement) while having a different success criterion (goal).

    Parameters
    ----------
    comp_level : str, optional
        Composition difficulty level: "l1" (object-target generalization) or
        "l2" (multi-step chain generalization). Default: "l2".

    Returns
    -------
    list of dict
        A list of task info dictionaries, one per task, each containing:
          - "task"             : libero.libero.benchmark.Task NamedTuple
          - "init_states"      : list of initial state vectors (loaded via torch.load)
          - "task_description" : str, natural language command parsed from BDDL
          - "bddl_path"        : str, absolute path to the task's BDDL file

    Raises
    ------
    AssertionError
        If the BDDL file does not exist at the expected path.
    AssertionError
        If the natural language description cannot be extracted from the BDDL file.
    AssertionError
        If the .pruned_init initial states file does not exist.

    Notes
    -----
    - `get_libero_path("bddl_files")` returns the LIBERO package's canonical
      directory for BDDL task definition files.
    - `.pruned_init` files are pre-generated initial state archives created by
      filtering LIBERO's raw init states to remove invalid configurations.
    - `torch.load(..., weights_only=False)` is required here because the init
      states may contain Python objects beyond plain tensors (e.g., dicts, lists).
    """
    # Resolve paths to the libero_goal scene's BDDL and init_states directories
    bddl_dir = os.path.join(get_libero_path("bddl_files"), "libero_goal")
    init_dir = os.path.join(get_libero_path("init_states"), "libero_goal")

    custom_tasks = []
    for task_def in TASK_COMP_REGISTRY[comp_level]:
        bddl_filename = task_def["bddl_file"]    # BDDL filename for this compositional task
        init_from = task_def["init_states_from"] # Training task name whose init states are reused

        bddl_path = os.path.join(bddl_dir, bddl_filename)
        # Verify the BDDL file exists before proceeding
        assert os.path.exists(bddl_path), f"BDDL file not found: {bddl_path}"

        # Parse the task's natural language description from the BDDL file header
        task_description = extract_command_from_bddl(bddl_path)
        assert task_description is not None, f"Could not extract language from {bddl_path}"

        # Construct the task name as the BDDL filename without extension
        task_name = bddl_filename.replace(".bddl", "")

        # Build the LIBERO Task NamedTuple:
        #   - problem_folder determines which scene directory is used
        #   - init_states_file points to the training task's pruned init states
        task = Task(
            name=task_name,
            language=task_description,
            problem="Libero",
            problem_folder="libero_goal",
            bddl_file=bddl_filename,
            init_states_file=f"{init_from}.pruned_init",  # Reuse training task initial states
        )

        # Build the absolute path to the init states file and verify it exists
        init_states_path = os.path.join(init_dir, f"{init_from}.pruned_init")
        assert os.path.exists(init_states_path), f"Init states not found: {init_states_path}"

        # Load the initial state list from disk; weights_only=False allows arbitrary objects
        init_states = torch.load(init_states_path, weights_only=False)

        # Append the complete task info dict to the result list
        custom_tasks.append({
            "task": task,
            "init_states": init_states,
            "task_description": task_description,
            "bddl_path": bddl_path,
        })

    return custom_tasks


def create_env_from_bddl(bddl_path, resolution=256):
    """
    Instantiate a LIBERO off-screen simulation environment from a BDDL file.

    This function creates a headless (no display required) MuJoCo-based
    environment that renders camera observations at the specified resolution.
    The environment is seeded with 0 for deterministic object placement.

    Parameters
    ----------
    bddl_path : str
        Absolute path to the BDDL file defining the task, scene objects,
        initial conditions, and success criteria.
    resolution : int, optional
        Height and width in pixels for both the agentview and wrist cameras.
        Default: 256.

    Returns
    -------
    env : OffScreenRenderEnv
        Initialized LIBERO simulation environment ready for `env.reset()`.

    Notes
    -----
    - `env.seed(0)` is called after construction because the MuJoCo seed
      affects physical simulation randomness including object initial placement,
      even when using fixed initial states. Consistent seeding ensures comparable
      results across evaluation runs.
    - The LIBERO `OffScreenRenderEnv` wraps a MuJoCo simulation with
      robosuite-style APIs: `env.reset()`, `env.step(action)`, `env.close()`.
    """
    env_args = {
        "bddl_file_name": bddl_path,      # Path to the task BDDL specification
        "camera_heights": resolution,      # Vertical resolution for all cameras
        "camera_widths": resolution,       # Horizontal resolution for all cameras
    }
    env = OffScreenRenderEnv(**env_args)   # Instantiate the headless simulation
    env.seed(0)                            # Seed for deterministic physics behavior
    return env


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(cfg: GenerateConfig):
    """
    Initialize evaluation logging: file, WandB, and run ID construction.

    Creates a unique run identifier from the configuration parameters and
    opens a text log file in `cfg.local_log_dir`. Optionally initializes
    a Weights & Biases run for remote experiment tracking.

    The run ID format is:
        EVAL-task_comp_{comp_level}-{model_family}-{DATE_TIME}
        [--{run_id_note}]          (if run_id_note is set)
        [--ckpt{checkpoint_size}]  (if checkpoint_size > 0)

    Parameters
    ----------
    cfg : GenerateConfig
        The runtime configuration dataclass populated from CLI arguments.

    Returns
    -------
    log_file : file object
        Writable file object for the text log. Caller is responsible for closing.
    local_log_filepath : str
        Absolute path to the created log file.
    run_id : str
        The unique run identifier string (used for log file naming and W&B).

    Side Effects
    ------------
    - Creates `cfg.local_log_dir` (and parents) if it doesn't exist.
    - Opens a log file for writing (NOT appending).
    - Optionally calls `wandb.init()` to start a new W&B run.
    """
    # Build a unique, human-readable run identifier
    run_id = f"EVAL-task_comp_{cfg.comp_level}-{cfg.model_family}-{DATE_TIME}"

    # Optionally append a user-provided note for better run identification
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Append checkpoint size for traceability when comparing across checkpoints
    if cfg.checkpoint_size > 0:
        run_id += f"--ckpt{cfg.checkpoint_size}"

    # Create the log directory (and all intermediate directories) if they don't exist
    os.makedirs(cfg.local_log_dir, exist_ok=True)

    # Open the log file for writing; this will be written to throughout the evaluation
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Optionally initialize Weights & Biases for remote experiment tracking
    if cfg.use_wandb:
        import wandb
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    return log_file, local_log_filepath, run_id


# ============================================================================
# Episode & Task Execution
# ============================================================================

def run_episode(
    cfg, env, task_description, policy, policy_config, resize_size,
    initial_state=None, log_file=None,
):
    """
    Execute a single evaluation episode in the LIBERO simulation environment.

    An episode consists of:
      1. Resetting the environment to a fixed or random initial state
      2. Executing `cfg.num_steps_wait` no-op steps for physics stabilization
      3. Iteratively querying the policy for action chunks and executing them
         step-by-step until task success or episode timeout

    The policy uses temporal aggregation (if enabled): instead of executing
    only the most recently predicted action, it maintains a buffer of overlapping
    predictions and computes an exponentially weighted average across them,
    reducing jitter and improving temporal consistency.

    Parameters
    ----------
    cfg : GenerateConfig
        Evaluation configuration (num_steps_wait, model_family, etc.).
    env : OffScreenRenderEnv
        LIBERO simulation environment (already instantiated and seeded).
    task_description : str
        Natural language task instruction passed to the policy as context.
    policy : llava_pythia_act_policy
        Loaded TinyVLA policy instance.
    policy_config : dict
        Policy configuration dict (used to select action head type).
    resize_size : int
        Target image size passed to the policy (currently 224 for CLIP compatibility).
    initial_state : object or None, optional
        If provided, `env.set_init_state(initial_state)` is called to reproduce
        a specific starting configuration. Otherwise, `env.reset()` uses defaults.
    log_file : file object or None, optional
        Open log file for writing episode-level messages.

    Returns
    -------
    success : bool
        True if the environment returned `done=True` before timeout, indicating
        the task success criterion was met.
    replay_traj : dict
        Dictionary recording the episode trajectory with keys:
          - "images"        : list of (256, 256, 3) uint8 numpy arrays (agentview frames)
          - "task_command"  : str, the natural language instruction used
          - "states"        : list of normalized proprioceptive state arrays
          - "actions"       : list of (7,) numpy arrays (executed actions)

    Notes
    -----
    Temporal Aggregation (Zhao et al., ACT):
    -----------------------------------------
    At each timestep t, the policy predicts a chunk of `num_queries` future actions.
    These are stored in `all_time_actions[t, t:t+num_queries]`. When executing
    step t, we gather all predictions that cover timestep t (from previous queries),
    filter out zero-initialized entries, and compute a weighted average:

        a_t = Σ exp(-k * i) * a_{t,i} / Σ exp(-k * i)

    where i indexes predictions from oldest (i=0) to most recent, and k=0.01
    controls the recency bias. This soft ensemble reduces action inconsistencies
    between query intervals.

    Model Warm-up:
    --------------
    At t=0, the policy is invoked 10 times without using the output. This forces
    CUDA kernels to initialize and stabilizes GPU memory allocation, preventing
    the first "real" inference from being anomalously slow.
    """
    env.reset()                              # Reset the environment to clear prior state
    to_tensor = transforms.ToTensor()        # Transform: converts HWC uint8 numpy → CHW float32 tensor

    # Set the initial state if provided (for reproducible evaluation episodes)
    if initial_state is not None:
        obs = env.set_init_state(initial_state)  # Restore a specific scene configuration
    else:
        obs = env.reset()                         # Use default/random reset

    # Determine action head-specific flags
    if policy_config["action_head"] == 'act':
        rand_crop_resize = False   # ACT does not use random crop augmentation at test time
        temporal_agg = True        # ACT uses temporal aggregation for smoother execution
    else:
        rand_crop_resize = True    # Diffusion head uses random crop for consistency with training
        temporal_agg = True

    action_dim = policy.config.action_dim  # Dimensionality of the raw action output (e.g., 10)
    policy.policy.eval()                   # Set model to evaluation mode (disables dropout, BN tracking)

    # Load dataset normalization statistics from a pickle file co-located with the checkpoint
    stats_path = os.path.join(
        "/".join(policy_config['model_path'].split('/')[:-1]), 'dataset_stats.pkl'
    )
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)   # Contains: action_mean, action_std, action_min, action_max, qpos_mean, qpos_std

    # Define the post-processing function to un-normalize raw model outputs:
    #   - ACT head: actions normalized as z-scores → reverse with (a * std + mean)
    #   - Diffusion heads: actions normalized to [-1, 1] → reverse with linear map to [min, max]
    if policy_config["action_head"] == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif policy_config["action_head"] == 'transformer_diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    elif policy_config["action_head"] == 'droid_diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']

    # Query frequency: how often to call the policy (every N steps)
    query_frequency = policy.config.chunk_size / 2
    if temporal_agg:
        query_frequency = 1          # With temporal aggregation, query at every step
        num_queries = policy.config.chunk_size   # Number of future steps predicted per query

    max_timesteps = int(200)         # Maximum episode length in simulation steps

    # Pre-allocate the temporal aggregation buffer:
    # Shape: (max_timesteps, max_timesteps + num_queries, action_dim)
    # all_time_actions[t, s] stores the prediction for step s made at query time t
    if temporal_agg:
        all_time_actions = torch.zeros(
            [max_timesteps, max_timesteps + num_queries, action_dim],
            dtype=torch.float32
        ).cuda()

    # Initialize episode tracking variables
    t = 0                       # Current simulation timestep counter
    replay_traj = dict()        # Dictionary to store the recorded trajectory
    image_list = []             # List of agentview frames (for video recording)
    robot_state_list = []       # List of proprioceptive state arrays
    target_action_list = []     # List of executed action vectors
    success = False             # Episode outcome flag (set to True if task is solved)

    with torch.inference_mode():    # Disable gradient computation for all operations inside
        try:
            # --- Physics Stabilization Phase ---
            # Execute no-op actions for num_steps_wait steps to let the scene settle
            while t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1

            t = 0  # Reset step counter after stabilization phase

            # --- Main Control Loop ---
            while t < max_timesteps:
                # Extract preprocessed images and normalized robot state from current observation
                traj_rgb_np, robot_state = get_obs(obs=obs, stats=stats)

                # Record the agentview frame (resized to 256×256 for video consistency)
                image_list.append(cv2.resize(traj_rgb_np[0], (256, 256)))
                robot_state_list.append(robot_state)

                # Convert robot state to a CUDA float tensor for model input
                robot_state = torch.from_numpy(robot_state).float().cuda()

                # --- Image Preprocessing (at query frequency) ---
                if t % query_frequency == 0:
                    curr_image = []
                    for img in traj_rgb_np:
                        # Convert each HWC numpy image to CHW float32 tensor and move to GPU
                        curr_image.append(to_tensor(img).float().cuda())
                    # Stack both camera images: shape (2, C, H, W)
                    curr_image = torch.stack(curr_image, dim=0)

                    if rand_crop_resize:
                        # Center crop to 95% of original size, then resize back:
                        # Simulates slight zoom-in augmentation used during training
                        original_size = curr_image.shape[-2:]  # (H, W)
                        ratio = 0.95                            # Keep 95% of the image
                        # Compute crop boundaries symmetrically around center
                        curr_image = curr_image[:, :,
                            int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                            int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
                        curr_image = curr_image.squeeze(0)              # Remove batch dim temporarily
                        resize_transform = transforms.Resize(original_size, antialias=True)
                        curr_image = resize_transform(curr_image)       # Resize back to original resolution
                        curr_image = curr_image.unsqueeze(0)            # Restore batch dimension

                # --- Model Warm-up at t=0 ---
                # Run 10 dummy forward passes to initialize CUDA kernels and allocate GPU memory.
                # These outputs are discarded; this prevents anomalously slow first inference.
                if t == 0:
                    for _ in range(10):
                        batch = policy.process_batch_to_llava(curr_image, robot_state, task_description)
                        policy.policy(**batch, eval=True)

                # --- Policy Inference ---
                if policy_config['action_head'] in ["act", "droid_diffusion"]:
                    if t % query_frequency == 0:
                        # Build multimodal input batch and run forward pass
                        batch = policy.process_batch_to_llava(curr_image, robot_state, task_description)
                        # all_actions: shape (1, chunk_size, action_dim) — predicted action sequence
                        all_actions = policy.policy(**batch, eval=True)

                    if temporal_agg:
                        # Store this query's predictions in the temporal buffer
                        # all_time_actions[t, t:t+num_queries] ← predicted actions for future steps
                        all_time_actions[[t], t:t + num_queries] = all_actions

                        # Gather all predictions that cover the current timestep t
                        actions_for_curr_step = all_time_actions[:, t]   # Shape: (max_timesteps, action_dim)

                        # Filter out buffer slots that were never written (still all-zeros)
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]

                        # Exponential recency weighting: older predictions get lower weight
                        # k=0.01 gives a mild bias toward more recent predictions
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()    # Normalize to sum=1
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)

                        # Weighted sum across all predictions for step t → shape (1, action_dim)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        # Without temporal aggregation: use the t-th action in the current chunk
                        raw_action = all_actions[:, t % query_frequency]
                else:
                    raise NotImplementedError  # Only ACT and droid_diffusion heads are supported

                # --- Action Post-processing ---
                raw_action = raw_action.squeeze(0).cpu().numpy()  # Remove batch dim, move to CPU
                action = post_process(raw_action)                  # Un-normalize using dataset stats
                action = convert_actions(action)                   # Convert 10-DOF → 7-DOF robot command

                # --- Environment Step ---
                obs, reward, done, info = env.step(action.tolist())  # Execute action in simulation
                target_action_list.append(action)                    # Record executed action

                # Check for task success (env returns done=True when goal is achieved)
                if done:
                    success = True
                    break   # Exit the control loop immediately upon success

                t += 1  # Advance the timestep counter

        except Exception as e:
            # Catch all exceptions (e.g., MuJoCo instability, CUDA OOM) and mark as failure
            log_message(f"Episode error: {e}", log_file)
            success = False

    # Store trajectory data for later video/npy export
    replay_traj['images'] = image_list
    replay_traj['task_command'] = task_description
    replay_traj['states'] = robot_state_list
    replay_traj['actions'] = target_action_list

    return success, replay_traj


def run_custom_task(
    cfg, task_info, task_idx, num_tasks, policy, policy_config,
    log_file, total_episodes=0, total_successes=0,
):
    """
    Execute the full multi-trial evaluation for a single custom task composition task.

    For each trial (episode), this function:
      1. Retrieves the corresponding initial state from the preloaded list
      2. Calls `run_episode()` to execute the episode and obtain success/trajectory
      3. Saves the rollout as an MP4 video and a .npy trajectory file
      4. Logs per-episode and cumulative statistics
      5. Optionally logs per-task success rates to Weights & Biases

    Parameters
    ----------
    cfg : GenerateConfig
        Runtime configuration (num_trials_per_task, comp_level, run_number, etc.).
    task_info : dict
        Task info dictionary as returned by `load_custom_tasks()`, containing:
          - "task"             : LIBERO Task NamedTuple
          - "init_states"      : list of initial state objects
          - "task_description" : str, natural language task instruction
          - "bddl_path"        : str, absolute BDDL file path
    task_idx : int
        0-based index of this task within the current task subset (for logging).
    num_tasks : int
        Total number of tasks in the current evaluation run (for progress display).
    policy : llava_pythia_act_policy
        Loaded TinyVLA policy instance.
    policy_config : dict
        Policy configuration dict.
    log_file : file object
        Open text file for logging.
    total_episodes : int, optional
        Running total of episodes across all tasks so far. Default: 0.
    total_successes : int, optional
        Running total of successful episodes across all tasks so far. Default: 0.

    Returns
    -------
    total_episodes : int
        Updated total episode count (previous + episodes run in this task).
    total_successes : int
        Updated total success count (previous + successes in this task).
    task_description : str
        Natural language description of this task (used as dict key in results).
    task_success_rate : float
        Success rate for this task: task_successes / task_episodes ∈ [0.0, 1.0].
    task_episodes : int
        Number of episodes run for this task (= cfg.num_trials_per_task).

    Side Effects
    ------------
    - Creates output directory: `.../rollouts/libero_goal/task_composition/tinyvla/task_comp_{level}/run_{N}/`
    - Saves one .mp4 video and one .npy file per episode.
    - Calls `env.close()` and `gc.collect()` to free resources after all trials.
    - Logs task-level success rate to W&B if `cfg.use_wandb` is True.
    """
    # Unpack task metadata from the task_info dictionary
    task = task_info["task"]
    init_states = task_info["init_states"]
    task_description = task_info["task_description"]
    bddl_path = task_info["bddl_path"]

    # Instantiate a fresh environment from the task's BDDL file
    env = create_env_from_bddl(bddl_path, resolution=cfg.env_img_res)

    # Log task header information
    log_message("=" * 80, log_file)
    log_message(f"TASK {task_idx + 1}/{num_tasks} (Task Composition L1)", log_file)
    log_message(f"BDDL: {task.bddl_file}", log_file)
    log_message(f"Command: {task_description}", log_file)
    log_message(f"Init states from: {task.init_states_file}", log_file)
    log_message("=" * 80, log_file)

    task_episodes, task_successes = 0, 0  # Task-local counters

    # --- Trial Loop ---
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        # Select the initial state for this episode by index
        initial_state = init_states[episode_idx]

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run the episode and collect success flag + trajectory data
        success, replay_traj = run_episode(
            cfg, env, task_description, policy, policy_config,
            224,                # resize_size: CLIP-compatible 224px input
            initial_state, log_file,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # --- Save Rollout Video ---
        # Construct the output directory path using comp_level and run_number for versioning
        rollout_dir = (
            f"/mnt/beegfs/a.cardamone7/outputs/rollouts/libero_goal/"
            f"task_composition/tinyvla/task_comp_{cfg.comp_level}/run_{cfg.run_number}"
        )
        os.makedirs(rollout_dir, exist_ok=True)

        # Sanitize the task description for use as a filename component (max 50 chars)
        processed_desc = (
            task_description.lower()
            .replace(" ", "_")
            .replace("\n", "_")
            .replace(".", "_")[:50]
        )
        # Build a descriptive MP4 filename including episode count and success status
        mp4_path = (
            f"{rollout_dir}/{DATE_TIME}"
            f"--episode={total_episodes}"
            f"--success={success}"
            f"--task={processed_desc}.mp4"
        )

        # Write the recorded frames as an MP4 video at 30 FPS
        import imageio
        video_writer = imageio.get_writer(mp4_path, fps=30)
        for img in replay_traj['images']:
            video_writer.append_data(img)   # Append each frame to the video
        video_writer.close()                # Finalize and close the video file
        log_message(f"Saved rollout MP4 at {mp4_path}", log_file)

        # Also save the full trajectory dict as a .npy file (for offline analysis)
        npy_path = mp4_path.replace('.mp4', '.npy')
        np.save(npy_path, replay_traj)
        log_message(f"Saved trajectory at {npy_path}", log_file)

        # Log per-episode statistics
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes: {total_episodes}", log_file)
        log_message(
            f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)",
            log_file
        )

    # Compute task-level success rate (guard against division by zero)
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    log_message(
        f"Task success rate: {task_success_rate:.4f} ({task_success_rate * 100:.1f}%)",
        log_file
    )

    # Log per-task metrics to Weights & Biases if enabled
    if cfg.use_wandb:
        import wandb
        wandb.log({
            f"success_rate/{task_description}": task_success_rate,
            f"num_episodes/{task_description}": task_episodes,
        })

    # --- Environment Cleanup ---
    try:
        env.close()                                        # Release MuJoCo simulation resources
        log_message("Environment closed successfully", log_file)
    except Exception as e:
        log_message(f"Warning: Error closing environment: {e}", log_file)
    gc.collect()  # Force Python garbage collection to free memory before the next task

    return total_episodes, total_successes, task_description, task_success_rate, task_episodes

# ============================================================================
# Results
# ============================================================================

def print_results_table(task_results, all_results):
    """Print a summary table of task composition L1 results."""
    print("\n" + "=" * 100)
    print("TASK COMPOSITION L1 (TinyVLA) - RESULTS TABLE")
    print("=" * 100)

    print(f"{'Task':<60} | {'Success Rate':>20} | {'Episodes':>8}")
    print("-" * 100)

    for task_name, result in task_results.items():
        sr = result['success_rate']
        eps = result['episodes']
        successes = int(sr * eps)
        print(f"{task_name:<60} | {sr:>11.1%} ({successes:>2}/{eps:<2}) | {eps:>8}")

    print("-" * 100)
    overall_sr = all_results['success_rate']
    overall_succ = all_results['total_successes']
    overall_eps = all_results['total_episodes']
    print(f"{'OVERALL':<60} | {overall_sr:>11.1%} ({overall_succ:>3}/{overall_eps:<3}) | {overall_eps:>8}")
    print("=" * 100)


# ============================================================================
# Main Entry Point
# ============================================================================

@draccus.wrap()
def eval_task_comp(cfg: GenerateConfig) -> float:
    """Evaluate TinyVLA on task composition L1 scenarios."""
    if cfg.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    assert cfg.model_path, "model_path must not be empty!"
    assert cfg.model_base, "model_base must not be empty!"

    # Set seed
    set_seed_everywhere(cfg.seed)

    # Initialize policy
    action_head = 'droid_diffusion'
    policy_config = {
        "model_path": cfg.model_path,
        "model_base": cfg.model_base,
        "enable_lora": True,
        "conv_mode": "pythia",
        "action_head": action_head,
    }
    policy = llava_pythia_act_policy(policy_config)

    # Load custom tasks
    all_custom_tasks = load_custom_tasks(cfg.comp_level)
    total_num_tasks = len(all_custom_tasks)

    # Select task subset
    task_end = cfg.task_end if cfg.task_end >= 0 else total_num_tasks
    task_start = cfg.task_start
    custom_tasks = all_custom_tasks[task_start:task_end]
    num_tasks = len(custom_tasks)

    log_message(f"Loaded {total_num_tasks} total task composition {cfg.comp_level.upper()} tasks", None)
    log_message(f"Running task subset [{task_start}:{task_end}] ({num_tasks} tasks)", None)
    for i, ct in enumerate(custom_tasks):
        log_message(f"  [{task_start + i}] {ct['task_description']} ({ct['task'].bddl_file})", None)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    log_message("=" * 80, log_file)
    log_message(f"TASK COMPOSITION {cfg.comp_level.upper()} EVALUATION (TinyVLA)", log_file)
    log_message(f"Model: {cfg.model_path}", log_file)
    log_message(f"Model base: {cfg.model_base}", log_file)
    log_message(f"Seed: {cfg.seed}", log_file)
    log_message(f"Num trials per task: {cfg.num_trials_per_task}", log_file)
    log_message(f"Num tasks: {num_tasks} (subset [{task_start}:{task_end}] of {total_num_tasks})", log_file)
    log_message("=" * 80, log_file)

    # Run evaluation
    total_episodes, total_successes = 0, 0
    task_results = {}

    for task_idx in tqdm.tqdm(range(num_tasks), desc=f"Task Comp {cfg.comp_level.upper()}"):
        total_episodes, total_successes, task_name, task_sr, task_eps = run_custom_task(
            cfg, custom_tasks[task_idx], task_idx, num_tasks,
            policy, policy_config, log_file,
            total_episodes, total_successes,
        )

        task_results[task_name] = {
            'success_rate': task_sr,
            'episodes': task_eps,
        }

    # Final results
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    all_results = {
        'success_rate': final_success_rate,
        'total_episodes': total_episodes,
        'total_successes': total_successes,
    }

    log_message("=" * 80, log_file)
    log_message("FINAL RESULTS - TASK COMPOSITION L1 (TinyVLA):", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    log_message("=" * 80, log_file)

    if cfg.use_wandb:
        import wandb
        wandb.log({
            "success_rate/task_comp_l1_overall": final_success_rate,
            "num_episodes/task_comp_l1_overall": total_episodes,
        })
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    print_results_table(task_results, all_results)

    return final_success_rate


if __name__ == "__main__":
    eval_task_comp()
