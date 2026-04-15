"""
extract_embeddings_rollout.py
==============================

Overview
--------
This script extracts the internal language-conditioned representations (embeddings)
produced by the TinyVLA model during **real inference rollouts** on LIBERO robotic
manipulation tasks. Unlike static text-only embedding extraction, this approach
captures how the model encodes each task command in the full multimodal context
of actual policy execution — conditioned on both the camera image and the robot
proprioceptive state at each timestep.

The extracted vectors are the `hidden_states` of the **GPT-NeoX / Pythia backbone**
(the language model component inside LLaVA-Pythia), captured at the final layer
BEFORE they are passed to the diffusion action head. This is the richest available
representation: it encodes language, vision, and robot state jointly.

Why Capture Before the Action Head?
-------------------------------------
The diffusion action head projects the language backbone's output into the robot
action space (10D: xyz + rot6d + gripper). Capturing the embedding before this
projection gives a pure semantic representation that can be compared across
different language commands without any action-space distortion.

Technical Mechanism: PyTorch Forward Hook
------------------------------------------
Embedding extraction is implemented via a **PyTorch forward hook** attached to the
GPT-NeoX backbone module. A forward hook is a callback registered on a `nn.Module`
that is automatically called after the module's `forward()` method executes, receiving
the module itself, its inputs, and its outputs. This allows activations to be captured
non-invasively without modifying the model's code.

The hook captures `hidden_states` of shape `(B, seq_len, hidden_dim)` and:
  1. Identifies the position of the IMAGE_TOKEN_INDEX placeholder in the token sequence.
  2. Determines how many visual patch tokens replaced that placeholder during
     `prepare_inputs_labels_for_multimodal` (multimodal fusion).
  3. Excludes all visual patch tokens from the embedding, keeping only text token
     hidden states.
  4. Mean-pools the text hidden states over the sequence dimension to produce a single
     fixed-size vector of shape `(hidden_dim,)`.

This text-only mean pooling is consistent with the OpenVLA embedding extraction
approach and ensures that the embedding reflects the language understanding rather
than the visual content.

Two Extraction Modes
---------------------
The script supports two granularities of embedding extraction:

  FULL ROLLOUT mode (default):
    The policy executes the full episode (up to max_steps). At each step, one
    embedding is captured via the hook. After all steps, embeddings are mean-pooled
    over the timestep dimension to produce one representative vector per rollout.
    Across N rollouts, these per-rollout mean vectors are again mean-pooled into a
    single canonical embedding for the task-level analysis.

  FIRST STEP ONLY mode (--first_step_only):
    Only the embedding from the first inference step (t=0) is captured per rollout.
    The policy then terminates without executing any actions. This is useful for
    studying the model's initial encoding of the language command before any robot
    state feedback influences the representation.

Temporal Aggregation (Action Chunking)
----------------------------------------
TinyVLA uses action chunking with temporal aggregation (from the ACT paper).
At each timestep t, the policy predicts a chunk of `chunk_size` future actions.
Rather than executing only the first predicted action (which would cause jerky motion),
temporal aggregation maintains a buffer of all past predictions for the current step
and computes a weighted average with exponentially decaying weights:

    \[
        a_t = \frac{\sum_{i=0}^{t} e^{-k \cdot (t-i)} \hat{a}^{(i)}_t}{\sum_{i=0}^{t} e^{-k \cdot (t-i)}}
    \]

where \(\hat{a}^{(i)}_t\) is the prediction for timestep t made at inference step i,
and k=0.01 is the decay rate. This produces smoother actions by blending recent
predictions with older ones, with older predictions weighted less.

Embedding Data Schema
----------------------
Each extracted record stored in the output `.pkl` file has the following structure:

  Key   : "task_{id:02d}_{level}"  (e.g., "task_03_l2")
  Value : dict
    - "task_id"              (int)        : 0-based task index
    - "task_name"            (str)        : Task name string from LIBERO
    - "command_level"        (str)        : "default", "l1", "l2", or "l3"
    - "command_text"         (str)        : Exact natural language command used
    - "embedding"            (np.ndarray) : Shape (hidden_dim,) — mean over all rollouts
    - "embedding_per_rollout"(np.ndarray) : Shape (N_rollouts, hidden_dim)
    - "num_rollouts"         (int)        : Number of rollouts that yielded embeddings
    - "first_step_only"      (bool)       : Extraction mode flag
    - "model"                (str)        : "tinyvla"
    [Full rollout mode only]:
    - "embedding_all_steps"  (np.ndarray) : All per-step embeddings concatenated
    - "rollout_successes"    (list of bool): Per-rollout success flags
    - "num_successes"        (int)        : Number of successful rollouts
    - "total_steps"          (int)        : Total inference steps across all rollouts
    - "success_rate"         (float)      : Fraction of successful rollouts

Output
------
One pickle file per extraction run, saved to `output_dir`:
  rollout_embeddings_tinyvla_{suite}_{levels}_{mode}_r{N}.pkl

Plus one timestamped `.log` file per run for full reproducibility tracing.

Dependencies
------------
  - torch, torchvision : Model loading, tensor ops, image transforms
  - numpy, cv2         : Image preprocessing and array operations
  - pickle             : Serialization of embedding dicts
  - einops             : Tensor reshaping utilities
  - libero             : LIBERO benchmark task suite and environments
  - llava_pythia       : LLaVA-Pythia model, tokenizer, and config
  - logging            : Structured dual-destination logging
  - gc                 : Python garbage collector for VRAM management

Usage
-----
    python extract_embeddings_rollout.py \\
        --model_path /path/to/checkpoint-54000 \\
        --model_base /path/to/1.3B \\
        --task_suite libero_goal \\
        --command_levels default l1 l2 l3 \\
        --num_rollouts 10 \\
        --output_dir /mnt/beegfs/.../embeddings/tinyvla

    # First-step-only mode (faster, no full rollout):
    python extract_embeddings_rollout.py \\
        --model_path ... --model_base ... --first_step_only
"""

import os       # Filesystem path construction, environment variable setting
import sys      # System-level operations (imported for potential interpreter exit)
import torch    # PyTorch: model inference, tensor operations, GPU memory management
import pickle   # Serialization of Python objects (embedding dicts, dataset stats)
import numpy as np   # Numerical operations on embedding arrays
import cv2      # OpenCV: image resizing for preprocessing pipeline
import argparse # Command-line argument parsing
import gc       # Python garbage collector: manual VRAM cleanup between model loads
import logging  # Structured logging with configurable handlers and formatters
from datetime import datetime   # Wall-clock timestamps for log filenames and reports
from collections import deque   # Double-ended queue (imported for potential action buffer use)
from dataclasses import dataclass  # Decorator for structured data classes (available for config)
from typing import Optional, List, Dict, Any  # PEP 484 type hint annotations


# ============================================================================
# Logging Setup
# ============================================================================


def setup_logging(output_dir: str, task_suite: str):
    """
    Initialize and configure a dual-destination logger for the extraction run.

    Creates a named logger that simultaneously writes to:
      1. A timestamped `.log` file (DEBUG level — full verbosity, for post-hoc analysis)
      2. The console / stderr (INFO level — key milestones only, for live monitoring)

    This separation allows detailed GPU memory traces and per-step debugging to be
    recorded in the file without cluttering the terminal output during long runs.

    Parameters
    ----------
    output_dir : str
        Directory where the log file will be saved. Created if it does not exist.
    task_suite : str
        Name of the LIBERO task suite (e.g., "libero_goal"). Embedded in the
        log filename so logs from different suites don't overwrite each other.

    Returns
    -------
    logger : logging.Logger
        Configured logger instance with both file and console handlers attached.
        The logger is named 'TinyVLA_Extraction' to avoid conflicts with other
        loggers in the process.
    log_file : str
        Absolute path to the created `.log` file. Returned so the main function
        can record the log file location in the extraction summary.

    Notes
    -----
    Log filename format: `extraction_log_{task_suite}_{YYYYMMDD_HHMMSS}.log`
    This ensures each run produces a unique log file, enabling comparison between
    runs without overwriting previous logs.

    Logger hierarchy:
      'TinyVLA_Extraction' (DEBUG)
       ├── FileHandler     → extraction_log_*.log (DEBUG: all messages)
       └── StreamHandler   → console/stderr       (INFO: key milestones only)
    """
    # Ensure the output directory exists; create intermediate directories as needed
    os.makedirs(output_dir, exist_ok=True)

    # Build a unique timestamp string (e.g., "20250601_143207") for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the full log file path
    log_file = os.path.join(output_dir, f"extraction_log_{task_suite}_{timestamp}.log")

    # ── Create the named logger ────────────────────────────────────────────
    logger = logging.getLogger('TinyVLA_Extraction')
    logger.setLevel(logging.DEBUG)  # Accept all message levels; handlers filter further

    # ── File handler: records all messages (DEBUG and above) ───────────────
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)  # Most verbose: captures memory traces, per-step info
    # Timestamp + level + message format for file records
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)

    # ── Console handler: records INFO and above only ────────────────────────
    ch = logging.StreamHandler()  # Default destination: sys.stderr
    ch.setLevel(logging.INFO)     # Less verbose: shows key milestones, warnings, errors
    # Compact format for console: level + message only (no timestamp clutter)
    ch_formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch.setFormatter(ch_formatter)

    # Attach both handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_file


def log_gpu_memory(logger, stage: str):
    """
    Log current GPU memory statistics at a named pipeline stage (DEBUG level).

    Provides three memory metrics for diagnosing VRAM pressure and memory leaks
    during model loading and inference:
      - **Allocated**: Memory currently occupied by live tensors.
      - **Reserved**: Memory reserved by PyTorch's caching allocator but not
        necessarily occupied by live tensors. May differ from allocated when
        tensors have been deleted but the VRAM block hasn't been released yet.
      - **Max Allocated**: Peak allocated memory since the last `reset_peak_stats()`.
        Useful for identifying the high-water mark during a forward pass.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance to write the memory report to (at DEBUG level).
    stage : str
        Human-readable label for the pipeline stage being measured
        (e.g., "Before cleanup", "After base model load"). Embedded in the
        log message to enable timeline reconstruction from the log file.

    Returns
    -------
    tuple of (float, float)
        `(allocated_GB, reserved_GB)` — the currently allocated and reserved
        GPU memory in gigabytes. Returns `(0, 0)` if no CUDA device is available
        (CPU-only machines or environments without GPU access).
    """
    if torch.cuda.is_available():
        # Convert bytes → gigabytes (1 GB = 1024^3 bytes)
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        logger.debug(
            f"[{stage}] GPU Memory - "
            f"Allocated: {allocated:.2f}GB, "
            f"Reserved: {reserved:.2f}GB, "
            f"Max: {max_allocated:.2f}GB"
        )
        return allocated, reserved

    # No CUDA device available: return zeros without logging
    return 0, 0


# ── Global environment variable configuration ──────────────────────────────
# Disable Hugging Face tokenizer parallelism to prevent deadlocks when
# the tokenizer's Rust backend forks multiple threads inside a multiprocessed
# PyTorch DataLoader worker. Safe to disable for single-process inference.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force all model components to use CUDA (GPU) as the compute device
os.environ['DEVICE'] = "cuda"

# Disable Weights & Biases experiment tracking (not needed for embedding extraction)
os.environ["WANDB_DISABLED"] = "true"


# ── LLaVA-Pythia model imports ─────────────────────────────────────────────
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig
# LlavaPythiaConfig: HuggingFace PretrainedConfig subclass containing model hyperparameters:
# action_dim (action vector size), chunk_size (number of future actions predicted per step),
# vision tower config, mm_use_im_start_end flag, etc.

from llava_pythia.conversation import conv_templates, SeparatorStyle
# conv_templates: dict of conversation format templates keyed by name (e.g., "pythia")
# Each template defines how to format the system prompt, user input, and assistant response
# into the token sequence expected by the model.

from llava_pythia.model.builder import load_pretrained_model
# Loads tokenizer, model weights, image processor, and context length from a checkpoint path.
# Supports optional LoRA adapter loading when model_base is provided.

from llava_pythia.mm_utils import tokenizer_image_token, get_model_name_from_path
# tokenizer_image_token: tokenizes a prompt containing IMAGE_TOKEN_INDEX placeholders
# get_model_name_from_path: infers the model name from the checkpoint directory name

from llava_pythia.constants import (
    IMAGE_TOKEN_INDEX,       # Special integer token ID marking where image features are injected
    DEFAULT_IMAGE_TOKEN,     # String placeholder "<image>" inserted into the text prompt
    DEFAULT_IM_START_TOKEN,  # Optional start-of-image marker token (if mm_use_im_start_end=True)
    DEFAULT_IM_END_TOKEN,    # Optional end-of-image marker token
)
from llava_pythia.model import *  # Import all model components (vision towers, projectors, etc.)

from torchvision import transforms  # Standard torchvision image transforms (ToTensor, Resize)
from einops import rearrange        # Tensor reshaping with readable dimension labels

import models.TinyVLA.test.utils.torch_utils as TorchUtils
# TorchUtils: custom rotation utilities (rot_6d_to_euler_angles) and tensor helpers

# ── LIBERO benchmark imports ───────────────────────────────────────────────
from libero.libero import get_libero_path, benchmark
# get_libero_path: returns absolute paths to LIBERO data directories (bddl_files, datasets)
# benchmark: module containing task suite factory functions

from libero.libero.envs import OffScreenRenderEnv
# MuJoCo-based headless rendering environment for LIBERO manipulation tasks

# ── Project utility imports ────────────────────────────────────────────────
from models.TinyVLA.test.utils.libero_utils import (
    get_libero_dummy_action,   # Returns [0,0,0,0,0,0,-1] no-op action for stabilization
    get_libero_image,           # Extracts and flips agentview camera image
    get_libero_wrist_image,     # Extracts and flips wrist camera image
    get_libero_env,             # Environment factory with BDDL/command override support
    quat2axisangle,             # Converts quaternion (x,y,z,w) to axis-angle vector
    extract_command_from_bddl,  # Parses (:language ...) field from a BDDL file
)
from models.TinyVLA.test.utils.robot_utils import set_seed_everywhere
# set_seed_everywhere: seeds all RNGs (Python, NumPy, PyTorch, TF) simultaneously

# Module-level logger placeholder: initialized by setup_logging() in the main function
# and then assigned to this global variable for access across all module-level functions.
logger = None


# ============================================================================
# Constants
# ============================================================================

# Maximum number of simulation steps allowed per episode for each task suite.
# These limits are empirically chosen to allow sufficient time for task completion
# while preventing infinite loops on stuck episodes.
# libero_10 and libero_90 have higher limits because they contain longer-horizon tasks.
TASK_MAX_STEPS = {
    "libero_spatial": 220,  # Short spatial rearrangement tasks
    "libero_object":  280,  # Object manipulation tasks (grasping, placing)
    "libero_goal":    300,  # Goal-conditioned tasks (moderate horizon)
    "libero_10":      520,  # 10 long-horizon demonstration tasks
    "libero_90":      400,  # 90-task generalization benchmark
}


# ============================================================================
# Policy class
# ============================================================================


class llava_pythia_act_policy:
    """
    Wrapper class for the LLaVA-Pythia VLA policy used in TinyVLA evaluation.

    This class manages the full lifecycle of the TinyVLA policy model:
      - Loading the LLaVA-Pythia checkpoint (backbone + LoRA adapters)
      - Loading the model configuration (action_dim, chunk_size, etc.)
      - Preprocessing multimodal inputs (image + language + robot state)
        into the format expected by the model's forward pass
      - Padding images to square aspect ratio for the vision encoder

    Architecture Context:
    ─────────────────────
    TinyVLA (LLaVA-Pythia) is a Vision-Language-Action model built on:
      - **Vision encoder**: CLIP ViT or SigLIP ViT that produces patch tokens
      - **GPT-NeoX / Pythia backbone**: Causal language model that fuses
        visual patch tokens with language tokens via `prepare_inputs_labels_for_multimodal`
      - **Diffusion action head**: Denoising diffusion network that takes the
        backbone's hidden states and predicts a chunk of future robot actions

    Two camera inputs:
      - `image`   : Main agentview camera (third-person scene overview)
      - `image_r` : Wrist camera (robot0_eye_in_hand, close-up of end-effector)

    Attributes
    ----------
    policy : nn.Module
        The loaded LLaVA-Pythia model (backbone + action head).
    tokenizer : PreTrainedTokenizer
        The tokenizer used to convert text prompts to token ID sequences.
    image_processor : BaseImageProcessor
        Hugging Face image processor for normalization and preprocessing.
    context_len : int
        Maximum token sequence length the model supports.
    config : LlavaPythiaConfig
        Model configuration object containing action_dim, chunk_size, etc.
    policy_config : dict
        The configuration dict passed at construction time.
    data_args : Any or None
        Optional data arguments (reserved for future use).
    conv : Conversation
        Conversation template instance, reset at each inference call.
    """

    def __init__(self, policy_config, data_args=None):
        """
        Initialize the policy by loading the model checkpoint.

        Parameters
        ----------
        policy_config : dict
            Configuration dictionary with the following required keys:
              - "model_path"   (str)  : Path to the TinyVLA checkpoint directory
              - "model_base"   (str)  : Path to the base model (for LoRA loading)
              - "enable_lora"  (bool) : Whether the checkpoint uses LoRA adapters
              - "conv_mode"    (str)  : Conversation template key (e.g., "pythia")
              - "action_head"  (str)  : Action head type ("droid_diffusion" or "act")
        data_args : Any, optional
            Optional data processing arguments. Default: None.
        """
        super(llava_pythia_act_policy).__init__()

        # Log initialization boundaries for easy visual scanning in log files
        logger.info("=" * 60)
        logger.info("INITIALIZING POLICY")
        logger.info("=" * 60)

        self.load_policy(policy_config)   # Load model weights, tokenizer, and config
        self.data_args = data_args        # Store optional data arguments
        logger.info("Policy initialization complete")

    def load_policy(self, policy_config):
        """
        Load the LLaVA-Pythia model checkpoint, tokenizer, image processor, and config.

        Performs the following steps in order:
          1. Clears GPU cache to maximize available VRAM before loading.
          2. Resolves model_base (only needed when LoRA adapters are enabled).
          3. Loads the tokenizer, model, image processor, and context_len
             via `load_pretrained_model`.
          4. Loads the LlavaPythiaConfig from the checkpoint's parent directory
             (one level up from the checkpoint folder).

        Parameters
        ----------
        policy_config : dict
            Same configuration dictionary as passed to `__init__`.

        Raises
        ------
        Exception
            Any exception from `load_pretrained_model` is logged with full traceback
            and re-raised to terminate initialization cleanly.

        Notes
        -----
        Config Path Convention:
          The LlavaPythiaConfig is loaded from the parent directory of the
          checkpoint (one level up from `model_path`). This is because LIBERO
          evaluation checkpoints are typically saved as:
            `.../run_name/checkpoint-54000/`   ← model_path
            `.../run_name/`                     ← config location
          Using `'/'.join(model_path.split('/')[:-1])` navigates one level up.

        LoRA Loading:
          `model_base` is passed to `load_pretrained_model` only when
          `enable_lora=True`. When not using LoRA, model_base is set to None
          and the full model weights are loaded directly from model_path.
        """
        global logger  # Access module-level logger (initialized in extract_embeddings_rollout)

        logger.info("Starting policy loading...")
        logger.info(f"Model path: {policy_config['model_path']}")
        logger.info(f"Model base: {policy_config.get('model_base', 'None')}")
        logger.info(f"LoRA enabled: {policy_config.get('enable_lora', False)}")

        # ── Step 1: Free all unused cached VRAM tensors ────────────────────
        logger.info("Clearing GPU cache before loading...")
        log_gpu_memory(logger, "Before cleanup")
        torch.cuda.empty_cache()  # Release VRAM held by PyTorch's caching allocator
        gc.collect()              # Run Python GC to free CPU-side objects referencing GPU memory
        log_gpu_memory(logger, "After cleanup")

        self.policy_config = policy_config  # Store for later reference during inference

        # ── Step 2: Resolve LoRA arguments ────────────────────────────────
        # model_base is the pre-trained backbone onto which LoRA adapters are loaded.
        # Set to None for non-LoRA (full fine-tuned) checkpoints.
        model_base = policy_config["model_base"] if policy_config['enable_lora'] else None

        # Infer the model name from the last component of the checkpoint path
        model_name = get_model_name_from_path(policy_config['model_path'])
        model_path = policy_config["model_path"]

        logger.info(f"Loading model: {model_name}")
        logger.info("Step 1/3: Loading tokenizer...")

        try:
            # ── Step 3: Load model, tokenizer, and image processor ─────────
            # load_pretrained_model returns (tokenizer, model, image_processor, context_len)
            # The two False args disable 8-bit and 4-bit quantization (full precision)
            self.tokenizer, self.policy, self.image_processor, self.context_len = \
                load_pretrained_model(model_path, model_base, model_name, False, False)
            logger.info("Step 2/3: Base model loaded successfully")
            log_gpu_memory(logger, "After base model load")

            # ── Step 4: Load model config from checkpoint parent directory ──
            # Navigate one level up from checkpoint-XXXXX/ to get the config directory
            config_dir = '/'.join(model_path.split('/')[:-1])
            self.config = LlavaPythiaConfig.from_pretrained(
                config_dir,
                trust_remote_code=True  # Required for custom config classes
            )
            logger.info("Step 3/3: Config loaded successfully")

            # Log key model hyperparameters for verification
            logger.info(f"Context length: {self.context_len}")
            logger.info(f"Action dimension: {self.config.action_dim}")
            logger.info(f"Chunk size: {self.config.chunk_size}")
            logger.info("Policy loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load policy: {str(e)}")
            log_gpu_memory(logger, "Error state")
            raise  # Re-raise to propagate to the caller

    def process_batch_to_llava(self, curr_image, robo_state, raw_lang):
        """
        Construct the multimodal input batch dictionary for a single inference step.

        This method assembles all inputs required for one forward pass of the
        LLaVA-Pythia model:
          1. Resets the conversation template for a fresh turn.
          2. Splits the dual-camera image tensor into agentview and wrist images.
          3. Pads both images to square with the image processor's mean color.
          4. Preprocesses both images with the vision encoder's normalizer.
          5. Formats the language command into a conversation-style prompt.
          6. Tokenizes the prompt with image token placeholders.
          7. Packages all tensors into a dict for the model's forward() call.

        Parameters
        ----------
        curr_image : torch.Tensor
            Stacked dual-camera image tensor. Expected shape:
              `(2, C, H, W)` or `(1, 2, C, H, W)` (with batch dimension).
            The first frame (index 0) is the agentview image; the second (index 1)
            is the wrist camera image.
        robo_state : torch.Tensor
            Normalized robot proprioceptive state vector of shape `(state_dim,)`.
            Contains: end-effector xyz position, axis-angle orientation, gripper
            position — all normalized by dataset mean and std.
        raw_lang : str
            The natural language task command string (e.g., "Put the bowl on the stove").
            Used as the user message in the conversation prompt.

        Returns
        -------
        data_dict : dict
            Input batch dictionary with the following keys:
              - "input_ids"       (torch.LongTensor)  : Tokenized prompt, shape (1, seq_len)
              - "attention_mask"  (torch.BoolTensor)  : Non-padding mask, shape (1, seq_len)
              - "images"          (torch.FloatTensor) : Preprocessed agentview image
              - "images_r"        (torch.FloatTensor) : Preprocessed wrist camera image
              - "states"          (torch.FloatTensor) : Robot state, shape (1, 1, state_dim)

        Notes
        -----
        Image Preprocessing:
          The image processor's `preprocess()` is called with:
            - `do_normalize=True`    : Apply mean/std normalization
            - `do_rescale=False`     : Skip [0,255]→[0,1] rescaling (already float)
            - `do_center_crop=False` : Skip center crop (we use random crop resize elsewhere)

        Prompt Format (with mm_use_im_start_end=False):
          "<image>\n{task_command}<|endoftext|>"
          The "<|endoftext|>" token is appended to signal the end of the input
          sequence to the causal language model.

        Conversation Template:
          The conversation template is reset (`conv_templates[...].copy()`) at every
          call to ensure no cross-step context leaks between inference steps.
        """
        # Reset the conversation object for a fresh turn (no history from prior steps)
        self.conv = conv_templates[self.policy_config['conv_mode']].copy()

        # ── Remove the spurious batch dimension if present ─────────────────
        # curr_image may be (1, 2, C, H, W) from unsqueeze in the caller
        if len(curr_image.shape) == 5:
            curr_image = curr_image.squeeze(0)  # → (2, C, H, W)

        # ── Split agentview (image) and wrist (image_r) frames ─────────────
        # torch.chunk splits along dim=0 into two equal tensors of shape (1, C, H, W)
        image, image_r = torch.chunk(curr_image, 2, dim=0)

        # ── Preprocess agentview image ──────────────────────────────────────
        # Pad to square using the image processor's per-channel mean as background color
        image = self.expand2square(image, tuple(x for x in self.image_processor.image_mean))
        image_tensor = self.image_processor.preprocess(
            image,
            return_tensors='pt',     # Return PyTorch tensors
            do_normalize=True,        # Normalize with mean/std from image_processor
            do_rescale=False,         # Input is already in [0,1] float range
            do_center_crop=False      # Crop is handled externally in run_episode
        )['pixel_values']
        # Move to same device and dtype as the model
        image_tensor = image_tensor.to(self.policy.device, dtype=self.policy.dtype)

        # ── Preprocess wrist camera image (same pipeline) ───────────────────
        image_r = self.expand2square(image_r, tuple(x for x in self.image_processor.image_mean))
        image_tensor_r = self.image_processor.preprocess(
            image_r,
            return_tensors='pt',
            do_normalize=True,
            do_rescale=False,
            do_center_crop=False
        )['pixel_values']
        image_tensor_r = image_tensor_r.to(self.policy.device, dtype=self.policy.dtype)

        # ── Format language input with conversation template ───────────────
        inp = raw_lang  # Start with the raw task command string
        assert image is not None, 'image must be provided.'

        # Optionally wrap with start/end image tokens (depends on model config)
        if self.policy.config.mm_use_im_start_end:
            # Format: "<im_start><image><im_end>\n{command}"
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            # Format: "<image>\n{command}"
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

        # Add user message (human role) to the conversation
        self.conv.append_message(self.conv.roles[0], inp)
        image = None  # Clear Python reference (image tensor is already in image_tensor)

        # Add empty assistant placeholder (gpt role with None signals generation start)
        self.conv.append_message(self.conv.roles[1], None)

        # Serialize the full conversation to a single string
        prompt = self.conv.get_prompt()
        # Append the EOS token to signal end of the input to the causal LM
        prompt += " <|endoftext|>"

        # ── Tokenize prompt with image token placeholders ──────────────────
        # tokenizer_image_token places IMAGE_TOKEN_INDEX at each <image> occurrence
        # .unsqueeze(0) adds the batch dimension: (seq_len,) → (1, seq_len)
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        # Attention mask: 1 for all real tokens, 0 for padding tokens
        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Move state to model device and dtype; add batch dim: (state_dim,) → (1, 1, state_dim)
        states = robo_state.to(self.policy.device, dtype=self.policy.dtype)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attn_mask,
            images=image_tensor,         # Agentview camera (main visual input)
            images_r=image_tensor_r,     # Wrist camera (close-up manipulation view)
            states=states.unsqueeze(0)   # Robot proprioceptive state with batch dim
        )
        return data_dict

    def expand2square(self, pil_imgs, background_color):
        """
        Pad a batch of images to a square aspect ratio by centering the content.

        Vision encoders (e.g., CLIP ViT) require square input images. If the input
        image is not square, padding is added symmetrically to the shorter dimension
        with the specified background color. This is the same preprocessing used
        during TinyVLA training, ensuring evaluation preprocessing matches training.

        Parameters
        ----------
        pil_imgs : torch.Tensor
            Input image batch tensor of shape `(B, C, H, W)` in float format.
            Values should be in [0.0, 1.0] (not yet normalized).
        background_color : tuple of float
            Per-channel background fill color used for padding, typically the
            image processor's per-channel mean (e.g., `(0.48145466, 0.4578275, 0.40821073)`
            for CLIP). Using the mean color minimizes the visual impact of the padding.

        Returns
        -------
        torch.Tensor
            Square image batch tensor of shape `(B, max_dim, max_dim, C)` with
            values on the same device and dtype as `pil_imgs`.
            Note: channels are in the LAST dimension (HWC format), as returned
            after `.permute(0, 2, 3, 1)` — this matches the image processor's
            expected input format.

        Notes
        -----
        Three cases are handled:
          - H == W : Image is already square; no padding needed.
          - H > W  : Image is taller than wide; pad left and right (width dimension).
          - H < W  : Image is wider than tall; pad top and bottom (height dimension).

        The offset calculation `(max_dim - short_dim) // 2` centers the content
        within the padded square, which is more natural than left/top alignment.
        """
        batch_size, channels, height, width = pil_imgs.shape
        max_dim = max(height, width)  # Target square side length

        # Allocate padded output array filled with background_color
        # Shape: (B, max_dim, max_dim, C) in HWC format (channels last for numpy)
        expanded_imgs = np.full(
            (batch_size, max_dim, max_dim, channels),
            background_color,
            dtype=np.float32
        )

        if height == width:
            # No padding needed: convert directly from BCHW tensor to BHWC numpy
            expanded_imgs = pil_imgs.permute(0, 2, 3, 1).cpu().numpy()

        elif height > width:
            # Taller than wide: pad left and right symmetrically
            offset = (max_dim - width) // 2   # Number of padding pixels on each side
            # Copy image content into the centered position in the padded array
            expanded_imgs[:, :height, offset:offset + width, :] = \
                pil_imgs.permute(0, 2, 3, 1).cpu().numpy()

        else:
            # Wider than tall: pad top and bottom symmetrically
            offset = (max_dim - height) // 2
            expanded_imgs[:, offset:offset + height, :width, :] = \
                pil_imgs.permute(0, 2, 3, 1).cpu().numpy()

        # Convert back to PyTorch tensor on the original device and dtype
        expanded_imgs = torch.tensor(
            expanded_imgs,
            dtype=pil_imgs.dtype,
            device=pil_imgs.device
        )
        return expanded_imgs


# ============================================================================
# Helpers
# ============================================================================


def convert_actions(pred_action):
    """
    Convert a 10-dimensional predicted action vector to the 7-dimensional format
    expected by the LIBERO environment's robot controller.

    TinyVLA's action head outputs actions in a 10D representation:
      - `pred_action[0:3]`  : End-effector XYZ position delta (3D)
      - `pred_action[3:9]`  : Rotation in 6D continuous representation (rot6d)
      - `pred_action[9]`    : Gripper command (scalar, normalized)

    LIBERO's environment controller expects a 7D action:
      - `action[0:3]`  : End-effector XYZ position delta (3D, unchanged)
      - `action[3:6]`  : Rotation in Euler angles XYZ convention (3D)
      - `action[6]`    : Gripper command (scalar, unchanged)

    The conversion applies the rot6d → Euler angles transformation using
    `TorchUtils.rot_6d_to_euler_angles`, which reconstructs the full rotation
    matrix from the 6D representation and then extracts Euler angles.

    Why 6D rotation?
    ────────────────
    The 6D continuous rotation representation (Zhou et al., 2019) is better
    suited for neural network outputs than Euler angles because:
      - It is free from gimbal lock (singularity at 90° pitch).
      - It forms a continuous, connected space (unlike Euler angles which
        have discontinuities at ±180° boundaries).
      - Networks learn it more reliably from demonstration data.

    Parameters
    ----------
    pred_action : numpy.ndarray
        The raw 10-dimensional action vector output by the policy, after
        post-processing (denormalization). Shape: `(10,)`.

    Returns
    -------
    numpy.ndarray
        The converted 7-dimensional action vector `[dx, dy, dz, rx, ry, rz, gripper]`
        where `[rx, ry, rz]` are Euler angles in the XYZ convention (radians).
        Shape: `(7,)`.
    """
    cur_xyz = pred_action[:3]       # End-effector position delta (x, y, z)
    cur_rot6d = pred_action[3:9]    # 6D rotation representation (6 values)
    # Expand gripper scalar to (1,) array for safe concatenation
    cur_gripper = np.expand_dims(pred_action[-1], axis=0)

    # Convert rot6d numpy array to torch tensor and add batch dimension for the utility function
    cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)  # (1, 6)

    # Convert 6D rotation → Euler angles (XYZ convention), then squeeze batch dim
    cur_euler = TorchUtils.rot_6d_to_euler_angles(
        rot_6d=cur_rot6d,
        convention="XYZ"
    ).squeeze().numpy()  # (3,)

    # Assemble the final 7D action: xyz + euler + gripper
    return np.concatenate((cur_xyz, cur_euler, cur_gripper))


def get_obs(obs, stats):
    """
    Extract, resize, and normalize observations from the LIBERO environment.

    Processes the raw environment observation dict into the format expected
    by TinyVLA's inference pipeline:
      - **Images**: Both cameras are resized to (320, 180) pixels (landscape
        format matching training data) and 180°-rotated to correct MuJoCo's
        inverted rendering coordinate system.
      - **Robot state**: End-effector position + orientation + gripper state
        are concatenated and z-score normalized using dataset statistics.

    Parameters
    ----------
    obs : dict
        Raw observation dictionary from `env.step()` or `env.reset()`.
        Expected keys:
          - "agentview_image"          : (H, W, 3) uint8 array (third-person view)
          - "robot0_eye_in_hand_image" : (H, W, 3) uint8 array (wrist camera)
          - "robot0_eef_pos"           : (3,) float array — end-effector XYZ position
          - "robot0_eef_quat"          : (4,) float array — end-effector quaternion (x,y,z,w)
          - "robot0_gripper_qpos"      : (2,) float array — gripper joint positions
    stats : dict
        Dataset normalization statistics loaded from `dataset_stats.pkl`. Required keys:
          - "qpos_mean" : (state_dim,) float array — per-dimension mean
          - "qpos_std"  : (state_dim,) float array — per-dimension std deviation

    Returns
    -------
    images : numpy.ndarray
        Stacked dual-camera image array of shape `(2, 180, 320, 3)` with dtype
        matching the input (uint8). Index 0 = agentview, index 1 = wrist camera.
    states : numpy.ndarray
        Z-score normalized robot state vector of shape `(state_dim,)` where
        state_dim = 3 (pos) + 3 (axis-angle) + 2 (gripper) = 8.

    Notes
    -----
    Rotation Convention:
      `quat2axisangle` converts the robot's end-effector quaternion `(x,y,z,w)` to
      a 3D axis-angle vector `(ax*θ, ay*θ, az*θ)`, where the magnitude encodes the
      rotation angle θ and the direction encodes the rotation axis. This representation
      is used by the LIBERO robot controller for orientation control.

    Z-score Normalization:
      `(states - mean) / std` maps each dimension to approximately N(0,1).
      This normalization was applied during training data preprocessing, so it
      must be replicated identically during inference.

    Image Flip:
      `[::-1, ::-1]` reverses both height (rows) and width (columns) axes,
      equivalent to a 180° rotation. This corrects MuJoCo's inverted Y-axis
      rendering (identical to `get_libero_image` in `libero_utils.py`).
    """
    images = np.array([
        # Agentview: flip 180° then resize to (width=320, height=180) for training format
        cv2.resize(obs['agentview_image'][::-1, ::-1], (320, 180)),
        # Wrist camera: same 180° flip and resize
        cv2.resize(obs['robot0_eye_in_hand_image'][::-1, ::-1], (320, 180))
    ])

    # Concatenate robot state components into a single vector:
    # [eef_x, eef_y, eef_z, ax*θ, ay*θ, az*θ, gripper_left, gripper_right]
    states = np.concatenate((
        obs["robot0_eef_pos"],                   # (3,) XYZ end-effector position
        quat2axisangle(obs["robot0_eef_quat"]),  # (3,) axis-angle orientation
        obs["robot0_gripper_qpos"]               # (2,) gripper joint positions
    ))

    # Z-score normalization: center and scale each dimension by training statistics
    states = (states - stats["qpos_mean"]) / stats["qpos_std"]

    return images, states


def build_bddl_path(task, level: str) -> str:
    """
    Construct the full filesystem path to the BDDL definition file for a task
    at a specified command variation level.

    This function is used in the extraction loop to locate the BDDL file for
    each (task, level) combination without going through the full `get_libero_env`
    factory. It supports two cases:
      - "default": The standard BDDL file registered with the task object.
      - "l1"/"l2"/"l3": A synonym BDDL file following the naming convention
        `{base_name}_syn_{level}.bddl` in the same directory.

    Parameters
    ----------
    task : libero.libero.benchmark.Task
        LIBERO task object providing:
          - `task.problem_folder` : Subdirectory under the BDDL files root
          - `task.bddl_file`      : Default BDDL filename for this task
    level : str
        Command variation level: "default", "l1", "l2", or "l3".

    Returns
    -------
    str
        Absolute path to the BDDL file. Note: this path is not validated —
        the caller should check `os.path.exists()` before using it.

    Examples
    --------
    For a task with bddl_file = "put_the_bowl_on_the_stove.bddl":
      build_bddl_path(task, "default")
        → "/path/to/bddl_files/libero_goal/put_the_bowl_on_the_stove.bddl"
      build_bddl_path(task, "l2")
        → "/path/to/bddl_files/libero_goal/put_the_bowl_on_the_stove_syn_l2.bddl"
    """
    if level == "default":
        # Return the standard BDDL path directly from task attributes
        return os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file
        )

    # Strip ".bddl" extension to get the base filename stem
    base_name = task.bddl_file.replace(".bddl", "")

    # Construct the synonym variation filename following the naming convention
    bddl_filename = f"{base_name}_syn_{level}.bddl"

    return os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        bddl_filename
    )


# ============================================================================
# Embedding Extraction via Forward Hook
# ============================================================================


class EmbeddingCapture:
    """
    Non-invasive embedding extractor using a PyTorch forward hook on the GPT-NeoX backbone.

    A PyTorch forward hook is a callback registered on an `nn.Module` that is
    automatically called after that module's `forward()` method completes. The hook
    receives `(module, input, output)` and can capture any intermediate activation
    without modifying the model's source code.

    This class attaches such a hook to TinyVLA's GPT-NeoX / Pythia backbone (the
    language model component) to capture its output hidden states immediately BEFORE
    they are projected by the diffusion action head. This is the richest available
    representation of how the model jointly encodes language, vision, and robot state.

    Text-Only Pooling Strategy:
    ────────────────────────────
    After multimodal fusion (in `prepare_inputs_labels_for_multimodal`), the token
    sequence contains both text tokens and visual patch tokens:

      [text_before_image | visual_patch_1 ... visual_patch_N | text_after_image]

    We pool ONLY the text token hidden states (before and after the visual block)
    to produce the embedding. This ensures that the extracted vector reflects the
    model's language understanding rather than the visual content, making it
    appropriate for comparing different linguistic commands.

    This approach is consistent with the OpenVLA embedding extraction methodology.

    Attributes
    ----------
    hidden_states : numpy.ndarray or None
        The last captured text-only mean-pooled embedding of shape `(1, hidden_dim)`
        as a numpy float32 array. None if no forward pass has been run yet.
    _handle : torch.utils.hooks.RemovableHook or None
        The handle returned by `register_forward_hook`, used to unregister the hook.
    _input_ids : torch.LongTensor or None
        The ORIGINAL input_ids (before multimodal fusion) stored before each forward
        pass so the hook can determine where visual tokens were inserted.
    """

    def __init__(self):
        """Initialize the EmbeddingCapture with empty state."""
        self.hidden_states = None   # Will hold the captured embedding array
        self._handle = None         # Hook handle (None until register() is called)
        self._input_ids = None      # Input IDs before multimodal fusion (set per step)
        logger.debug("EmbeddingCapture initialized")

    def set_input_ids(self, input_ids):
        """
        Store the original (pre-fusion) input_ids before each forward pass.

        This must be called immediately before each `policy.policy(**batch, eval=True)`
        call. The stored IDs are used by `hook_fn` to determine the position of the
        IMAGE_TOKEN_INDEX placeholder in the original sequence, which allows the hook
        to compute exactly how many visual patch tokens replaced it in the fused sequence.

        Parameters
        ----------
        input_ids : torch.LongTensor
            The tokenized prompt tensor of shape `(1, seq_len)` containing
            IMAGE_TOKEN_INDEX at the position where the image was inserted.
        """
        self._input_ids = input_ids  # Store for access during the next hook callback

    def hook_fn(self, module, input, output):
        """
        Forward hook callback: capture and pool backbone hidden states.

        Called automatically by PyTorch after each forward pass through the
        registered backbone module. Performs text-only mean pooling of the
        backbone's output hidden states.

        Parameters
        ----------
        module : nn.Module
            The backbone module that triggered the hook (GPT-NeoX / Pythia).
        input : tuple
            Inputs passed to the backbone's forward() method (not used here).
        output : tuple
            Outputs of the backbone's forward() method.
            `output[0]` contains the hidden states tensor of shape
            `(B, seq_len_full, hidden_dim)` where `seq_len_full` is the expanded
            sequence length after visual patch token insertion.

        Notes
        -----
        Sequence Length Calculation:
          After `prepare_inputs_labels_for_multimodal`, the sequence length expands:
            `seq_len_full = seq_len_original - 1 + n_img`
          Solving for n_img (number of visual patch tokens):
            `n_img = seq_len_full - (seq_len_original - 1)`
          This uses `hs.shape[1]` for seq_len_full and `self._input_ids.shape[1]`
          for seq_len_original.

        `.detach().cpu().float()`:
          - `.detach()`: Removes the tensor from the computation graph to prevent
            accidental gradient tracking during embedding storage.
          - `.cpu()`: Moves the tensor to CPU memory (free from VRAM pressure).
          - `.float()`: Converts to float32 for consistent downstream processing,
            regardless of whether the model runs in float16 or bfloat16.
        """
        hs = output[0]  # Backbone output hidden states: (B, seq_len_full, hidden_dim)

        if self._input_ids is not None and (self._input_ids == IMAGE_TOKEN_INDEX).any():
            # ── Locate the image token position in the ORIGINAL input_ids ──
            # Find the index of IMAGE_TOKEN_INDEX in the 1D token sequence
            img_pos = (self._input_ids[0] == IMAGE_TOKEN_INDEX) \
                .nonzero(as_tuple=True)[0][0].item()

            # ── Compute number of visual patch tokens that replaced the placeholder ──
            # Original sequence: [... text_A | IMAGE_TOKEN_INDEX | text_B ...]  length = L_orig
            # Fused sequence:    [... text_A | patch_1 ... patch_N | text_B ...] length = L_full
            # Relation: L_full = L_orig - 1 + N  →  N = L_full - (L_orig - 1)
            n_img = hs.shape[1] - (self._input_ids.shape[1] - 1)

            # ── Extract text-only hidden states by excluding the visual block ──
            text_hidden = torch.cat([
                hs[:, :img_pos, :],            # Tokens BEFORE the visual block
                hs[:, img_pos + n_img:, :],    # Tokens AFTER the visual block
            ], dim=1)  # (B, text_seq_len, hidden_dim)

            # Mean-pool over text token sequence → (B, hidden_dim)
            self.hidden_states = (
                text_hidden.mean(dim=1)  # Average over text token positions
                .detach()                # Detach from computation graph
                .cpu()                   # Move to CPU
                .float()                 # Convert to float32
                .numpy()                 # Convert to numpy array
            )
        else:
            # Fallback: no image tokens in the sequence (text-only input)
            # Pool the entire hidden state sequence
            self.hidden_states = (
                hs.mean(dim=1).detach().cpu().float().numpy()
            )

    def register(self, model):
        """
        Register the forward hook on the LLaVA-Pythia model's backbone.

        The hook is attached to the module returned by `model.get_model()`, which
        is the GPT-NeoX / Pythia backbone (excluding the vision encoder and
        the action head). This ensures the hook fires after language-visual
        fusion but before action prediction.

        Parameters
        ----------
        model : nn.Module
            The full LLaVA-Pythia model instance (as stored in `policy.policy`).

        Notes
        -----
        `register_forward_hook` returns a `RemovableHook` handle stored in
        `self._handle`. This handle must be used to remove the hook when extraction
        is complete, otherwise memory leaks and unexpected behavior may occur during
        subsequent model operations.
        """
        backbone = model.get_model()  # Get the GPT-NeoX backbone sub-module
        # Register hook_fn to be called after every backbone forward pass
        self._handle = backbone.register_forward_hook(self.hook_fn)
        logger.info("Forward hook registered on GPT-NeoX backbone")

    def remove(self):
        """
        Deregister the forward hook and release the hook handle.

        Should be called after all embedding extraction is complete to prevent
        the hook from interfering with any subsequent model operations and to
        allow the hook's closure memory to be garbage collected.
        """
        if self._handle is not None:
            self._handle.remove()  # Deregister from the backbone module
            self._handle = None    # Clear reference to allow GC
            logger.debug("Forward hook removed")

    def get_embedding(self):
        """
        Return the most recently captured text-token mean-pooled embedding.

        Converts the stored `(1, hidden_dim)` array to a 1D `(hidden_dim,)` vector
        by squeezing the batch dimension.

        Returns
        -------
        numpy.ndarray or None
            The captured embedding as a 1D float32 array of shape `(hidden_dim,)`.
            Returns None if no forward pass has been captured yet (e.g., if called
            before the first `policy.policy(**batch)` call).
        """
        if self.hidden_states is None:
            return None
        return self.hidden_states.squeeze(0)  # Remove batch dim: (1, D) → (D,)

# ============================================================================
# Episode Runner with Embedding Extraction
# ============================================================================


def run_episode_with_embeddings(
    env,
    task_description,
    policy,
    policy_config,
    stats,
    emb_capture,
    initial_state=None,
    max_steps=300,
    num_steps_wait=10,
    first_step_only=False,
):
    """
    Execute a single LIBERO episode with TinyVLA and capture per-step
    pre-action-head embeddings via a registered forward hook.

    This is the innermost execution unit of the embedding extraction pipeline.
    It runs the complete TinyVLA inference loop — including temporal aggregation
    (ACT-style action chunking) — for a single rollout episode, while passively
    capturing one hidden-state embedding per inference step via the
    `EmbeddingCapture` hook.

    Episode Execution Phases
    ------------------------
    The episode proceeds through four sequential phases:

    **Phase 1 — Environment Reset** (pre-loop):
      The environment is reset and an optional fixed initial state is loaded via
      `env.set_init_state()`. Using fixed initial states ensures each rollout
      starts from exactly the same object configuration, enabling reproducible
      comparisons across command levels.

    **Phase 2 — Physics Stabilization** (t = 0 .. num_steps_wait - 1):
      `num_steps_wait` dummy no-op actions are sent to the simulation.
      MuJoCo's physics engine may produce residual velocities from object
      placement during `set_init_state()`. Waiting for stabilization ensures
      that the first real observation reflects a physically settled scene.

    **Phase 3 — GPU Kernel Warmup** (t = 0, 10 silent forward passes):
      Ten forward passes are run on the first observation WITHOUT advancing
      the simulation or recording embeddings. This pre-heats PyTorch's CUDA
      kernel compilation and cuDNN algorithm selection caches, so all
      subsequent inference steps run at steady-state GPU throughput.

    **Phase 4 — Main Rollout Loop** (t = 0 .. max_timesteps - 1):
      At each timestep:
        a. Observe: extract dual-camera images and proprioceptive robot state.
        b. Preprocess: resize, crop, and normalize images and state.
        c. Infer: run the policy forward pass (hook fires automatically).
        d. Capture: retrieve the embedding captured by the hook.
        e. Aggregate: compute the smoothed action via temporal aggregation.
        f. Execute: step the simulation with the computed action.
        g. Check: if `done=True`, mark success and terminate.

    Temporal Aggregation (ACT-style Action Chunking)
    -------------------------------------------------
    TinyVLA predicts a **chunk** of `chunk_size` future actions at each step t.
    Rather than always executing only the first prediction, temporal aggregation
    maintains a sliding buffer `all_time_actions` of shape
    `(max_steps, max_steps + chunk_size, action_dim)` where row `i` stores the
    chunk predicted at step `i`.

    The action executed at step t is computed as the exponentially weighted
    average of ALL predictions made for step t (from steps 0 through t):

    \[
        a_t = \frac{\sum_{i=0}^{t} e^{-k \cdot (t-i)} \hat{a}_t^{(i)}}
                   {\sum_{i=0}^{t} e^{-k \cdot (t-i)}}
    \]

    where \(\hat{a}_t^{(i)}\) is the prediction for time t made at step i,
    and k = 0.01 is the decay rate (recent predictions weighted higher).
    This produces smoother, more consistent actions by blending predictions
    across multiple planning horizons.

    Parameters
    ----------
    env : OffScreenRenderEnv
        An initialized, seeded LIBERO off-screen simulation environment ready
        for reset and stepping. Should already have the correct BDDL file
        and camera resolution configured.
    task_description : str
        Natural language command string passed to the policy at every step
        (e.g., "Put the bowl on the stove"). This is the BDDL (:language ...)
        field, potentially already overridden by a variation or ablation BDDL.
    policy : llava_pythia_act_policy
        Fully initialized TinyVLA policy wrapper with the model loaded.
        The model should already have the `EmbeddingCapture` hook registered
        on its backbone before this function is called.
    policy_config : dict
        Policy configuration dictionary. Required keys:
          - "action_head" (str): "droid_diffusion" or "act" — determines
            which post-processing (denormalization) formula to apply.
    stats : dict
        Dataset normalization statistics loaded from `dataset_stats.pkl`.
        Required keys depend on the action head type:
          - For "droid_diffusion": "action_min", "action_max"
          - For "act":             "action_mean", "action_std"
          - Always required:       "qpos_mean", "qpos_std" (for robot state)
    emb_capture : EmbeddingCapture
        Pre-configured hook object with `set_input_ids()` called immediately
        before each forward pass. The hook fires automatically during the
        model's forward pass and stores the text-mean-pooled hidden states.
    initial_state : object or None, optional
        LIBERO initial state object from `task_suite.get_task_init_states()`.
        When provided, `env.set_init_state(initial_state)` is called to
        deterministically reproduce a specific scene configuration (object
        positions, orientations). If None, the environment uses its default
        random reset. Default: None.
    max_steps : int, optional
        Maximum number of inference steps before the episode times out.
        Should be set per-suite using `TASK_MAX_STEPS`. Default: 300.
    num_steps_wait : int, optional
        Number of no-op dummy actions to execute before starting inference,
        used for MuJoCo physics stabilization. Default: 10.
    first_step_only : bool, optional
        If True, the function captures the embedding from t=0 only and
        returns immediately without executing any robot actions or advancing
        the simulation. This is the fast-extraction mode for offline
        language representation analysis. Default: False.

    Returns
    -------
    embeddings : list of numpy.ndarray
        List of per-step embedding arrays, each of shape `(hidden_dim,)`.
        In full rollout mode, `len(embeddings)` equals the number of
        inference steps executed (≤ max_steps). In first_step_only mode,
        `len(embeddings) == 1`.
    success : bool
        True if the simulation returned `done=True` (task succeeded) before
        reaching max_steps. Always False in first_step_only mode (no actions
        are executed, so the task is never attempted).
    num_steps : int
        The total number of inference steps for which embeddings were captured.
        In first_step_only mode, always 1.

    Raises
    ------
    No exceptions are propagated. Any exception during the rollout loop is
    caught, logged with full traceback at ERROR level, and the function returns
    whatever partial results were accumulated before the failure.

    Notes
    -----
    Image Crop-Resize Augmentation:
      For the "droid_diffusion" action head, a center-preserving 5% spatial crop
      followed by resize-back is applied to match the training data augmentation
      pipeline. This is skipped for the "act" head, which was trained without
      this augmentation.

    `torch.inference_mode()`:
      Wraps the entire rollout in `torch.inference_mode()` rather than
      `torch.no_grad()`. The stronger guarantee of `inference_mode` disables
      both gradient computation AND gradient tracking metadata, yielding
      slightly lower memory overhead during long rollouts.

    Action Execution Format:
      `env.step()` requires a Python `list` of floats, not a numpy array.
      `.tolist()` is called on the final 7D action before stepping.
    """
    logger.debug(f"Starting episode with task: '{task_description}'")
    logger.debug(f"Max steps: {max_steps}, First step only: {first_step_only}")

    # ── Phase 1: Environment Reset ─────────────────────────────────────────
    env.reset()  # Initial reset (required before set_init_state on some envs)

    # Build the image-to-tensor transform: HWC uint8 → CHW float32 in [0,1]
    to_tensor = transforms.ToTensor()

    # Load fixed initial state for reproducibility, or fall back to random reset
    if initial_state is not None:
        obs = env.set_init_state(initial_state)   # Deterministic start from saved state
        logger.debug("Set initial state from checkpoint")
    else:
        obs = env.reset()   # Non-deterministic random reset
        logger.debug("Reset environment to default state")

    # ── Configuration ──────────────────────────────────────────────────────
    action_dim = policy.config.action_dim   # Dimensionality of raw action space (e.g., 10 for TinyVLA)
    policy.policy.eval()    # Disable dropout and batch norm training mode for inference

    # ── Action Denormalization (post-processing) ───────────────────────────
    # Define the inverse normalization function to convert model output → robot action space.
    # The function is chosen based on the normalization used during training.
    if policy_config["action_head"] == 'droid_diffusion':
        # Diffusion head outputs values in [-1, 1]; map back to [action_min, action_max]
        post_process = lambda a: (
            ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
        )
    elif policy_config["action_head"] == 'act':
        # ACT head outputs zero-mean unit-variance values; map back with mean + std scaling
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    else:
        # Default fallback: use diffusion-style denormalization for unknown head types
        post_process = lambda a: (
            ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
        )

    # ── Temporal Aggregation Setup ─────────────────────────────────────────
    temporal_agg = True      # Enable ACT-style temporal aggregation (strongly recommended)
    query_frequency = 1      # Query the policy at EVERY timestep (no subsampling)
    num_queries = policy.config.chunk_size   # Number of future actions per chunk prediction
    max_timesteps = int(max_steps)           # Cast to int for safety in range comparisons

    if temporal_agg:
        # Pre-allocate the full aggregation buffer as a zero-filled GPU tensor.
        # Shape explanation:
        #   Dim 0 (max_timesteps): one row per inference step i that made a prediction
        #   Dim 1 (max_timesteps + num_queries): all possible future target timesteps
        #   Dim 2 (action_dim): action vector components
        # at row i, the policy's chunk prediction for times [i, i+num_queries) is stored.
        all_time_actions = torch.zeros(
            [max_timesteps, max_timesteps + num_queries, action_dim],
            dtype=torch.float32
        ).cuda()

    embeddings = []     # Accumulator for per-step embedding vectors
    t = 0               # Unified step counter (used in both stabilization and main loop)
    success = False     # Episode success flag (set True only when env returns done=True)

    with torch.inference_mode():    # Strongest no-grad context: disables all grad tracking
        try:
            # ── Phase 2: Physics Stabilization ────────────────────────────
            logger.debug(
                f"Waiting {num_steps_wait} steps for environment stabilization..."
            )
            while t < num_steps_wait:
                # Step with a do-nothing action: [dx=0, dy=0, dz=0, drx=0, dry=0, drz=0, gripper=-1]
                obs, _, _, _ = env.step(get_libero_dummy_action("tiny_vla"))
                t += 1  # Advance step counter

            # Reset counter: main inference loop counts from t=0 separately
            t = 0
            logger.debug("Starting main rollout loop...")

            # ── Phase 4: Main Rollout Loop ─────────────────────────────────
            while t < max_timesteps:

                # Periodic progress log to track long episodes (every 50 steps)
                if t % 50 == 0 and t > 0:
                    logger.debug(f"Episode progress: {t}/{max_timesteps} steps")

                # ── Step 4a: Observation Extraction ───────────────────────
                # get_obs resizes, flips, and normalizes both camera images and robot state
                traj_rgb_np, robot_state = get_obs(obs=obs, stats=stats)

                # Move normalized robot state to GPU as float32 tensor
                robot_state = torch.from_numpy(robot_state).float().cuda()

                # ── Step 4b: Image Preprocessing (every query_frequency steps) ──
                if t % query_frequency == 0:
                    curr_image = []
                    for img in traj_rgb_np:
                        # Convert each (H, W, C) uint8 numpy image → (C, H, W) float32 in [0,1]
                        curr_image.append(to_tensor(img).float().cuda())

                    # Stack along dim=0: list of 2 tensors (C,H,W) → (2, C, H, W)
                    curr_image = torch.stack(curr_image, dim=0)

                    # ── Random Crop-Resize Augmentation (diffusion head only) ──
                    # Replicates the 5% center-preserving random crop from training.
                    # Skipped for ACT head which uses different preprocessing.
                    if policy_config["action_head"] != 'act':
                        original_size = curr_image.shape[-2:]   # Capture (H, W) before crop

                        ratio = 0.95    # Retain 95% of each spatial dimension (5% crop)

                        # Crop symmetrically: each side loses (1-ratio)/2 of its dimension
                        curr_image = curr_image[:, :,
                            int(original_size[0] * (1 - ratio) / 2) :   # Top crop start
                            int(original_size[0] * (1 + ratio) / 2),     # Top crop end
                            int(original_size[1] * (1 - ratio) / 2) :   # Left crop start
                            int(original_size[1] * (1 + ratio) / 2)      # Left crop end
                        ]

                        curr_image = curr_image.squeeze(0)  # Remove batch dim: (2,C,H',W') → (2,C,H',W')

                        # Resize cropped image back to the original resolution (antialias=True
                        # uses a Lanczos-like filter to reduce aliasing during downsampling)
                        resize_transform = transforms.Resize(original_size, antialias=True)
                        curr_image = resize_transform(curr_image)

                        curr_image = curr_image.unsqueeze(0)   # Re-add batch dim for model input

                # ── Phase 3: GPU Kernel Warmup (first step only, 10 passes) ──
                # Runs 10 silent forward passes to initialize CUDA runtime caches.
                # These passes DO update `emb_capture.hidden_states` but that
                # state is overwritten by the real query immediately after.
                if t == 0:
                    logger.debug("Running warmup inference (10 iterations)...")
                    for warmup_iter in range(10):
                        batch = policy.process_batch_to_llava(
                            curr_image, robot_state, task_description
                        )
                        # set_input_ids MUST be called before forward pass so hook
                        # can correctly locate image token positions in the sequence
                        emb_capture.set_input_ids(batch['input_ids'])
                        policy.policy(**batch, eval=True)   # Forward pass (hook fires)
                    logger.debug("Warmup complete")

                # ── Step 4c: Policy Inference (real query) ─────────────────
                if t % query_frequency == 0:
                    # Build the multimodal input dict (images + language + state)
                    batch = policy.process_batch_to_llava(
                        curr_image, robot_state, task_description
                    )
                    # CRITICAL: must be set before forward() so hook_fn has the
                    # original token IDs to compute visual patch token positions
                    emb_capture.set_input_ids(batch['input_ids'])

                    # Forward pass: runs backbone + action head.
                    # Side effect: EmbeddingCapture.hook_fn fires and stores hidden states.
                    # Returns: all_actions of shape (1, chunk_size, action_dim)
                    all_actions = policy.policy(**batch, eval=True)

                # ── Step 4d: Embedding Capture ─────────────────────────────
                # Retrieve the text-mean-pooled hidden state stored by the hook
                embedding = emb_capture.get_embedding()

                if embedding is not None:
                    embeddings.append(embedding)    # Append (hidden_dim,) vector to list
                    if t == 0:
                        logger.debug(
                            f"First embedding captured, shape: {embedding.shape}"
                        )
                else:
                    # Hook did not fire or backbone produced no output (should not happen)
                    logger.warning(f"No embedding captured at step {t}")

                # ── First-Step-Only Early Return ───────────────────────────
                # In first_step_only mode, we only need the t=0 embedding.
                # Return immediately: no actions are executed, success is undefined.
                if first_step_only:
                    logger.debug("First step extraction complete, returning early")
                    return embeddings, False, 1     # (embeddings, success=False, steps=1)

                # ── Step 4e: Temporal Aggregation ─────────────────────────
                if temporal_agg:
                    # Store this step's predicted action chunk into the buffer.
                    # all_actions has shape (1, chunk_size, action_dim); we store it at
                    # row t, columns t..t+num_queries of the aggregation matrix.
                    all_time_actions[[t], t:t + num_queries] = all_actions

                    # Retrieve all predictions ever made for the CURRENT step t.
                    # Column t of all_time_actions contains: row 0's prediction for t,
                    # row 1's prediction for t, ..., row t's prediction for t.
                    actions_for_curr_step = all_time_actions[:, t]  # (max_timesteps, action_dim)

                    # Filter out rows that were never populated (remain zero from initialization).
                    # A row is considered populated if ALL action dimensions are non-zero.
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    # Shape after filtering: (num_populated_rows, action_dim)

                    # Exponential decay weights: older predictions (smaller index = made earlier)
                    # receive lower weight; most recent prediction receives highest weight.
                    k = 0.01    # Decay rate: small k → slow exponential decay
                    # np.arange(n) = [0, 1, ..., n-1] where 0 = oldest prediction
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()   # Normalize: weights sum to 1

                    # Move to GPU as column vector (n, 1) for elementwise broadcasting
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)

                    # Weighted sum: Σ(weight_i × action_i) across all populated predictions
                    # keepdim=True preserves shape (1, action_dim) for downstream squeeze
                    raw_action = (actions_for_curr_step * exp_weights).sum(
                        dim=0, keepdim=True
                    )
                else:
                    # Without temporal aggregation: directly use the prediction for this step
                    # (the t-th action in the current chunk, modulo query_frequency)
                    raw_action = all_actions[:, t % query_frequency]

                # ── Step 4f: Action Execution ──────────────────────────────
                # Remove batch dimension and move to CPU numpy for post-processing
                raw_action = raw_action.squeeze(0).cpu().numpy()   # (action_dim,) numpy array

                # Denormalize: map from normalized space back to robot action range
                action = post_process(raw_action)   # Still 10D (xyz + rot6d + gripper)

                # Convert from 10D (xyz + rot6d + gripper) to 7D (xyz + euler + gripper)
                action = convert_actions(action)    # (7,) numpy array

                # Execute the action in simulation; returns next obs, reward, done flag, info
                # env.step() requires a Python list, not a numpy array
                obs, reward, done, info = env.step(action.tolist())

                # ── Step 4g: Termination Check ─────────────────────────────
                if done:
                    success = True   # Task was completed successfully
                    logger.debug(f"Episode completed successfully at step {t}")
                    break   # Exit the main rollout loop

                t += 1  # Advance the step counter for the next iteration

        except Exception as e:
            # Catch all exceptions to prevent a single bad episode from crashing the
            # entire extraction run. Log both the message and the full stack trace.
            logger.error(f"Episode error at step {t}: {str(e)}")
            logger.exception("Full traceback:")  # Includes the full Python traceback

    logger.debug(
        f"Episode finished: {len(embeddings)} embeddings extracted, success={success}"
    )
    return embeddings, success, len(embeddings)


# ============================================================================
# Main Extraction
# ============================================================================


def extract_embeddings_rollout(
    model_path: str,
    model_base: str,
    task_suite_name: str = "libero_goal",
    command_levels=("default", "l1", "l2", "l3"),
    output_dir: str = "/mnt/beegfs/a.cardamone7/outputs/embeddings/tinyvla",
    resolution: int = 256,
    seed: int = 0,
    num_rollouts_per_task: int = 10,
    first_step_only: bool = False,
):
    """
    Orchestrate the full TinyVLA pre-action-head embedding extraction pipeline.

    This is the top-level function that coordinates every stage of the
    embedding extraction process, from model loading to final `.pkl` output.
    It iterates over all tasks in a LIBERO benchmark suite and over all
    specified linguistic command levels, running `num_rollouts_per_task`
    rollout episodes per (task, level) pair.

    Pipeline Overview
    -----------------
    1. **Logging Setup**: Initialize dual-destination logger (file + console).
    2. **Policy Loading**: Load the LLaVA-Pythia model checkpoint and config.
    3. **Stats Loading**: Load dataset normalization statistics from `.pkl`.
    4. **Hook Registration**: Attach the `EmbeddingCapture` forward hook to
       the GPT-NeoX backbone.
    5. **Task Suite Loading**: Instantiate the LIBERO benchmark suite and
       retrieve task objects and initial states.
    6. **Extraction Loop** (3 nested levels):
         - Outer: task_id (0..num_tasks-1)
         - Middle: command_level (default, l1, l2, l3)
         - Inner: rollout_idx (0..num_rollouts_per_task-1)
       For each rollout: create environment → run episode → collect embeddings.
    7. **Aggregation**: Mean-pool per-step embeddings within each rollout;
       mean-pool per-rollout embeddings across all rollouts per (task, level).
    8. **Hook Cleanup**: Remove the forward hook from the backbone.
    9. **Serialization**: Save the embedding dict to a `.pkl` file.
    10. **Summary Logging**: Report per-level success rates and avg steps.

    Embedding Aggregation Hierarchy
    --------------------------------
    Three levels of averaging are applied, from finest to coarsest:

    **Level 1 — Step-level** (inside `run_episode_with_embeddings`):
      One embedding is captured per inference step. In full rollout mode,
      these are NOT averaged inside the episode function — they are returned
      as a list and averaged in Level 2.

    **Level 2 — Rollout-level** (in `extract_embeddings_rollout`):
      The per-step embeddings from one rollout are stacked into an array of
      shape `(num_steps, hidden_dim)` and mean-pooled over the step dimension
      to produce one representative vector per rollout:
      `rollout_mean = np.mean(np.stack(episode_embeddings, axis=0), axis=0)`

    **Level 3 — Task-level** (in `extract_embeddings_rollout`):
      The per-rollout mean vectors across all rollouts are stacked into
      `(num_rollouts, hidden_dim)` and mean-pooled to produce one canonical
      embedding per (task, level) combination:
      `mean_embedding = np.mean(np.stack(rollout_embeddings, axis=0), axis=0)`

    In `first_step_only` mode, Level 1 has exactly 1 step, so only
    Levels 2 and 3 apply meaningfully.

    Environment Lifecycle per Rollout
    ----------------------------------
    A fresh `OffScreenRenderEnv` is created and destroyed for each rollout:
      - `get_libero_env()` is called to instantiate the environment with the
        correct BDDL file and camera resolution.
      - `env.seed(seed + rollout_idx)` sets a unique per-rollout MuJoCo seed
        to vary the physics noise while keeping the initial state fixed.
      - `env.close()` is called in the `finally` block to release MuJoCo
        resources regardless of rollout success or failure.

    Parameters
    ----------
    model_path : str
        Absolute path to the TinyVLA checkpoint directory
        (e.g., `/mnt/beegfs/a.cardamone7/models/tinyvla/checkpoint-54000`).
        Must contain the LoRA adapter weights and the action head checkpoint.
    model_base : str
        Absolute path to the base Pythia model directory
        (e.g., `/mnt/beegfs/a.cardamone7/models/pythia/1.3B`).
        Required for loading LoRA adapters on top of the base weights.
    task_suite_name : str, optional
        Name of the LIBERO benchmark task suite to evaluate on.
        Must be a key in `benchmark.get_benchmark_dict()`.
        Supported values: "libero_spatial", "libero_object", "libero_goal",
        "libero_10", "libero_90". Default: "libero_goal".
    command_levels : tuple of str, optional
        Ordered tuple of linguistic variation levels to process for each task.
        Each level must have a corresponding BDDL file (for non-default levels)
        following the naming convention `{base_name}_syn_{level}.bddl`.
        Default: ("default", "l1", "l2", "l3").
    output_dir : str, optional
        Directory where the output `.pkl` file and log file will be saved.
        Created automatically if it does not exist.
        Default: "/mnt/beegfs/a.cardamone7/outputs/embeddings/tinyvla"
    resolution : int, optional
        Camera rendering resolution in pixels (height = width = resolution).
        Passed to `get_libero_env()` for both camera_heights and camera_widths.
        Default: 256.
    seed : int, optional
        Master random seed applied to all RNGs via `set_seed_everywhere()`.
        Also used as the base for per-rollout environment seeds:
        rollout i uses `env.seed(seed + rollout_idx)`. Default: 0.
    num_rollouts_per_task : int, optional
        Number of rollout episodes to run per (task, level) combination.
        More rollouts produce more robust (less noisy) mean embeddings at the
        cost of proportionally more compute time. Default: 10.
    first_step_only : bool, optional
        If True, switches to first-step extraction mode: only the embedding
        at t=0 is captured per rollout (no actions executed). Significantly
        faster than full rollout mode. Default: False.

    Returns
    -------
    all_embeddings : dict
        The complete embedding dictionary. Keys are strings of the form
        "task_{id:02d}_{level}" (e.g., "task_03_l2"). Values are dicts
        containing (see module docstring for full schema):
          - "embedding"             : (hidden_dim,) mean over all rollouts
          - "embedding_per_rollout" : (N_rollouts, hidden_dim) per-rollout means
          - "command_text"          : the exact language command used
          - "success_rate"          : fraction of successful rollouts (full mode only)
          - ... (other metadata fields)
    output_file : str
        Absolute path to the saved `.pkl` file.

    Notes
    -----
    Stats File Convention:
      The `dataset_stats.pkl` file is expected in the PARENT directory of
      `model_path` (one level up from the checkpoint folder). This convention
      follows the TinyVLA training pipeline which saves stats alongside the
      top-level run directory rather than inside individual checkpoints.

    Initial State Cycling:
      If `num_rollouts_per_task > len(initial_states)`, initial states are
      cycled using modular indexing: `initial_states[rollout_idx % len(initial_states)]`.
      This avoids index-out-of-bounds errors while maximizing state diversity.

    Error Resilience:
      Each environment creation and episode execution is wrapped in
      try/except/finally blocks. A single failed rollout is logged and skipped
      without aborting the entire extraction run. The `finally` block ensures
      `env.close()` is always called to prevent MuJoCo resource leaks.

    Output Filename Convention:
      `rollout_embeddings_tinyvla_{suite}_{levels}_{mode}_r{N}.pkl`
      Example: `rollout_embeddings_tinyvla_libero_goal_default_l1_l2_l3_full_r10.pkl`
    """
    global logger   # Write to module-level logger (shared with all helper functions)

    # ── Stage 1: Logging Initialization ────────────────────────────────────
    logger, log_file = setup_logging(output_dir, task_suite_name)

    # Print a visually distinctive header for easy identification in long log files
    logger.info("=" * 80)
    logger.info("TINYVLA EMBEDDING EXTRACTION STARTED")
    logger.info("=" * 80)
    # Log all configuration parameters for full reproducibility tracing
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model base: {model_base}")
    logger.info(f"Task suite: {task_suite_name}")
    logger.info(f"Command levels: {command_levels}")
    logger.info(f"Rollouts per task: {num_rollouts_per_task}")
    logger.info(f"First step only: {first_step_only}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Resolution: {resolution}")
    logger.info("=" * 80)

    # ── Stage 2: Policy Loading ─────────────────────────────────────────────
    # Hardcoded configuration for TinyVLA with LoRA + droid_diffusion action head
    policy_config = {
        "model_path": model_path,
        "model_base": model_base,
        "enable_lora": True,            # TinyVLA always uses LoRA adapters
        "conv_mode": "pythia",          # Conversation template for Pythia backbone
        "action_head": "droid_diffusion",  # Diffusion-based action prediction head
    }

    policy = llava_pythia_act_policy(policy_config)   # Load model checkpoint
    policy.policy.eval()    # Set all submodules to inference mode (disable dropout, BN)
    logger.info("Policy set to eval mode")

    # ── Stage 3: Dataset Stats Loading ─────────────────────────────────────
    # Navigate one directory level up from the checkpoint folder to find stats
    stats_path = os.path.join(
        "/".join(model_path.split('/')[:-1]),   # Parent of checkpoint-XXXXX/
        'dataset_stats.pkl'
    )
    logger.info(f"Loading dataset stats from: {stats_path}")
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)  # Dict with qpos_mean, qpos_std, action_min, action_max
    logger.info("Dataset stats loaded successfully")

    # ── Stage 4: Hook Registration ──────────────────────────────────────────
    emb_capture = EmbeddingCapture()       # Instantiate the hook container
    emb_capture.register(policy.policy)   # Attach hook to GPT-NeoX backbone

    # ── Stage 5: Task Suite Loading ─────────────────────────────────────────
    logger.info(f"Setting seed: {seed}")
    set_seed_everywhere(seed)   # Seed all RNGs for reproducibility

    logger.info(f"Loading task suite: {task_suite_name}")
    benchmark_dict = benchmark.get_benchmark_dict()   # Get dict of all LIBERO task suites
    task_suite = benchmark_dict[task_suite_name]()    # Instantiate the requested suite
    num_tasks = task_suite.n_tasks                    # Total number of tasks in this suite

    # Look up suite-specific max steps; fall back to 300 for unknown suites
    max_steps = TASK_MAX_STEPS.get(task_suite_name, 300)

    logger.info(
        f"Task suite loaded: {num_tasks} tasks, max {max_steps} steps per episode"
    )

    # Compute and log total episode count for progress estimation
    mode_str = "FIRST STEP ONLY" if first_step_only else "FULL ROLLOUT"
    total_episodes = num_tasks * num_rollouts_per_task * len(command_levels)
    logger.info(f"Mode: {mode_str}")
    logger.info(f"Total episodes to run: {total_episodes}")
    logger.info("=" * 80)

    all_embeddings = {}   # Final output dict: "task_XX_level" → embedding record
    episode_counter = 0   # Global counter across all tasks, levels, and rollouts

    # ── Stage 6: Extraction Loop ────────────────────────────────────────────
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)   # Retrieve LIBERO task object by index

        # Get task name; fall back to str(task) if .name attribute is missing
        task_name = getattr(task, 'name', str(task))

        # Load all fixed initial states for this task (for reproducibility)
        initial_states = task_suite.get_task_init_states(task_id)

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"TASK {task_id + 1}/{num_tasks}: {task_name}")
        logger.info(f"Initial states available: {len(initial_states)}")
        logger.info("=" * 80)

        # ── Middle Loop: Command Variation Levels ──────────────────────────
        for level in command_levels:
            # Build the expected BDDL file path for this (task, level) pair
            bddl_file = build_bddl_path(task, level)

            # Skip this level if the BDDL file doesn't exist on disk
            if not os.path.exists(bddl_file):
                logger.warning(
                    f"  {level.upper():8s}: BDDL file not found: {bddl_file}"
                )
                continue

            # Extract the natural language command from the BDDL file
            command = extract_command_from_bddl(bddl_file)
            if command is None:
                logger.warning(
                    f"  {level.upper():8s}: Could not extract command"
                )
                continue

            logger.info(f"\n  COMMAND LEVEL: {level.upper()}")
            logger.info(f"  Command text: '{command}'")
            logger.info(f"  Running {num_rollouts_per_task} rollouts...")

            # Accumulators for this (task, level) pair across all rollouts
            rollout_embeddings = []         # List of per-rollout mean embeddings: [(hidden_dim,), ...]
            rollout_all_embeddings = []     # List of per-rollout step arrays: [(steps, hidden_dim), ...]
            rollout_successes = []          # List of bool: success/failure per rollout
            successes = 0                   # Count of successful rollouts
            total_steps = 0                 # Cumulative inference steps across all rollouts

            # ── Inner Loop: Rollout Episodes ───────────────────────────────
            for rollout_idx in range(num_rollouts_per_task):
                episode_counter += 1   # Increment global episode counter for progress display
                logger.info(
                    f"\n  Rollout {rollout_idx + 1}/{num_rollouts_per_task} "
                    f"(Episode {episode_counter}/{total_episodes})"
                )

                # ── Environment Creation ────────────────────────────────────
                try:
                    logger.debug(f"    Creating environment...")
                    env, task_description, _ = get_libero_env(
                        task, "tiny_vla",
                        # Use variation BDDL only for non-default levels
                        change_command=(level != "default"),
                        command_level=level if level != "default" else None,
                        resolution=resolution,
                    )
                    # Set per-rollout seed: seed+0, seed+1, ... for variation across rollouts
                    env.seed(seed + rollout_idx)
                    logger.debug(
                        f"    Environment created, seed={seed + rollout_idx}"
                    )
                except Exception as e:
                    logger.error(f"    Failed to create environment: {e}")
                    continue   # Skip this rollout if environment creation fails

                # Cycle through initial states to cover more configurations
                init_state = initial_states[rollout_idx % len(initial_states)]
                logger.debug(
                    f"    Using initial state {rollout_idx % len(initial_states)}"
                )

                # ── Episode Execution ────────────────────────────────────────
                try:
                    episode_embeddings, success, num_steps = run_episode_with_embeddings(
                        env=env,
                        task_description=command,   # Use BDDL-extracted command (not task.language)
                        policy=policy,
                        policy_config=policy_config,
                        stats=stats,
                        emb_capture=emb_capture,
                        initial_state=init_state,
                        max_steps=max_steps,
                        num_steps_wait=10,
                        first_step_only=first_step_only,
                    )

                    if episode_embeddings:
                        if first_step_only:
                            # First-step mode: take the single captured embedding directly
                            rollout_embeddings.append(episode_embeddings[0])   # (hidden_dim,)
                            logger.info(f"    ✓ Embedding extracted (1 step)")
                        else:
                            # Full rollout mode: compute rollout-level mean over all steps
                            rollout_emb = np.stack(episode_embeddings, axis=0)
                            # Shape: (num_steps, hidden_dim)

                            rollout_mean = np.mean(rollout_emb, axis=0)
                            # Shape: (hidden_dim,) — mean over step dimension

                            rollout_embeddings.append(rollout_mean)     # Store rollout mean
                            rollout_all_embeddings.append(rollout_emb)  # Store all steps
                            rollout_successes.append(success)           # Store success flag

                            successes += int(success)   # Increment success counter
                            total_steps += num_steps    # Accumulate total inference steps

                            status = "✓ SUCCESS" if success else "✗ FAILURE"
                            logger.info(
                                f"    {status} - {num_steps} steps, "
                                f"{len(episode_embeddings)} embeddings"
                            )
                    else:
                        # No embeddings were captured (e.g., hook failed to fire)
                        logger.warning(f"    No embeddings extracted")

                except Exception as e:
                    # Log error and continue to next rollout without crashing
                    logger.error(f"    Error during rollout: {e}")
                    logger.exception("    Full traceback:")  # Full stack trace at DEBUG level

                finally:
                    # Always close the environment to release MuJoCo/OpenGL resources
                    try:
                        env.close()
                        logger.debug("    Environment closed")
                    except:
                        pass  # Silently ignore close() errors (env may already be invalid)

            # ── Stage 7: Embedding Aggregation (Task-Level) ─────────────────
            if rollout_embeddings:
                # Stack all per-rollout mean embeddings: (N_rollouts, hidden_dim)
                rollout_embeddings_arr = np.stack(rollout_embeddings, axis=0)

                # Compute the task-level canonical embedding: mean over rollout dimension
                mean_embedding = np.mean(rollout_embeddings_arr, axis=0)  # (hidden_dim,)

                # Construct the output dictionary key and record
                key = f"task_{task_id:02d}_{level}"   # e.g., "task_03_l2"
                all_embeddings[key] = {
                    "task_id": task_id,             # 0-based task index
                    "task_name": task_name,         # Task name string from LIBERO
                    "command_level": level,         # "default", "l1", "l2", or "l3"
                    "command_text": command,        # Exact language command used
                    "embedding": mean_embedding,    # (hidden_dim,) — canonical embedding
                    "embedding_per_rollout": rollout_embeddings_arr,  # (N_r, hidden_dim)
                    "num_rollouts": len(rollout_embeddings),          # Actual rollout count
                    "first_step_only": first_step_only,               # Extraction mode flag
                    "model": "tinyvla",             # Model identifier for provenance
                }

                # Append full-rollout-only fields (not meaningful in first_step_only mode)
                if not first_step_only and rollout_all_embeddings:
                    # Concatenate all per-step embedding arrays across all rollouts
                    all_embeddings[key]["embedding_all_steps"] = np.concatenate(
                        rollout_all_embeddings, axis=0
                    )  # (total_steps_across_all_rollouts, hidden_dim)

                    all_embeddings[key]["rollout_successes"] = rollout_successes   # List[bool]
                    all_embeddings[key]["num_successes"] = successes               # int
                    all_embeddings[key]["total_steps"] = total_steps               # int

                    # Success rate: avoid division by zero with max(..., 1)
                    all_embeddings[key]["success_rate"] = (
                        successes / max(len(rollout_embeddings), 1)
                    )

                # Log per-level summary for this task
                if first_step_only:
                    logger.info(
                        f"  Summary: {len(rollout_embeddings)} embeddings, "
                        f"shape: {mean_embedding.shape}"
                    )
                else:
                    success_rate = successes / max(len(rollout_embeddings), 1)
                    logger.info(
                        f"  Summary: {successes}/{num_rollouts_per_task} success "
                        f"({success_rate:.1%}), {total_steps} total steps, "
                        f"embedding shape: {mean_embedding.shape}"
                    )
            else:
                # No embeddings at all for this (task, level) pair
                logger.warning(f"  No embeddings extracted for {level}")

    # ── Stage 8: Hook Cleanup ───────────────────────────────────────────────
    emb_capture.remove()    # Deregister hook from GPT-NeoX backbone
    logger.info("\nEmbedding capture hook removed")

    # ── Stage 9: Serialization ──────────────────────────────────────────────
    logger.info(f"\nSaving results to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Build output filename encoding the key configuration parameters
    mode_suffix = "first_step" if first_step_only else "full"
    output_file = os.path.join(
        output_dir,
        f"rollout_embeddings_tinyvla_"
        f"{task_suite_name}_"
        f"{'_'.join(command_levels)}_"   # e.g., "default_l1_l2_l3"
        f"{mode_suffix}_"                # "first_step" or "full"
        f"r{num_rollouts_per_task}.pkl"  # e.g., "r10"
    )

    logger.info(f"Writing to file: {output_file}")
    with open(output_file, "wb") as f:
        pickle.dump(all_embeddings, f)  # Serialize the entire embedding dict to disk
    logger.info("Results saved successfully")

    # ── Stage 10: Summary Logging ───────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 80)

    if all_embeddings:
        first_key = next(iter(all_embeddings.keys()))   # Retrieve any key for shape inspection
        logger.info(f"Total entries: {len(all_embeddings)}")
        logger.info(
            f"Mean embedding shape: {all_embeddings[first_key]['embedding'].shape}"
        )
        logger.info(
            f"Mode: {'First step only' if first_step_only else 'Full rollout'}"
        )

        if not first_step_only:
            # Report per-level success rates and average steps across all tasks
            logger.info("\nSuccess rates by command level:")
            for level in command_levels:
                # Collect all records for this command level
                level_data = [
                    v for k, v in all_embeddings.items()
                    if v['command_level'] == level
                ]
                if level_data:
                    # Average success rate across all tasks at this level
                    avg_sr = np.mean([d.get('success_rate', 0) for d in level_data])
                    # Average steps per rollout: total_steps / num_rollouts, then average over tasks
                    avg_steps = np.mean([
                        d.get('total_steps', 0) / d.get('num_rollouts', 1)
                        for d in level_data
                    ])
                    logger.info(
                        f"  {level.upper():8s}: {avg_sr:6.1%} success rate, "
                        f"{avg_steps:5.1f} avg steps"
                    )

    # Log final output paths and wall-clock end time for full run traceability
    logger.info(f"\nOutput file: {output_file}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    return all_embeddings, output_file


# ============================================================================
# CLI Entry Point
# ============================================================================


if __name__ == "__main__":
    """
    Command-line interface for `extract_embeddings_rollout.py`.

    Parses command-line arguments and invokes `extract_embeddings_rollout()`
    with the provided configuration. All arguments correspond directly to
    parameters of the main function; see its docstring for full details.

    Required Arguments
    ------------------
    --model_path : str
        Path to the TinyVLA LoRA checkpoint directory.
        Example: /mnt/beegfs/.../checkpoint-54000
    --model_base : str
        Path to the base Pythia model (pre-trained backbone before LoRA).
        Example: /mnt/beegfs/.../pythia-1.3B

    Optional Arguments
    ------------------
    --task_suite : str
        LIBERO benchmark suite name. Default: "libero_goal".
    --command_levels : list of str
        Space-separated command variation levels to process.
        Default: ["default", "l1", "l2", "l3"].
    --output_dir : str
        Output directory for embeddings .pkl and log files.
        Default: "/mnt/beegfs/a.cardamone7/outputs/embeddings/tinyvla"
    --resolution : int
        Camera image resolution. Default: 256.
    --seed : int
        Master random seed. Default: 0.
    --num_rollouts : int
        Number of rollout episodes per (task, level) pair. Default: 10.
    --first_step_only : flag
        If present, captures only the first-step embedding per rollout
        (no full execution). Significantly reduces runtime.

    Usage Examples
    --------------
    Full rollout extraction (10 episodes per task per level):
        python extract_embeddings_rollout.py \\
            --model_path /mnt/beegfs/.../checkpoint-54000 \\
            --model_base /mnt/beegfs/.../pythia-1.3B \\
            --task_suite libero_goal \\
            --command_levels default l1 l2 l3 \\
            --num_rollouts 10 \\
            --output_dir /mnt/beegfs/.../embeddings/tinyvla

    Fast first-step-only extraction (default + l1 only):
        python extract_embeddings_rollout.py \\
            --model_path /mnt/beegfs/.../checkpoint-54000 \\
            --model_base /mnt/beegfs/.../pythia-1.3B \\
            --task_suite libero_goal \\
            --command_levels default l1 \\
            --first_step_only \\
            --num_rollouts 10
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract pre-action-head embeddings during real inference rollouts "
            "from TinyVLA on LIBERO"
        )
    )

    # ── Required arguments ─────────────────────────────────────────────────
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,   # No default: must be specified at runtime
        help="Path to TinyVLA checkpoint (e.g., .../checkpoint-54000)",
    )
    parser.add_argument(
        "--model_base",
        type=str,
        required=True,   # No default: base model path must be provided
        help="Path to base model (e.g., .../1.3B)",
    )

    # ── Optional arguments ─────────────────────────────────────────────────
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_goal",
        # No help text needed: the default clearly indicates the expected format
    )
    parser.add_argument(
        "--command_levels",
        type=str,
        nargs="+",      # Accept one or more space-separated values as a list
        default=["default", "l1", "l2", "l3"],
        help=(
            "Command levels to extract embeddings for. "
            "Example: --command_levels default l1 l2 l3"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/beegfs/a.cardamone7/outputs/embeddings/tinyvla",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,    # 256×256 pixels matches TinyVLA training resolution
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=10,
        help="Number of rollout episodes per task per command level (default: 10)",
    )
    parser.add_argument(
        "--first_step_only",
        action="store_true",   # Boolean flag: True if flag is present, False otherwise
        help=(
            "If set, extract embedding only from the first observation "
            "without executing any actions (faster, no full rollout)"
        ),
    )

    args = parser.parse_args()

    # ── Invoke main extraction function with parsed arguments ───────────────
    extract_embeddings_rollout(
        model_path=args.model_path,
        model_base=args.model_base,
        task_suite_name=args.task_suite,
        command_levels=tuple(args.command_levels),  # Convert list → tuple for consistency
        output_dir=args.output_dir,
        resolution=args.resolution,
        seed=args.seed,
        num_rollouts_per_task=args.num_rollouts,
        first_step_only=args.first_step_only,
    )