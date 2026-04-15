"""
robot_utils.py
==============

Overview
--------
This module provides foundational utility constants and functions for evaluating
Vision-Language-Action (VLA) robot policies — specifically TinyVLA — across
simulation and real-world environments.

It establishes three categories of shared infrastructure used throughout the
entire evaluation pipeline:

1. **Timestamp Constants** (`DATE`, `DATE_TIME`):
   Human-readable strings capturing the wall-clock time at module import.
   Used to construct unique, time-stamped filenames and directory names for
   log files, rollout videos, and trajectory archives across all evaluation
   scripts. By computing these once at import time, all components of a single
   evaluation run share the same consistent timestamp prefix.

2. **Device Constant** (`DEVICE`):
   A PyTorch device object that selects CUDA GPU (cuda:0) when available,
   falling back to CPU otherwise. Centralizing device selection here ensures
   that all model tensors and data are consistently placed on the same device
   without scattered `torch.cuda.is_available()` checks throughout the codebase.

3. **Reproducibility Utility** (`set_seed_everywhere`):
   A comprehensive seed-setting function that initializes the random state of
   every relevant random number generator (Python, NumPy, PyTorch CPU, PyTorch
   CUDA, TensorFlow) simultaneously. This is critical for scientific
   reproducibility: robot policy evaluation results must be exactly reproducible
   across independent runs with the same seed to enable valid statistical
   comparisons and ablation studies.

Design Philosophy
-----------------
This module is intentionally minimal: it contains only shared constants and a
single utility function. It is imported by all evaluation entry-point scripts
(`run_libero_eval.py`, `run_libero_ablation.py`, etc.) and by the lower-level
`libero_utils.py`, making it the shared base layer of the evaluation stack.

The use of module-level constants (rather than function-return values) for
`DATE`, `DATE_TIME`, and `DEVICE` ensures that all scripts sharing an
interpreter process see the same values, preventing timestamp drift within
a single evaluation run.

Dependencies
------------
  - torch       : PyTorch deep learning framework (GPU detection, seeding)
  - numpy       : Scientific computing library (random seeding, print formatting)
  - tensorflow  : TensorFlow deep learning framework (op determinism, seeding)
  - os          : Operating system interface (environment variable setting)
  - random      : Python standard library RNG (seeding)
  - time        : System time access for timestamp generation
  - typing      : PEP 484 type hint utilities (imported for downstream use)
"""

import os       # Used for setting PYTHONHASHSEED environment variable
import random   # Python's built-in random number generator (used in data augmentation, etc.)
import time     # Used to capture the current wall-clock time for timestamp constants
from typing import Any, Dict, List, Optional, Union  # Type hint primitives for annotating functions
import numpy as np      # NumPy for random seeding and print format configuration
import torch            # PyTorch for device detection, manual seeding, and cuDNN configuration
import tensorflow as tf # TensorFlow for op-level determinism and global seed setting


# =============================================================================
# Module-level constants
# =============================================================================

# DATE: A human-readable date string formatted as "YYYY_MM_DD".
# Captured once at module import time (not at function call time), ensuring all
# output paths generated within a single run share the same date prefix.
# Example: "2025_06_01"
DATE = time.strftime("%Y_%m_%d")

# DATE_TIME: A human-readable datetime string formatted as "YYYY_MM_DD-HH_MM_SS".
# More granular than DATE; used to create unique filenames for log files, videos,
# and trajectory archives, preventing collisions when multiple runs occur on
# the same day.
# Example: "2025_06_01-14_32_07"
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

# DEVICE: The PyTorch compute device to use for model inference and tensor operations.
# Automatically selects the first available CUDA GPU (cuda:0) for maximum throughput.
# Falls back to CPU if no CUDA-capable GPU is detected (e.g., on CPU-only dev machines).
# This constant is imported by all evaluation scripts and policy classes to ensure
# consistent tensor placement without redundant device-detection logic.
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# =============================================================================
# NumPy print configuration
# =============================================================================

# Configure NumPy to print all float values with exactly 3 decimal places.
# This affects any numpy array printed via print() or repr(), making logged
# robot state vectors (positions, velocities, actions) consistently readable.
# Without this, NumPy uses scientific notation for very small/large values,
# which is harder to scan in log files.
# Example output: "[0.123 -0.456  0.789]" instead of "[1.23e-01 -4.56e-01  7.89e-01]"
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


# =============================================================================
# Utility functions
# =============================================================================

def set_seed_everywhere(seed: int) -> None:
    """
    Initialize the random state of every random number generator used in the
    evaluation pipeline to a fixed seed value, ensuring fully reproducible runs.

    In robotics policy evaluation, reproducibility is essential:
      - Initial object positions must be the same across runs for fair comparison.
      - Stochastic policy sampling (e.g., diffusion action heads) must produce
        identical action sequences given the same observation and seed.
      - Data augmentation during any fine-tuning must be deterministic.
      - Statistical results (success rates across N episodes) must be exactly
        reproducible from seed and checkpoint alone.

    This function seeds all six sources of randomness that may be active
    in a TinyVLA evaluation run:

    +---------------------------+------------------------------------------+
    | RNG / Framework           | Function called                          |
    +---------------------------+------------------------------------------+
    | PyTorch CPU               | torch.manual_seed(seed)                  |
    | PyTorch CUDA (all GPUs)   | torch.cuda.manual_seed_all(seed)         |
    | NumPy                     | np.random.seed(seed)                     |
    | Python built-in random    | random.seed(seed)                        |
    | Python hash randomization | os.environ["PYTHONHASHSEED"] = str(seed) |
    | TensorFlow                | tf.random.set_seed(seed)                 |
    | TensorFlow op determinism | tf.config.experimental.enable_op_determinism() |
    +---------------------------+------------------------------------------+

    Additionally, PyTorch's cuDNN backend is configured for determinism:
      - `torch.backends.cudnn.deterministic = True` : Forces cuDNN to use
        only deterministic algorithms for convolutions. This may be slower
        but guarantees bit-exact results across runs.
      - `torch.backends.cudnn.benchmark = False`    : Disables cuDNN's
        auto-tuner, which benchmarks multiple convolution algorithms and
        selects the fastest. The auto-tuner is non-deterministic across
        runs (it depends on hardware timing), so it must be disabled.

    Parameters
    ----------
    seed : int
        The integer seed value to use for all RNGs. Commonly set to 42 in
        the default configuration (`GenerateConfig.seed = 42`). The same
        seed is re-applied at the start of each evaluation run via the
        `cfg.seed` field to allow exact reproduction of any reported result.

    Returns
    -------
    None
        This function has no return value; all effects are side effects on
        global RNG state and environment variables.

    Notes
    -----
    PYTHONHASHSEED:
      Python 3.3+ randomizes hash values of strings by default (hash
      randomization) to prevent certain denial-of-service attacks based on
      hash collisions. When reproducibility matters (e.g., any dict or set
      whose iteration order depends on hash values), `PYTHONHASHSEED` must be
      set BEFORE the Python interpreter starts. Setting it here at runtime
      affects only hash computation in the current process after this point;
      for true reproducibility from process start, it should also be set in
      the shell before launching the script:
        `PYTHONHASHSEED=42 python run_libero_eval.py ...`

    TensorFlow Op Determinism:
      `tf.config.experimental.enable_op_determinism()` forces TensorFlow to
      use only deterministic GPU kernels for all operations. Some GPU kernels
      (e.g., certain reduction operations) are non-deterministic by default
      due to floating-point addition reordering in parallel reductions.
      Enabling op determinism may reduce throughput for TF-based model
      components.

    cuDNN Benchmark Mode:
      Setting `benchmark = False` prevents cuDNN from running timing
      benchmarks to select the fastest convolution algorithm. This is
      important not only for reproducibility but also for stability when
      input tensor shapes vary between calls (cuDNN benchmark mode caches
      results by shape, and shape changes trigger re-benchmarking).

    Examples
    --------
    >>> set_seed_everywhere(42)   # Standard seed used in default configuration
    >>> set_seed_everywhere(0)    # Alternative seed for a different evaluation run
    """
    # ── PyTorch CPU random state ───────────────────────────────────────────
    # Seeds the default CPU generator used for torch.rand, torch.randn,
    # torch.randint, and all random tensor operations on CPU tensors.
    torch.manual_seed(seed)

    # ── PyTorch CUDA random state (all GPUs) ──────────────────────────────
    # Seeds the random state for every available CUDA device simultaneously.
    # This covers GPU-side operations such as dropout, random weight init,
    # and diffusion noise sampling on the policy's action head.
    # Safe to call even when no GPU is available (no-op in that case).
    torch.cuda.manual_seed_all(seed)

    # ── NumPy random state ────────────────────────────────────────────────
    # Seeds NumPy's global random module, used for:
    #   - Dataset shuffling during evaluation (if applicable)
    #   - Exponential weight computation in temporal aggregation
    #   - Any numpy-based data augmentation
    np.random.seed(seed)

    # ── Python built-in random state ──────────────────────────────────────
    # Seeds Python's `random` module, used for:
    #   - random.shuffle, random.choice, random.sample calls
    #   - Any library internally using Python's random module
    random.seed(seed)

    # ── cuDNN determinism flags ───────────────────────────────────────────
    # Force cuDNN to use only deterministic convolution algorithms.
    # This may be slower (some non-deterministic algorithms are faster) but
    # guarantees bit-exact outputs for identical inputs across runs.
    torch.backends.cudnn.deterministic = True

    # Disable cuDNN's auto-benchmark: prevents non-deterministic algorithm
    # selection based on runtime hardware timing measurements.
    torch.backends.cudnn.benchmark = False

    # ── Python hash randomization ─────────────────────────────────────────
    # Sets the seed for Python's built-in hash function.
    # Affects dict and set ordering when iterating over string-keyed
    # containers. Must be a string value in os.environ.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # ── TensorFlow op-level determinism ───────────────────────────────────
    # Forces TensorFlow to use only deterministic GPU kernels.
    # Affects TF operations in any model component using TensorFlow
    # (e.g., the data pipeline or TF-based vision encoders).
    tf.config.experimental.enable_op_determinism()

    # ── TensorFlow global random seed ─────────────────────────────────────
    # Sets TensorFlow's global seed, which seeds all TF random ops
    # (tf.random.normal, tf.random.shuffle, etc.) deterministically.
    tf.random.set_seed(seed)