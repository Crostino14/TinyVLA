"""
libero_utils.py
================

Overview
--------
This module provides utility functions for evaluating Vision-Language-Action (VLA)
policies — specifically the TinyVLA model family — inside LIBERO robotic manipulation
simulation environments.

It handles the full evaluation support pipeline:
  - Parsing BDDL task description files to extract natural language task commands
  - Constructing LIBERO off-screen rendering environments with optional command
    variations (linguistic paraphrase levels L1/L2/L3) or ablation BDDL overrides
  - Providing no-op (dummy) actions for physics stabilization at episode start
  - Extracting and preprocessing camera observations (agentview and wrist cameras)
  - Saving episode rollouts as MP4 video files and NumPy trajectory archives
  - Converting quaternion orientations to axis-angle representation for robot control

Context: BDDL Files and Command Variations
-------------------------------------------
LIBERO tasks are defined in BDDL (Behavior Domain Definition Language) files.
Each BDDL file contains a `(:language ...)` field that encodes the natural
language instruction given to the policy. This module supports three types of
instruction sources:

  DEFAULT: The original task language from the standard BDDL file.
           Example: "Put the bowl on the stove"

  L1 / L2 / L3 Variations: Synonym/paraphrase BDDL files generated for
           linguistic robustness evaluation. File naming convention:
           `{base_name}_syn_{level}_v{N}.bddl`
           Example: "Set the bowl on the stove" (L1 paraphrase)

  Ablation: Custom BDDL files with deliberately degraded or shortened commands
           used in keyword-shortcut ablation studies.
           Example: "stove" (single keyword)

This module was originally adapted from the OpenVLA evaluation utilities and
extended to support TinyVLA's dual-camera setup, custom BDDL command injection,
and the ablation study framework.

Dependencies
------------
  - libero          : LIBERO benchmark (environment, BDDL path utilities)
  - imageio         : MP4 video writing for rollout visualization
  - numpy           : Array manipulation for images and trajectories
  - tensorflow      : Imported (available for downstream TF-based models)
  - math            : Trigonometric functions for quaternion conversion
  - os              : Filesystem path operations
  - re              : Regular expressions for BDDL content parsing

Author note
-----------
`quat2axisangle` is directly ported from the robosuite library:
https://github.com/ARISE-Initiative/robosuite
"""

import math     # Used in quat2axisangle for acos, isclose, and sqrt
import os       # Used for path construction, file existence checks, and directory creation

import imageio  # Used for writing rollout frames as MP4 video files
import numpy as np  # Used for array operations: image flipping, trajectory saving, zero vectors
import tensorflow as tf  # Imported for compatibility with TF-based downstream model components
from libero.libero import get_libero_path  # Returns canonical paths to LIBERO data directories
from libero.libero.envs import OffScreenRenderEnv  # LIBERO headless simulation environment class
from models.TinyVLA.test.utils.robot_utils import (
    DATE,       # Current date string (YYYY-MM-DD), used for output directory naming
    DATE_TIME,  # Current datetime string (YYYY-MM-DD_HH-MM-SS), used for unique file naming
)


def extract_command_from_bddl(bddl_file_path):
    """
    Extract the natural language task command from a BDDL task definition file.

    BDDL (Behavior Domain Definition Language) files define robotic manipulation
    tasks for the LIBERO benchmark. Each file may contain a `(:language ...)` field
    that specifies the natural language instruction to be given to the policy.
    This function parses that field and returns the command string.

    This function was introduced to support linguistic variation evaluation
    (L1/L2/L3 command levels) and ablation studies, where custom BDDL files
    encode modified or degraded task instructions that differ from the default
    `task.language` attribute provided by the LIBERO task object.

    Parameters
    ----------
    bddl_file_path : str
        Absolute or relative path to the BDDL file to parse.

    Returns
    -------
    str or None
        The extracted task command string if the `(:language ...)` field is found
        and the file is accessible. Returns None if:
          - The file does not exist at `bddl_file_path`
          - The file exists but contains no `(:language ...)` field
          - An exception occurs while reading or parsing the file

    Notes
    -----
    The regex pattern `r'\(:language\s+([^)]+)\)'` works as follows:
      - `\(:language`   : Literal match for "(:language"
      - `\s+`           : One or more whitespace characters (space or newline)
      - `([^)]+)`       : Capture group: one or more characters that are NOT `)`
                          This captures the full command, including multi-word phrases
      - `\)`            : Literal closing parenthesis

    The `.strip()` on the matched group removes any trailing whitespace or
    newline characters that may have been captured by the `[^)]+` pattern.

    Examples
    --------
    Given a BDDL file containing:
        (:language Open the middle layer of the drawer)
    This function returns:
        "Open the middle layer of the drawer"

    Given a BDDL file containing:
        (:language stove)
    This function returns:
        "stove"  (used in ablation keyword-shortcut tests)
    """
    import re  # Imported here (lazy import) since this function may be called infrequently

    # Guard: return None immediately if the file does not exist at the given path
    if not os.path.exists(bddl_file_path):
        return None

    try:
        with open(bddl_file_path, 'r') as f:
            content = f.read()  # Read the full file content as a single string

        # Regex pattern that captures the text inside (:language ...) in the BDDL file
        # [^)]+ matches any characters except ')' to avoid over-capturing nested parens
        pattern = r'\(:language\s+([^)]+)\)'

        match = re.search(pattern, content)  # Search anywhere in the file content
        if match:
            # group(1) returns the first capture group: the command text
            command = match.group(1).strip()  # Strip trailing whitespace/newlines
            return command

        # Pattern not found in file: file is valid but has no (:language ...) field
        return None

    except Exception as e:
        # Catch all exceptions (IOError, PermissionError, UnicodeDecodeError, etc.)
        # and print a warning rather than crashing the evaluation pipeline
        print(f"Warning: Could not parse BDDL file {bddl_file_path}: {e}")
        return None


def get_libero_env(
    task,
    model_family,
    change_command=False,
    command_level=None,
    ablation_bddl_file=None,
    resolution=256
):
    """
    Initialize a LIBERO off-screen rendering environment for a given task,
    with optional support for linguistic command variations and ablation BDDL overrides.

    This function is the main environment factory for the TinyVLA evaluation pipeline.
    It constructs the LIBERO simulation environment from a BDDL file and determines
    which natural language command to pass to the policy. Three command sources are
    supported with the following priority order:

    **Priority 1 — Ablation BDDL** (highest priority):
      If `ablation_bddl_file` is provided, that exact BDDL file is used.
      The command is extracted from its `(:language ...)` field.
      Used for keyword-shortcut ablation studies.

    **Priority 2 — Linguistic Variation BDDL** (if `change_command=True`):
      Looks for files named `{base_name}_syn_{command_level}_v{N}.bddl`
      in the task's BDDL directory. If multiple version files exist, the
      highest version number is selected. Falls back to `{base_name}_syn_{level}.bddl`
      (no version suffix) if no versioned file is found.

    **Priority 3 — Default BDDL** (lowest priority / fallback):
      Uses the standard BDDL file registered with the task. The command is
      extracted from its `(:language ...)` field, with a further fallback to
      `task.language` if extraction fails.

    Parameters
    ----------
    task : libero.libero.benchmark.Task
        A LIBERO task object providing:
          - `task.problem_folder` : Subdirectory under the BDDL files root
          - `task.bddl_file`      : Default BDDL filename for this task
          - `task.language`       : Default natural language command (fallback)
    model_family : str
        Model family identifier (e.g., "tiny_vla"). Currently unused inside
        this function but passed through for compatibility with callers that
        use the same signature for different model families.
    change_command : bool, optional
        If True, attempt to load a linguistic variation BDDL file.
        Only has effect if `ablation_bddl_file` is None. Default: False.
    command_level : str or None, optional
        Variation level to load when `change_command=True`.
        Expected values: 'l1', 'l2', 'l3'. Default: None.
    ablation_bddl_file : str or None, optional
        Explicit BDDL filename (not full path) for ablation override.
        Example: "turn_on_the_stove_ablation_stove1.bddl".
        Must reside in the same `task.problem_folder` directory.
        If provided, overrides all other command sources. Default: None.
    resolution : int, optional
        Camera image resolution in pixels (both height and width).
        Applied to both `camera_heights` and `camera_widths`. Default: 256.

    Returns
    -------
    env : OffScreenRenderEnv
        Initialized and seeded LIBERO off-screen rendering environment.
    task_description : str
        The natural language command that will be passed to the policy.
        This is the ablation command, the variation command, or the original
        command, depending on which BDDL source was successfully loaded.
    original_description : str
        The original natural language command from the default BDDL file
        (before any variation or ablation override). Used for logging and
        for looking up initial states keyed by the canonical task name.

    Notes
    -----
    Environment Seeding:
      `env.seed(0)` is called after construction. This is critical: the LIBERO
      environment's random seed affects object placement during `env.reset()`,
      even when fixed initial states are provided via `env.set_init_state()`.
      Always seeding to 0 ensures reproducible object configurations across runs.

    Version Selection for Linguistic Variations:
      When multiple version files exist (e.g., `_v1`, `_v2`, `_v3`), the file
      with the highest version number is selected. This allows incremental
      improvement of variation files without breaking existing evaluations.
    """
    # ── Step 1: Resolve the default BDDL file path ─────────────────────────
    # get_libero_path("bddl_files") returns the root directory for all BDDL files
    # task.problem_folder is the subdirectory for this task's benchmark suite
    # task.bddl_file is the filename of the default BDDL for this task
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )

    # ── Step 2: Extract the original command from the default BDDL file ────
    original_description = extract_command_from_bddl(task_bddl_file)

    # Fallback: use task.language if BDDL parsing fails
    if original_description is None:
        original_description = task.language
        print(
            f"Warning: Could not extract command from BDDL, "
            f"using task.language: {original_description}"
        )

    # Initialize task_description to the original; may be overridden below
    task_description = original_description

    # ── Step 3: Determine the active BDDL file and command ─────────────────
    # Priority 1: Explicit ablation BDDL file (highest priority)
    if ablation_bddl_file is not None:
        # Construct the full path by combining the task's BDDL directory
        # with the explicitly provided ablation filename
        new_task_bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            ablation_bddl_file
        )

        if os.path.exists(new_task_bddl_file):
            print(f"✓ Using ablation BDDL: {new_task_bddl_file}")
            task_bddl_file = new_task_bddl_file  # Override active BDDL file

            # Extract the degraded command from the ablation BDDL's (:language ...) field
            ablation_description = extract_command_from_bddl(task_bddl_file)
            if ablation_description:
                task_description = ablation_description  # Override policy command
                print(f"✓ Ablation command from BDDL: '{task_description}'")
            else:
                print(f"Warning: Could not extract command from ablation BDDL")
        else:
            # Ablation file requested but not found: log error and fall back to default
            print(f"ERROR: Ablation BDDL file not found: {new_task_bddl_file}")
            print(f"Falling back to default BDDL file: {task_bddl_file}")

    elif change_command and command_level is not None:
        # ── Priority 2: Linguistic variation BDDL file ─────────────────────
        # Remove the ".bddl" extension to construct the variation filename stem
        base_name = task.bddl_file.replace('.bddl', '')

        # Resolve the directory containing this task's BDDL files
        bddl_folder = os.path.join(get_libero_path("bddl_files"), task.problem_folder)

        # Build the expected filename pattern: "{base_name}_syn_{level}_v" (prefix match)
        pattern = f"{base_name}_syn_{command_level}_v"

        # Scan the BDDL folder for files matching this prefix (case-insensitive)
        matching_files = []
        try:
            for filename in os.listdir(bddl_folder):
                # Check if the pattern appears anywhere in the filename (case-insensitive)
                # and that the file has a .bddl extension
                if pattern.lower() in filename.lower() and filename.endswith('.bddl'):
                    matching_files.append(filename)
        except Exception as e:
            print(f"Warning: Could not list files in {bddl_folder}: {e}")

        if matching_files:
            # Select the highest-versioned file among all matches
            def extract_version(filename):
                """Extract the integer version number from a filename containing '_vN'."""
                import re
                match = re.search(r'_v(\d+)', filename, re.IGNORECASE)
                return int(match.group(1)) if match else 0  # Default to 0 if no version found

            # Sort in descending order by version number; first element is highest version
            matching_files.sort(key=extract_version, reverse=True)
            selected_file = matching_files[0]  # Pick the highest version
            new_task_bddl_file = os.path.join(bddl_folder, selected_file)

            print(f"✓ Found {len(matching_files)} matching file(s) for {command_level}")
            print(f"✓ Using version file: {selected_file}")
            task_bddl_file = new_task_bddl_file  # Override active BDDL file

            # Extract the variation command from the selected BDDL file
            variation_description = extract_command_from_bddl(task_bddl_file)
            if variation_description:
                task_description = variation_description  # Override policy command
                print(f"✓ Command loaded: '{task_description}'")
            else:
                print(
                    f"Warning: Could not extract variation command from {selected_file}"
                )
        else:
            # Fallback: try an unversioned variation file (no _vN suffix)
            new_bddl_filename = f"{base_name}_syn_{command_level}.bddl"
            new_task_bddl_file = os.path.join(bddl_folder, new_bddl_filename)

            if os.path.exists(new_task_bddl_file):
                print(f"✓ Using custom BDDL (no version): {new_bddl_filename}")
                task_bddl_file = new_task_bddl_file  # Override active BDDL file
                variation_description = extract_command_from_bddl(task_bddl_file)
                if variation_description:
                    task_description = variation_description  # Override policy command
            else:
                # No variation file found at all: log warning and keep default BDDL
                print(f"WARNING: No custom BDDL files found for {command_level}")
                print(f"Falling back to default BDDL file: {task_bddl_file}")

    # ── Step 4: Build environment arguments and instantiate environment ────
    env_args = {
        "bddl_file_name": task_bddl_file,    # BDDL file defining the scene and goal
        "camera_heights": resolution,          # Render height in pixels
        "camera_widths": resolution,           # Render width in pixels
    }
    env = OffScreenRenderEnv(**env_args)  # Headless MuJoCo-based simulation environment

    # IMPORTANT: Seeding to 0 ensures reproducible object placements.
    # The seed affects MuJoCo's random state, which influences object
    # positions during env.reset() even when fixed initial states are provided.
    env.seed(0)

    return env, task_description, original_description


def get_libero_dummy_action(model_family: str):
    """
    Return a no-op (dummy) action for the LIBERO robot.

    This action is used during the physics stabilization phase at the start
    of each episode, before the policy begins generating real actions.
    During stabilization, the robot holds its current position while the
    simulation physics settle (e.g., objects stop vibrating from initial
    placement perturbations).

    The action format is a 7-dimensional list matching the LIBERO robot's
    action space:
      [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]

    All translation and rotation deltas are zero (no movement). The gripper
    value is set to -1, which corresponds to the gripper-open command in
    LIBERO's normalized action space.

    Parameters
    ----------
    model_family : str
        Model family identifier (e.g., "tiny_vla", "openvla"). Currently
        unused but included for API compatibility with evaluation scripts
        that call this function uniformly across model families.

    Returns
    -------
    list of int/float
        A 7-element list: [0, 0, 0, 0, 0, 0, -1]
          - Indices 0-2 : XYZ end-effector position delta = 0 (no translation)
          - Indices 3-5 : Rotation delta in Euler angles = 0 (no rotation)
          - Index 6     : Gripper command = -1 (open gripper)
    """
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """
    Extract and preprocess the third-person agentview camera image from a LIBERO observation.

    The agentview camera is mounted at a fixed position in the scene and provides
    a workspace-level perspective of the robot and objects. This is the primary
    visual input to the TinyVLA policy.

    IMPORTANT — Image Orientation:
      LIBERO's MuJoCo renderer returns images in a vertically and horizontally
      flipped orientation (MuJoCo renders with the Y-axis inverted relative to
      standard image conventions). Both axes must be flipped to restore the
      correct upright orientation that matches the training data preprocessing.

      The double flip `[::-1, ::-1]` reverses both rows (vertical axis) and
      columns (horizontal axis), equivalent to a 180-degree rotation.
      This is not a transpose — it preserves the channel dimension unchanged.

    Parameters
    ----------
    obs : dict
        LIBERO environment observation dictionary. Must contain the key
        "agentview_image" with a numpy array of shape (H, W, 3) in uint8
        RGB format, where H and W are the camera resolution.

    Returns
    -------
    numpy.ndarray
        Preprocessed image of shape (H, W, 3), dtype uint8, with both
        spatial axes flipped to correct the MuJoCo rendering orientation.
    """
    img = obs["agentview_image"]     # Extract raw agentview frame from observation dict
    img = img[::-1, ::-1]            # Flip both height (rows) and width (cols) axes: 180° rotation
    return img


def get_libero_wrist_image(obs):
    """
    Extract and preprocess the wrist camera image from a LIBERO observation.

    The wrist camera (robot0_eye_in_hand) is mounted on the robot's end-effector
    and provides a close-up view of the gripper-object interaction. TinyVLA uses
    this as a second visual input alongside the agentview image, giving the policy
    both global scene context and local manipulation feedback.

    IMPORTANT — Image Orientation:
      Like the agentview image, LIBERO's MuJoCo renderer returns the wrist camera
      image in a flipped orientation. The same `[::-1, ::-1]` 180-degree rotation
      is applied to restore correct orientation matching the training data.

    Parameters
    ----------
    obs : dict
        LIBERO environment observation dictionary. Must contain the key
        "robot0_eye_in_hand_image" with a numpy array of shape (H, W, 3)
        in uint8 RGB format, where H and W are the camera resolution.

    Returns
    -------
    numpy.ndarray
        Preprocessed wrist camera image of shape (H, W, 3), dtype uint8,
        with both spatial axes flipped to correct the MuJoCo rendering orientation.
    """
    img = obs["robot0_eye_in_hand_image"]  # Extract raw wrist camera frame from obs dict
    img = img[::-1, ::-1]                  # Flip both axes: 180° rotation to match train data
    return img


def save_rollout_video(
    rollout_traj,
    idx,
    success,
    task_description,
    log_file=None,
    dataset_name="libero",
    run=0,
    change_command=False,
    command_level=None
):
    """
    Save a completed episode rollout as an MP4 video and a NumPy trajectory archive.

    This function writes two output files for each episode:
      1. An MP4 video at 30 FPS showing the sequence of agentview frames.
      2. A `.npy` file containing the full rollout trajectory dictionary
         (images, states, actions, and task command).

    Output files are organized into a directory structure that reflects the
    evaluation configuration (default vs. linguistic variation, run number).

    Directory structure:
    ::

        /home/A.CARDAMONE7/outputs/rollouts/libero_goal/
        └── syntactic_variation/tinyvla/checkpoint_54000/
            ├── default/run_0/
            │   └── 2025-06-01_12-00-00--episode=1--success=True--task=put_the_bowl....mp4
            └── {level}_variations_test/{level}/run_0/
                └── 2025-06-01_12-00-00--episode=1--success=False--task=....mp4

    Parameters
    ----------
    rollout_traj : dict
        Trajectory dictionary collected during the episode. Expected keys:
          - "images"       : list of numpy arrays (H, W, 3) — agentview frames
          - "task_command" : str — the language command used during the episode
          - "states"       : list of numpy arrays — robot proprioceptive states
          - "actions"      : list of numpy arrays — actions executed at each step
    idx : int
        1-based episode index within the current task evaluation run.
        Used in the filename to identify which trial this video corresponds to.
    success : bool
        Whether the episode ended in task success. Encoded in the filename
        as "success=True" or "success=False" for easy filtering.
    task_description : str
        Natural language task command used during the episode. Embedded in
        the filename after sanitization (lowercased, spaces → underscores,
        periods and newlines → underscores, truncated to 50 characters).
    log_file : file object or None, optional
        Open text file object for appending log messages. If provided, the
        MP4 save path is written to the log. Default: None.
    dataset_name : str, optional
        Name of the benchmark dataset (e.g., "libero"). Currently unused in
        path construction but retained for API compatibility. Default: "libero".
    run : int, optional
        Run version number (e.g., 0, 1, 2 for seed replication). Used as
        the final subdirectory in the output path. Default: 0.
    change_command : bool, optional
        If True, saves to the linguistic variation subdirectory tree.
        If False, saves to the default subdirectory. Default: False.
    command_level : str or None, optional
        Variation level ("l1", "l2", "l3") used to construct the output
        subdirectory path when `change_command=True`. Default: None.

    Returns
    -------
    str
        The full absolute path to the saved MP4 video file.

    Notes
    -----
    - The `.npy` file is saved at the same path as the MP4, with `.mp4`
      replaced by `.npy`. This archive can be loaded with `np.load(path,
      allow_pickle=True).item()` to recover the full trajectory dict.
    - `imageio.get_writer` opens an MP4 writer with H.264 encoding at 30 FPS.
      Each frame is appended individually; the writer must be explicitly closed
      to flush and finalize the file.
    - Task description sanitization is performed before filename construction
      to ensure filesystem compatibility across all operating systems.
    """
    # ── Step 1: Determine the output directory based on evaluation mode ────
    if change_command and command_level:
        # Linguistic variation mode: directory encodes the variation level
        rollout_dir = (
            f"/home/A.CARDAMONE7/outputs/rollouts/libero_goal/"
            f"syntactic_variation/tinyvla/checkpoint_54000_"
            f"{command_level}_variations_test/{command_level}/run_{run}"
        )
    else:
        # Default (no variation) mode
        rollout_dir = (
            f"/home/A.CARDAMONE7/outputs/rollouts/libero_goal/"
            f"syntactic_variation/tinyvla/checkpoint_54000/default/run_{run}"
        )

    # Create the output directory and all intermediate parents if they don't exist
    os.makedirs(rollout_dir, exist_ok=True)

    # ── Step 2: Sanitize the task description for use in a filename ────────
    processed_task_description = (
        task_description
        .lower()              # Normalize to lowercase
        .replace(" ", "_")    # Replace spaces with underscores
        .replace("\n", "_")   # Replace newlines (rare but possible) with underscores
        .replace(".", "_")    # Replace periods with underscores
        [:50]                 # Truncate to 50 characters to avoid overly long filenames
    )

    # ── Step 3: Build the full MP4 output path ─────────────────────────────
    # DATE_TIME provides a unique timestamp prefix to prevent filename collisions
    mp4_path = (
        f"{rollout_dir}/{DATE_TIME}"
        f"--episode={idx}"
        f"--success={success}"
        f"--task={processed_task_description}.mp4"
    )

    # ── Step 4: Write the MP4 video file ───────────────────────────────────
    video_writer = imageio.get_writer(mp4_path, fps=30)  # H.264 MP4 at 30 frames per second
    # NOTE: TinyVLA stores frames under the key 'images' (plural), unlike
    # some other pipelines that use 'image' (singular)
    for img in rollout_traj['images']:
        video_writer.append_data(img)  # Append each (H, W, 3) uint8 frame to the video
    video_writer.close()  # Flush and finalize the video file (MUST be called)

    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        # Also write the path to the evaluation log file for traceability
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")

    # ── Step 5: Save the full trajectory dictionary as a NumPy archive ─────
    # Replace .mp4 extension with .npy for the companion trajectory file
    npy_path = mp4_path.replace('.mp4', '.npy')
    # allow_pickle=True required when loading, since rollout_traj is a dict
    np.save(npy_path, rollout_traj)
    print(f"Saved replay trajectory at path {npy_path}")

    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: [https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55](https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55)

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den