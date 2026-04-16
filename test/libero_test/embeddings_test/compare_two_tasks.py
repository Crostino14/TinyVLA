"""
compare_two_tasks.py  (TinyVLA version)

Confronto diretto tra embedding di DUE task specificati dall'utente via argparse.
Per ogni task si può specificare il nome e il livello di variazione linguistica
(default, l1, l2, l3).

Calcola:
  - Embedding POST-fusione: mean-pool degli hidden states dall'ultimo layer del GPT-NeoX,
    dopo che testo e visione sono stati fusi tramite il transformer.
  - Embedding PRE-fusione (--pre_fusion): mean-pool dei token testuali dalla lookup table
    embed_in(), PRIMA del forward pass. Puramente linguistico.

Uso:
    python compare_two_tasks.py \\
        --task_a put_the_bowl_on_the_stove \\
        --task_b put_the_bowl_on_the_stove --level_b l1

    python compare_two_tasks.py \\
        --task_a put_the_bowl_on_the_stove \\
        --task_b put_the_cream_cheese_in_the_bowl \\
        --scene_task put_the_bowl_on_the_stove --pre_fusion

    python compare_two_tasks.py --list_tasks
"""

import argparse
import os
import re
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from torchvision import transforms

# ─────────────────── path setup ───────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent                     # test/libero_test
TINYVLA_ROOT = SCRIPT_DIR.parent.parent                            # TinyVLA/
LIBERO_ROOT  = TINYVLA_ROOT.parent.parent / "LIBERO"              # robosuite_test/LIBERO

for p in [str(TINYVLA_ROOT), str(SCRIPT_DIR), str(LIBERO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig
from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava_pythia.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
)
from llava_pythia.conversation import conv_templates
from llava_pythia.model import *  # noqa: F401,F403

from libero.libero import benchmark
from models.TinyVLA.test.utils.libero_utils import (
    get_libero_env,
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
)

# ─────────────────── constants ───────────────────
CHECKPOINT_PATH = (
    "/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/"
    "checkpoints_saving_folder/tinyvla/"
    "post_processed_tiny_vla_llava_pythia_lora_libero_goal_no_noops_lora_r_64_processed/"
    "checkpoint-54000"
)
MODEL_BASE = (
    "/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/"
    "checkpoints_saving_folder/tinyvla/"
    "parte2_llava_pythia_libero_goal_no_noops_64/1.3B"
)

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

TASK_NAMES = [
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "turn_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
    "put_the_cream_cheese_in_the_bowl",
    "open_the_middle_drawer_of_the_cabinet",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_bowl_on_the_stove",
]

TASK_INDEX = {name: i for i, name in enumerate(TASK_NAMES)}
VALID_LEVELS = ["default", "l1", "l2", "l3"]

NUM_STEPS_WAIT = 10
ENV_IMG_RES = 256
MODEL_FAMILY = "tiny_vla"
CONV_MODE = "pythia"


# ─────────────────── BDDL command reader ───────────────────
def read_bddl_command(task_name: str, level: str = "default") -> str:
    bddl_dir = LIBERO_ROOT / "libero" / "libero" / "bddl_files" / "libero_goal"
    suffix = "" if level == "default" else f"_syn_{level}"
    bddl_path = bddl_dir / f"{task_name}{suffix}.bddl"
    if not bddl_path.exists():
        raise FileNotFoundError(f"File BDDL non trovato: {bddl_path}")
    text = bddl_path.read_text()
    m = re.search(r'\(:language\s+([^)]+)\)', text)
    if not m:
        raise ValueError(f"Campo :language non trovato in {bddl_path}")
    return m.group(1).strip()


# ─────────────────── model loading ───────────────────
def load_model(checkpoint_path: str, model_base: str):
    model_name = get_model_name_from_path(checkpoint_path)
    print(f"\nCaricamento TinyVLA da:\n  checkpoint: {checkpoint_path}\n  base: {model_base}")

    tokenizer, policy, image_processor, context_len = load_pretrained_model(
        checkpoint_path, model_base, model_name, False, False
    )
    policy.eval()
    hidden_size = policy.config.hidden_size
    print(f"✓ Modello caricato (hidden_size={hidden_size})")
    print(f"  visual_concat = {policy.config.concat}")
    print(f"  action_head   = {policy.config.action_head_type}")

    return tokenizer, policy, image_processor


# ─────────────────── image preprocessing ───────────────────
_TO_TENSOR = transforms.ToTensor()
_RAND_CROP_RATIO = 0.95


def preprocess_image_pair(
    img_np: np.ndarray,
    wrist_np: np.ndarray,
    image_processor,
    policy,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replica la pipeline di eval_libero.py per droid_diffusion:
      1. Resize → (320, 180)
      2. to_tensor → [0,1]
      3. rand_crop_resize 95%
      4. expand2square (pad con CLIP mean bg)
      5. image_processor.preprocess(normalize, no rescale)
    """
    def _to_tensor_and_crop(np_img: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(np_img, (320, 180))
        t = _TO_TENSOR(resized).float()
        orig_h, orig_w = t.shape[-2], t.shape[-1]
        ratio = _RAND_CROP_RATIO
        t_crop = t[
            :,
            int(orig_h * (1 - ratio) / 2): int(orig_h * (1 + ratio) / 2),
            int(orig_w * (1 - ratio) / 2): int(orig_w * (1 + ratio) / 2),
        ]
        resize_tf = transforms.Resize((orig_h, orig_w), antialias=True)
        return resize_tf(t_crop)

    bg_color = tuple(float(x) for x in image_processor.image_mean)

    def _expand2square(t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(0)
        _, c, h, w = t.shape
        max_dim = max(h, w)
        expanded = np.full((1, max_dim, max_dim, c), bg_color, dtype=np.float32)
        if h == w:
            expanded = t.permute(0, 2, 3, 1).cpu().numpy()
        elif h > w:
            offset = (max_dim - w) // 2
            expanded[:, :h, offset:offset + w, :] = t.permute(0, 2, 3, 1).cpu().numpy()
        else:
            offset = (max_dim - h) // 2
            expanded[:, offset:offset + h, :w, :] = t.permute(0, 2, 3, 1).cpu().numpy()
        return torch.tensor(expanded, dtype=t.dtype)

    def _preprocess(sq: torch.Tensor) -> torch.Tensor:
        pv = image_processor.preprocess(
            sq, return_tensors="pt",
            do_normalize=True, do_rescale=False, do_center_crop=False,
        )["pixel_values"]
        return pv.to(policy.device, dtype=policy.dtype)

    img_t = _to_tensor_and_crop(img_np)
    wrist_t = _to_tensor_and_crop(wrist_np)

    img_sq = _expand2square(img_t)
    wrist_sq = _expand2square(wrist_t)

    return _preprocess(img_sq), _preprocess(wrist_sq)


# ─────────────────── prompt building ───────────────────
def build_input_ids(tokenizer, policy, command: str):
    """
    Costruisce input_ids e attention_mask nel formato conversazionale Pythia:
      "A chat between... USER: <image>\n{command} ASSISTANT: <|endoftext|>"
    """
    conv = conv_templates[CONV_MODE].copy()

    if policy.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + command
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + command

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " <|endoftext|>"

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(DEVICE)

    attn_mask = input_ids.ne(tokenizer.pad_token_id)
    return input_ids, attn_mask, prompt


# ─────────────────── embedding extraction ───────────────────
def extract_embedding(policy, tokenizer, image_processor,
                      command: str,
                      img_np: np.ndarray, wrist_np: np.ndarray) -> np.ndarray:
    """
    Embedding POST-fusione: forward pass completo attraverso GPT-NeoX,
    mean-pool degli hidden states dell'ultimo layer (tutti i token).
    """
    with torch.no_grad():
        image_tensor, image_tensor_r = preprocess_image_pair(
            img_np, wrist_np, image_processor, policy
        )
        input_ids, attn_mask, _ = build_input_ids(tokenizer, policy, command)

        # Dummy robot state (zeri) — serve solo per il forward ma non influenza gli embedding
        states = torch.zeros(1, 8, device=policy.device, dtype=policy.dtype)

        # Forward con output_hidden_states
        # prepare_inputs_labels_for_multimodal fonde immagini + testo
        (input_ids_mm, attn_mask_mm, past_kv, inputs_embeds, _) = \
            policy.prepare_inputs_labels_for_multimodal(
                input_ids, attn_mask, None, None,
                image_tensor,
                images_r=image_tensor_r,
                visual_concat=policy.visual_concat,
                states=states,
            )

        outputs = policy.get_model()(
            input_ids=input_ids_mm,
            attention_mask=attn_mask_mm,
            past_key_values=past_kv,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_size)
        # Mean pool su tutti i token
        pooled = last_hidden.mean(dim=1)  # (1, hidden_size)

    return pooled.squeeze(0).detach().cpu().float().numpy()


def extract_prefusion_embedding(policy, tokenizer,
                                command: str) -> np.ndarray:
    """
    Embedding PRE-fusione: mean-pool dei token testuali dalla lookup table embed_in(),
    PRIMA del forward pass. Puramente linguistico, nessuna influenza visiva.

    Nota: esclude il token IMAGE_TOKEN_INDEX (-200) che è un placeholder per le immagini.
    """
    with torch.no_grad():
        input_ids, attn_mask, _ = build_input_ids(tokenizer, policy, command)

        # Filtra il token immagine placeholder (IMAGE_TOKEN_INDEX = -200)
        ids = input_ids[0]  # (seq_len,)
        text_mask = ids != IMAGE_TOKEN_INDEX
        text_ids = ids[text_mask].unsqueeze(0)  # (1, text_len)
        text_attn = attn_mask[0][text_mask].unsqueeze(0)

        # Lookup table embedding
        text_embeds = policy.get_model().embed_in(text_ids)  # (1, text_len, hidden_size)

        mask = text_attn.unsqueeze(-1).to(text_embeds.dtype)
        pooled = (text_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    return pooled.squeeze(0).detach().cpu().float().numpy()


# ─────────────────── first frame helper ───────────────────
def get_first_frame(task, task_id: int, task_suite, seed: int = 0):
    env, _, _ = get_libero_env(task, MODEL_FAMILY, change_command=False, resolution=ENV_IMG_RES)
    env.seed(seed)

    try:
        initial_states = task_suite.get_task_init_states(task_id)
        env.reset()
        obs = env.set_init_state(initial_states[0])
    except Exception:
        env.reset()
        obs = env.get_observation()

    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action(MODEL_FAMILY))

    img_np = get_libero_image(obs)
    wrist_np = get_libero_wrist_image(obs)
    env.close()

    return img_np, wrist_np


# ─────────────────── distance functions ───────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-12)
    b_n = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_n, b_n))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ─────────────────── main ───────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Confronta embedding di due task LIBERO Goal per TinyVLA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Task disponibili:
{chr(10).join(f'  {i:2d}. {name}' for i, name in enumerate(TASK_NAMES))}

Livelli di variazione: default, l1, l2, l3

Esempi:
  # Confronta task originale con la sua variazione L1
  python compare_two_tasks.py \\
      --task_a put_the_bowl_on_the_stove \\
      --task_b put_the_bowl_on_the_stove --level_b l1

  # Confronta due task diversi sulla stessa scena
  python compare_two_tasks.py \\
      --task_a put_the_bowl_on_the_stove \\
      --task_b put_the_cream_cheese_in_the_bowl \\
      --scene_task put_the_bowl_on_the_stove --pre_fusion
""",
    )
    parser.add_argument("--list_tasks", action="store_true",
                        help="Elenca i task disponibili ed esci.")
    parser.add_argument("--task_a", type=str,
                        help="Nome del primo task.")
    parser.add_argument("--level_a", type=str, default="default",
                        choices=VALID_LEVELS,
                        help="Livello di variazione per il primo task (default: default).")
    parser.add_argument("--task_b", type=str,
                        help="Nome del secondo task.")
    parser.add_argument("--level_b", type=str, default="default",
                        choices=VALID_LEVELS,
                        help="Livello di variazione per il secondo task (default: default).")
    parser.add_argument("--scene_task", type=str, default=None,
                        help="Task da cui prendere la scena visiva per ENTRAMBI i comandi.")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                        help="Percorso al checkpoint del modello.")
    parser.add_argument("--model_base", type=str, default=MODEL_BASE,
                        help="Percorso al modello base (Pythia 1.3B).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed per la simulazione (default: 0).")
    parser.add_argument("--pre_fusion", action="store_true",
                        help="Calcola ANCHE gli embedding pre-fusione (puramente testuali).")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Elenca task disponibili ──
    if args.list_tasks:
        print("\nTask disponibili (LIBERO Goal):")
        print("─" * 60)
        bddl_dir = LIBERO_ROOT / "libero" / "libero" / "bddl_files" / "libero_goal"
        for i, name in enumerate(TASK_NAMES):
            levels = ["default"]
            for lvl in ["l1", "l2", "l3"]:
                if (bddl_dir / f"{name}_syn_{lvl}.bddl").exists():
                    levels.append(lvl)
            print(f"  {i:2d}. {name}")
            print(f"      Livelli: {', '.join(levels)}")
        return

    # ── Validazione argomenti ──
    if not args.task_a or not args.task_b:
        print("Errore: specificare --task_a e --task_b. Usa --list_tasks per vedere i task.")
        sys.exit(1)
    for label, val in [("task_a", args.task_a), ("task_b", args.task_b)]:
        if val not in TASK_INDEX:
            print(f"Errore: {label} '{val}' non trovato. Usa --list_tasks.")
            sys.exit(1)
    if args.scene_task and args.scene_task not in TASK_INDEX:
        print(f"Errore: scene_task '{args.scene_task}' non trovato. Usa --list_tasks.")
        sys.exit(1)

    # ── Lettura comandi BDDL ──
    cmd_a = read_bddl_command(args.task_a, args.level_a)
    cmd_b = read_bddl_command(args.task_b, args.level_b)

    level_label_a = f"({args.level_a})" if args.level_a != "default" else "(default)"
    level_label_b = f"({args.level_b})" if args.level_b != "default" else "(default)"

    print("\n" + "=" * 90)
    print("CONFRONTO EMBEDDING - TinyVLA - LIBERO Goal")
    print("=" * 90)
    print(f"\n  Task A: {args.task_a} {level_label_a}")
    print(f"    Comando: \"{cmd_a}\"")
    print(f"\n  Task B: {args.task_b} {level_label_b}")
    print(f"    Comando: \"{cmd_b}\"")
    if args.scene_task:
        print(f"\n  Scena visiva forzata: {args.scene_task}")
    print()

    # ── Caricamento modello ──
    tokenizer, policy, image_processor = load_model(args.checkpoint, args.model_base)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_goal"]()

    # ── Cattura frame ──
    if args.scene_task:
        scene_tasks = [args.scene_task]
    else:
        scene_tasks = list(dict.fromkeys([args.task_a, args.task_b]))

    frame_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for task_key in scene_tasks:
        task_id = TASK_INDEX[task_key]
        task = task_suite.get_task(task_id)
        print(f"Cattura primo frame per: '{task_key}' (id={task_id})")
        img_np, wrist_np = get_first_frame(task, task_id, task_suite, seed=args.seed)
        frame_cache[task_key] = (img_np, wrist_np)

    # ── Estrazione embedding ──
    scene_key_a = args.scene_task if args.scene_task else args.task_a
    scene_key_b = args.scene_task if args.scene_task else args.task_b

    img_a, wrist_a = frame_cache[scene_key_a]
    img_b, wrist_b = frame_cache[scene_key_b]

    print(f"\nEstrazione embedding POST-fusione A: \"{cmd_a}\"")
    emb_a = extract_embedding(policy, tokenizer, image_processor, cmd_a, img_a, wrist_a)

    print(f"Estrazione embedding POST-fusione B: \"{cmd_b}\"")
    emb_b = extract_embedding(policy, tokenizer, image_processor, cmd_b, img_b, wrist_b)

    # ── Embedding pre-fusione (se richiesto) ──
    pre_a, pre_b = None, None
    if args.pre_fusion:
        print(f"Estrazione embedding PRE-fusione A: \"{cmd_a}\"")
        pre_a = extract_prefusion_embedding(policy, tokenizer, cmd_a)
        print(f"Estrazione embedding PRE-fusione B: \"{cmd_b}\"")
        pre_b = extract_prefusion_embedding(policy, tokenizer, cmd_b)

    # ── Calcolo distanze ──
    cos_sim = cosine_similarity(emb_a, emb_b)
    euc_dist = euclidean_distance(emb_a, emb_b)

    if args.pre_fusion:
        pre_cos_sim = cosine_similarity(pre_a, pre_b)
        pre_euc_dist = euclidean_distance(pre_a, pre_b)

    # ── Stampa risultati ──
    print("\n" + "=" * 90)
    print("RISULTATI")
    print("=" * 90)

    def trunc(s, n=55):
        return s if len(s) <= n else s[:n - 1] + "…"

    print(f"\n  {'Comando A:':<14} {trunc(cmd_a, 70)}")
    print(f"  {'  Task:':<14} {args.task_a} {level_label_a}")
    print(f"  {'  Scena:':<14} {scene_key_a}")
    print()
    print(f"  {'Comando B:':<14} {trunc(cmd_b, 70)}")
    print(f"  {'  Task:':<14} {args.task_b} {level_label_b}")
    print(f"  {'  Scena:':<14} {scene_key_b}")

    print(f"\n  {'═' * 50}")
    print(f"  POST-FUSIONE (multimodale, dopo il forward GPT-NeoX)")
    print(f"  {'─' * 50}")
    print(f"  Similarità coseno:  {cos_sim:.6f}")
    print(f"  Distanza euclidea:  {euc_dist:.4f}")
    print(f"  Dimensione embedding: {emb_a.shape}")
    print(f"  Norma embedding A:    {np.linalg.norm(emb_a):.4f}")
    print(f"  Norma embedding B:    {np.linalg.norm(emb_b):.4f}")

    if args.pre_fusion:
        print(f"\n  {'═' * 50}")
        print(f"  PRE-FUSIONE (puramente testuale, token lookup)")
        print(f"  {'─' * 50}")
        print(f"  Similarità coseno:  {pre_cos_sim:.6f}")
        print(f"  Distanza euclidea:  {pre_euc_dist:.4f}")
        print(f"  Dimensione embedding: {pre_a.shape}")
        print(f"  Norma embedding A:    {np.linalg.norm(pre_a):.4f}")
        print(f"  Norma embedding B:    {np.linalg.norm(pre_b):.4f}")

        print(f"\n  {'═' * 50}")
        print(f"  DELTA (post - pre fusione)")
        print(f"  {'─' * 50}")
        print(f"  Δ Similarità coseno:  {cos_sim - pre_cos_sim:+.6f}")
        print(f"  Δ Distanza euclidea:  {euc_dist - pre_euc_dist:+.4f}")

    print("\n" + "=" * 90)
    print("Fine confronto.")


if __name__ == "__main__":
    main()
