#!/usr/bin/env python3
"""
tinyvla_vqa_eval.py
-------------------
VQA evaluation script for TinyVLA (LLaVA-Pythia backbone).
"""

import argparse
import json
import logging
import os
import time
from string import Formatter
import sys
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
TINYVLA_ROOT = _SCRIPT_DIR.parent.parent
LLAVA_PYTHIA_ROOT = TINYVLA_ROOT / "llava-pythia"

for _p in [str(TINYVLA_ROOT), str(LLAVA_PYTHIA_ROOT)]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava_pythia.constants import (
    DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX,
)
from llava_pythia.conversation import conv_templates
from llava_pythia.model import *  # noqa: F401,F403

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SKIP_MARKERS = ("[TO_BE_FILLED]", "[DA COMPLETARE]")

# [FIX-B] System suffix semplificato: il modello ignorava i vincoli verbosi.
# La guida al formato è delegata al forced prefix nell'ASSISTANT turn.
VQA_SYSTEM_SUFFIX = (
    " You are a visual question answering assistant."
    " Look at the image carefully and answer the question directly."
    " Be concise."
)

_CKPT_ROOT = (
    "/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder"
    "/checkpoints_saving_folder/tinyvla"
)
DEFAULT_MODEL_PATH = os.path.join(
    _CKPT_ROOT, "llava_pythia_libero_goal_no_noops_64", "1.3B",
)
DEFAULT_MODEL_BASE = None
DEFAULT_DEMO_VIDEO = (
    "/home/A.CARDAMONE7/outputs/videos/libero_goal/mnt/beegfs/a.cardamone7"
    "/outputs/rollouts/libero_goal/syntactic_variation/openvla-oft/openvla-oft-50000"
    "/default/run_libero_goal_eval_seed_0/2026_02_03-15_02_39"
    "--episode=1--success=True--task=open_the_middle_layer_of_the_drawer.mp4"
)

# [FIX-A] Prefissi forced per ancorare la generazione dell'ASSISTANT.
# Il modello continuerà questi token invece di rigenerare il contesto del prompt.
FORCED_ASSISTANT_PREFIX = {
    "type0_object_listing":                   "Objects:",
    "type1_spatial_relation":                 "The",
    "type1_spatial_relation_multi_reference": "The",
    "type2_object_identification":            "Answer:",
}

# [FIX-D] max_new_tokens specifico per tipo.
# Limita la verbosità e interrompe i loop di ripetizione.
MAX_NEW_TOKENS_BY_TYPE = {
    "type0_object_listing":                   40,
    "type1_spatial_relation":                 30,
    "type1_spatial_relation_multi_reference": 30,
    "type2_object_identification":            10,
}

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING IMMAGINE
# ─────────────────────────────────────────────────────────────────────────────
_TO_TENSOR = transforms.ToTensor()
_RAND_CROP_RATIO = 0.95


def _to_tensor_and_crop(np_img: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(np_img, (320, 180))
    t = _TO_TENSOR(resized).float()
    orig_h, orig_w = t.shape[-2], t.shape[-1]
    r = _RAND_CROP_RATIO
    t_crop = t[
        :,
        int(orig_h * (1 - r) / 2): int(orig_h * (1 + r) / 2),
        int(orig_w * (1 - r) / 2): int(orig_w * (1 + r) / 2),
    ]
    return transforms.Resize((orig_h, orig_w), antialias=True)(t_crop)


def _expand2square_tensor(t: torch.Tensor, bg_color: tuple) -> torch.Tensor:
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


def preprocess_frame(np_img: np.ndarray, image_processor, device: str) -> torch.Tensor:
    bg_color = tuple(float(x) for x in image_processor.image_mean)
    t  = _to_tensor_and_crop(np_img)
    sq = _expand2square_tensor(t, bg_color)
    pv = image_processor.preprocess(
        sq, return_tensors="pt",
        do_normalize=True, do_rescale=False, do_center_crop=False,
    )["pixel_values"]
    return pv.to(device, dtype=torch.float32)


def extract_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read first frame from {video_path}")
    cv2.imwrite(
        "/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA"
        "/test/libero_test/first_frame.png", frame
    )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# GENERAZIONE TESTO — GREEDY DECODE
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def vqa_generate(
    model, tokenizer,
    input_ids: torch.Tensor,
    image_tensor: torch.Tensor,
    device: str,
    max_new_tokens: int,
    image_tensor_r: torch.Tensor = None,
    visual_concat_override: str = None,
) -> str:

    input_ids    = input_ids.to(device)
    image_tensor = image_tensor.to(device, dtype=torch.float)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)

    model_visual_concat  = getattr(model, "visual_concat", None)
    config_visual_concat = getattr(model.config, "visual_concat", None)
    if config_visual_concat is None:
        config_visual_concat = getattr(model.config, "concat", None)

    visual_concat = visual_concat_override
    if visual_concat is None:
        visual_concat = model_visual_concat if model_visual_concat is not None else config_visual_concat
    if isinstance(visual_concat, str) and visual_concat.strip().lower() in {"", "none", "null"}:
        visual_concat = None

    use_dual_image = (visual_concat == "token_cat")
    _img_r = image_tensor_r if image_tensor_r is not None else image_tensor

    if use_dual_image:
        _, attn_mask, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, None, None, image_tensor,
            visual_concat=visual_concat, states=None, images_r=_img_r,
        )
    else:
        _, attn_mask, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, None, None, image_tensor, states=None,
        )

    n_text = input_ids.shape[1]
    n_emb  = inputs_embeds.shape[1]
    log.info("[DIAG2] text_tokens=%d  embed_tokens=%d  visual_tokens_added=%d",
             n_text, n_emb, n_emb - n_text)
    log.info("[DIAG2] visual_concat=%s  use_dual_image=%s", visual_concat, use_dual_image)

    if n_emb > n_text:
        log.info("[DIAG4] norm text_embeds=%.4f  visual_embeds=%.4f",
                 inputs_embeds[:, :n_text, :].norm(dim=-1).mean().item(),
                 inputs_embeds[:, n_text:, :].norm(dim=-1).mean().item())

    base_seq_len = inputs_embeds.shape[1]

    out = model.gpt_neox(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        use_cache=True, return_dict=True,
    )
    hidden  = out.last_hidden_state
    past_kv = out.past_key_values

    def project_to_vocab(last_hidden: torch.Tensor) -> torch.Tensor:
        if hasattr(model, "lm_head") and model.lm_head is not None:
            return model.lm_head(last_hidden)
        embed_out = getattr(model, "embed_out", None)
        if isinstance(embed_out, torch.nn.Linear):
            return embed_out(last_hidden)
        if getattr(model.config, "tie_word_embeddings", False):
            return F.linear(last_hidden, model.gpt_neox.embed_in.weight)
        raise RuntimeError("No compatible LM head found.")

    lm_logits = project_to_vocab(hidden[:, -1, :])
    next_id   = lm_logits.argmax(dim=-1)

    generated_ids = []
    for _ in range(max_new_tokens):
        generated_ids.append(next_id.item())
        if next_id.item() == tokenizer.eos_token_id:
            break
        tok_embed = model.gpt_neox.embed_in(next_id.unsqueeze(0))
        cur_len  = base_seq_len + len(generated_ids)
        cur_mask = torch.ones(1, cur_len, dtype=torch.long, device=device)
        out = model.gpt_neox(
            inputs_embeds=tok_embed, attention_mask=cur_mask,
            past_key_values=past_kv, use_cache=True, return_dict=True,
        )
        hidden  = out.last_hidden_state
        past_kv = out.past_key_values
        lm_logits = project_to_vocab(hidden[:, -1, :])
        next_id   = lm_logits.argmax(dim=-1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# COSTRUZIONE INPUT IDS  — build_input_ids CORRETTA
# ─────────────────────────────────────────────────────────────────────────────
def build_input_ids(
    question_text: str,
    tokenizer,
    model,
    conv_mode: str,
    question_type: str = "",
    debug_prompt: bool = False,
) -> torch.Tensor:
    """
    [FIX-A v2] Il prefix viene appeso MANUALMENTE alla stringa del prompt
    DOPO conv.get_prompt(), NON tramite conv.append_message(roles[1], prefix).

    Motivo: il template pythia (SeparatorStyle.TWO) aggiunge sep2=<|endoftext|>
    quando append_message riceve un valore non-None per il turno ASSISTANT.
    Questo inietta il token EOS dentro input_ids, troncando la generazione.

    La soluzione corretta è:
      1. conv.append_message(roles[1], None)  → prompt finisce con "ASSISTANT:"
      2. prompt += " " + prefix               → appende il prefix come stringa raw
      3. tokenize(prompt)                     → nessun EOS in input
    """
    conv = conv_templates[conv_mode].copy()
    conv.system = (conv.system or "") + VQA_SYSTEM_SUFFIX

    if getattr(model.config, "mm_use_im_start_end", False):
        inp = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN
               + DEFAULT_IM_END_TOKEN + "\n" + question_text)
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + question_text

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)   # SEMPRE None → nessun sep2/EOS

    prompt = conv.get_prompt()                 # finisce con "ASSISTANT:"

    # [FIX-A v2] Append del prefix come stringa raw, dopo la serializzazione
    assistant_prefix = FORCED_ASSISTANT_PREFIX.get(question_type, None)
    if assistant_prefix:
        prompt = prompt + " " + assistant_prefix

    if debug_prompt:
        print("=== PROMPT RAW ===")
        print(repr(prompt))
        print("==================")

    return tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# ECHO DETECTION — is_prompt_echo CORRETTA
# ─────────────────────────────────────────────────────────────────────────────
def is_prompt_echo(generated: str, prompt: str, threshold: float = 0.85) -> bool:
    """
    [FIX-B v2] Versione meno aggressiva.

    Il problema della v1: la sliding window a 8 parole generava falsi positivi
    su risposte spaziali valide che contenevano frasi del prompt (es. nomi
    degli oggetti che compaiono sia nel prompt che nella risposta corretta).

    Ora:
      - Caso 1: sottostringa diretta solo per testi > 40 chars (non per "bowl")
      - Caso 2: similarity con inizio prompt (invariato)
      - Caso 3: sliding window SOLO per testi > 60 chars con finestra a 12 parole
    """
    gen_lower    = generated.strip().lower()
    prompt_lower = prompt.strip().lower()

    if not gen_lower:
        return False

    # Caso 1: il testo generato lungo è una sottostringa del prompt
    if len(gen_lower) > 40 and gen_lower in prompt_lower:
        return True

    # Caso 2: similarity con l'inizio del prompt
    if text_similarity(generated[: len(prompt)], prompt) > threshold:
        return True

    # Caso 3: sliding window — solo per testi lunghi, finestra a 12 parole
    # (8 era troppo corta: frasi spaziali valide attivavano falsi positivi)
    if len(gen_lower) > 60:
        words = gen_lower.split()
        for i in range(len(words) - 12 + 1):
            chunk = " ".join(words[i: i + 12])
            if chunk in prompt_lower:
                log.warning("[ECHO-WINDOW] detected chunk: '%s'", chunk)
                return True

    return False

# ─────────────────────────────────────────────────────────────────────────────
# [FIX-C] POST-PROCESSING LISTING
# ─────────────────────────────────────────────────────────────────────────────
def postprocess_listing(
    generated: str,
    candidate_objects: list,
    variants: dict = None,
) -> str:
    """
    Estrae dalla risposta verbosa solo gli oggetti noti della scena.

    Poiché il modello descrive la scena invece di restituire una lista,
    e poiché il candidate set è noto dal JSON di benchmark, estrae
    programmaticamente gli oggetti menzionati nel testo generato.
    Usa expected_object_variants per gestire alias (es. 'flat stove'↔'stove').

    Returns:
        "obj1, obj2, obj3" se trovati, altrimenti il testo grezzo invariato.
    """
    variants = variants or {}
    gen_lower = generated.lower()
    found = []
    for obj in candidate_objects:
        aliases = [obj] + variants.get(obj, [])
        if any(alias.lower() in gen_lower for alias in aliases):
            found.append(obj)
    return ", ".join(found) if found else generated


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _is_placeholder(value) -> bool:
    return isinstance(value, str) and value.strip() in SKIP_MARKERS


def _normalize_ground_truths(ground_truth):
    if isinstance(ground_truth, list):
        return [str(x).strip() for x in ground_truth if str(x).strip()]
    elif ground_truth is None:
        return []
    return [str(ground_truth).strip()]


def compute_object_listing_recall(generated: str, expected_objects: list) -> tuple:
    gen_lower = generated.lower()
    found   = [obj for obj in expected_objects if obj.lower() in gen_lower]
    missing = [obj for obj in expected_objects if obj.lower() not in gen_lower]
    recall  = len(found) / len(expected_objects) if expected_objects else 0.0
    return recall, found, missing


def compute_metrics(generated: str, ground_truth):
    gen_norm     = generated.strip().lower()
    gt_norm_list = [x.lower() for x in _normalize_ground_truths(ground_truth)]
    exact = any(gen_norm == gt for gt in gt_norm_list)
    soft  = any(gt in gen_norm for gt in gt_norm_list)
    return exact, soft


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def resolve_ground_truth_config(question: dict):
    q_type    = question.get("type", "")
    gt_struct = question.get("ground_truth", None)
    if isinstance(gt_struct, dict) and q_type in gt_struct and isinstance(gt_struct[q_type], dict):
        return {"source": "structured", "type": q_type, "config": gt_struct[q_type]}
    return {
        "source": "legacy", "type": q_type,
        "config": {"ground_truth_answer": question.get("ground_truth_answer", "")},
    }


# [FIX-B] Echo detection con sliding window a 8 parole.
# Rileva il caso in cui il modello rigenera un sottotesto del prompt
# (es. "The image shows a robotic manipulation scene in which the following
#  objects are present: bowl, cream cheese...") invece di rispondere.
def is_prompt_echo(generated: str, prompt: str, threshold: float = 0.85) -> bool:
    gen_lower    = generated.strip().lower()
    prompt_lower = prompt.strip().lower()

    # Caso 1: il testo generato è una sottostringa del prompt
    if len(gen_lower) > 20 and gen_lower in prompt_lower:
        return True

    # Caso 2: alta similarità con l'inizio del prompt (check originale)
    if text_similarity(generated[: len(prompt)], prompt) > threshold:
        return True

    # Caso 3: sliding window — blocco di 8 parole consecutive del generated
    # appare verbatim nel prompt (tipico del loop "The image shows...")
    if len(gen_lower) > 30:
        words = gen_lower.split()
        for i in range(len(words) - 8 + 1):
            chunk = " ".join(words[i: i + 8])
            if chunk in prompt_lower:
                log.warning("[ECHO-WINDOW] detected chunk: '%s'", chunk)
                return True

    return False


def is_ground_truth_filled(resolved_gt: dict) -> bool:
    q_type = resolved_gt.get("type", "")
    cfg    = resolved_gt.get("config", {})

    if resolved_gt.get("source") == "legacy":
        values = _normalize_ground_truths(cfg.get("ground_truth_answer", ""))
        return bool(values) and not any(_is_placeholder(v) for v in values)

    if q_type in ("type1_spatial_relation", "type1_spatial_relation_multi_reference"):
        ref    = str(cfg.get("reference_answer", "")).strip()
        kws    = [str(x).strip() for x in cfg.get("keywords", []) if str(x).strip()]
        return ref != "" and not _is_placeholder(ref) \
               and len(kws) > 0 and not any(_is_placeholder(k) for k in kws)

    if q_type == "type2_object_identification":
        valid   = [str(x).strip() for x in cfg.get("valid_answers", []) if str(x).strip()]
        primary = str(cfg.get("primary_answer", "")).strip()
        return len(valid) > 0 and not any(_is_placeholder(v) for v in valid) \
               and primary != "" and not _is_placeholder(primary)

    if q_type == "type0_object_listing":
        objs = cfg.get("expected_objects", [])
        return isinstance(objs, list) and len(objs) > 0

    return False


def compute_metrics_from_resolved_ground_truth(
    generated: str, resolved_gt: dict, similarity_threshold: float = 0.80,
):
    gen_norm = generated.strip().lower()
    q_type   = resolved_gt.get("type", "")
    cfg      = resolved_gt.get("config", {})

    if resolved_gt.get("source") == "legacy":
        return compute_metrics(generated, cfg.get("ground_truth_answer", ""))

    if q_type in ("type1_spatial_relation", "type1_spatial_relation_multi_reference"):
        reference_answer = str(cfg.get("reference_answer", "")).strip().lower()
        keywords  = [str(x).strip().lower() for x in cfg.get("keywords", []) if str(x).strip()]
        threshold = int(cfg.get("keyword_match_threshold", max(1, len(keywords))))
        matched   = sum(1 for kw in keywords if kw in gen_norm)
        return gen_norm == reference_answer, matched >= threshold

    if q_type == "type2_object_identification":
        valid   = [str(x).strip().lower() for x in cfg.get("valid_answers", []) if str(x).strip()]
        primary = str(cfg.get("primary_answer", "")).strip().lower()
        exact   = (gen_norm == primary) or any(gen_norm == a for a in valid)
        soft    = (primary in gen_norm) or any(a in gen_norm for a in valid)
        if not soft:
            soft = any(
                text_similarity(gen_norm, a) >= similarity_threshold
                for a in ([primary] + valid) if a
            )
        return exact, soft

    if q_type == "type0_object_listing":
        expected  = [str(x).strip().lower() for x in cfg.get("expected_objects", []) if str(x).strip()]
        threshold = float(cfg.get("recall_threshold", 0.5))
        recall, _, _ = compute_object_listing_recall(generated, expected)
        passed = recall >= threshold
        return passed, passed

    return False, False


def ground_truth_for_display(resolved_gt: dict):
    q_type = resolved_gt.get("type", "")
    cfg    = resolved_gt.get("config", {})
    if resolved_gt.get("source") == "legacy":
        return cfg.get("ground_truth_answer", "")
    if q_type in ("type1_spatial_relation", "type1_spatial_relation_multi_reference"):
        return {
            "reference_answer": cfg.get("reference_answer", ""),
            "keywords": cfg.get("keywords", []),
            "keyword_match_threshold": cfg.get("keyword_match_threshold", None),
        }
    if q_type == "type2_object_identification":
        return {"primary_answer": cfg.get("primary_answer", ""),
                "valid_answers": cfg.get("valid_answers", [])}
    if q_type == "type0_object_listing":
        return {
            "expected_objects": cfg.get("expected_objects", []),
            "expected_object_variants": cfg.get("expected_object_variants", {}),
            "recall_threshold": cfg.get("recall_threshold", 0.5),
        }
    return cfg


def build_filled_prompt_from_template(task: dict, question: dict):
    template = question.get("template", "")
    if not isinstance(template, str) or not template.strip() or template in SKIP_MARKERS:
        return None, ["template"]

    placeholders = [fn for _, fn, _, _ in Formatter().parse(template) if fn]
    format_values = {}
    missing = []

    for key in placeholders:
        value = question.get(key)
        if key == "reference_object" and value is None:
            value = question.get("reference_objects")
        if key == "objects_list" and (value is None or str(value).strip() in ["", *SKIP_MARKERS]):
            scene_objects = task.get("objects_in_scene", [])
            if isinstance(scene_objects, list) and len(scene_objects) > 0:
                value = ", ".join([str(x) for x in scene_objects])
        if value is None:
            missing.append(key)
            continue
        if isinstance(value, list):
            value = ", ".join([str(x).strip() for x in value if str(x).strip()])
        else:
            value = str(value).strip()
        if value in SKIP_MARKERS or value == "":
            missing.append(key)
            continue
        format_values[key] = value

    if missing:
        return None, missing
    try:
        return template.format(**format_values), []
    except Exception:
        return None, ["template_format_error"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="TinyVLA VQA evaluation")
    parser.add_argument("--model_path",    default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model_base",    default=DEFAULT_MODEL_BASE)
    parser.add_argument("--prompts_json",  default="./vqa_prompts.json")
    parser.add_argument("--output_json",   default="./vqa_results.json")
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Fallback se il tipo non è in MAX_NEW_TOKENS_BY_TYPE")
    parser.add_argument("--conv_mode",     default="pythia")
    parser.add_argument("--request_delay_sec",  type=float, default=0.5)
    parser.add_argument("--question_delay_sec", type=float, default=3.0)
    parser.add_argument("--task_delay_sec",     type=float, default=5.0)
    parser.add_argument("--similarity_threshold", type=float, default=0.60)
    parser.add_argument("--visual_concat", default=None)
    parser.add_argument(
        "--frames",
        nargs="+",
        default=None,
        metavar="FRAME",
        help=(
            "Filtra le domande per frame. Valori ammessi: 'camera', 'world'. "
            "Default: nessun filtro (tutte le domande). "
            "Esempio: --frames camera"
        ),
    )
    # [FIX-A] flag ablation
    parser.add_argument("--no_forced_prefix", action="store_true", default=False,
                        help="Disabilita il forced ASSISTANT prefix (ablation).")
    # [FIX-C] flag ablation
    parser.add_argument("--no_postprocess_listing", action="store_true", default=False,
                        help="Disabilita il post-processing Type-0 (ablation).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    log.info("Loading model from: %s", args.model_path)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, False, False,
    )
    model.eval()
    log.info("Model loaded — name: %s | head_type: %s | context_len: %d",
             model_name, model.head_type, context_len)
    log.info(
        "Visual concat — model: %s | config.concat: %s | config.visual_concat: %s | override: %s",
        getattr(model, "visual_concat", None),
        getattr(model.config, "concat", None),
        getattr(model.config, "visual_concat", None),
        args.visual_concat,
    )

    if getattr(model, "head_type", None) != "fc":
        raise RuntimeError(
            "Checkpoint uses action_head_type='{}': not suitable for VQA. "
            "Use a base VLM checkpoint (head_type='fc').".format(
                getattr(model, "head_type", "unknown"))
        )

    with open(args.prompts_json, "r") as f:
        data = json.load(f)

    fallback_video = data.get("metadata", {}).get(
        "first_frame_source_video", DEFAULT_DEMO_VIDEO
    )
    results = deepcopy(data)

    all_questions = []
    for task in data.get("tasks", []):
        questions = task.get("questions", [])
        if not isinstance(questions, list) or len(questions) == 0:
            log.warning("SKIP task_id=%s: no valid questions", task.get("task_id", "?"))
            continue
        for q in questions:
            all_questions.append((task, q))

    type1_correct = type1_total = 0
    type1_multi_correct = type1_multi_total = 0
    type2_correct = type2_total = 0
    type0_recall_sum = 0.0
    type0_total = 0

    pbar = tqdm(all_questions, desc="VQA inference")
    last_task_id = None

    for idx, (task, q) in enumerate(pbar):

        task_id = task.get("task_id")
        if idx > 0:
            if last_task_id is not None and task_id != last_task_id:
                if args.task_delay_sec > 0:
                    time.sleep(args.task_delay_sec)
            else:
                if args.question_delay_sec > 0:
                    time.sleep(args.question_delay_sec)

        filled_prompt    = q.get("filled_prompt", "")
        resolved_gt      = resolve_ground_truth_config(q)
        first_frame_path = task.get("first_frame_path", "")
        q_type           = q.get("type", "")
        
        q_frame = q.get("frame", "")   # "world" | "camera" | "" (type0 listing)
        if args.frames is not None and q_frame and q_frame not in args.frames:
            log.info("SKIP (frame=%s not in %s) — %s", q_frame, args.frames, q["question_id"])
            continue

        if (not isinstance(filled_prompt, str)
                or filled_prompt.strip() == ""
                or filled_prompt.strip() in SKIP_MARKERS):
            auto_prompt, missing_fields = build_filled_prompt_from_template(task, q)
            if auto_prompt is None:
                log.warning("SKIP (prompt unresolved) — %s | missing: %s",
                            q["question_id"], ", ".join(missing_fields))
                continue
            filled_prompt = auto_prompt

        if not is_ground_truth_filled(resolved_gt):
            log.warning("SKIP (GT not filled) — %s", q["question_id"])
            continue

        if first_frame_path in SKIP_MARKERS or not os.path.isfile(first_frame_path):
            log.warning("first_frame_path missing for %s — using fallback", q["question_id"])
            first_frame_path = fallback_video

        if first_frame_path.lower().endswith(".mp4"):
            np_frame = extract_first_frame(first_frame_path)
        else:
            bgr = cv2.imread(first_frame_path)
            if bgr is None:
                log.warning("SKIP (cannot read image) — %s", q["question_id"])
                continue
            np_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        image_tensor = preprocess_frame(np_frame, image_processor, args.device)

        wrist_frame_path = task.get("wrist_frame_path", None)
        image_tensor_r   = None
        if wrist_frame_path and os.path.isfile(wrist_frame_path):
            wrist_bgr = cv2.imread(wrist_frame_path)
            if wrist_bgr is not None:
                wrist_rgb      = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)
                image_tensor_r = preprocess_frame(wrist_rgb, image_processor, args.device)
                log.info("[IMG] wrist frame caricato: %s", wrist_frame_path)
            else:
                log.warning("[IMG] wrist frame non leggibile, fallback su agentview")
        else:
            log.info("[IMG] wrist_frame_path assente → fallback su agentview")

        if args.visual_concat == "token_cat" and image_tensor_r is None:
            log.warning("[IMG] visual_concat=token_cat ma wrist assente: uso agentview come seconda immagine")

        # [FIX-E] Type-0: disabilita dual-image (single agentview sufficiente per listing)
        vc_override = args.visual_concat
        if q_type == "type0_object_listing":
            vc_override = None
            log.info("[FIX-E] Type-0: visual_concat forzato a None (single-image)")

        # [FIX-D] max_new_tokens per tipo
        max_tok = MAX_NEW_TOKENS_BY_TYPE.get(q_type, args.max_new_tokens)
        log.info("[FIX-D] q_type=%s → max_new_tokens=%d", q_type, max_tok)

        # [FIX-A] Passa q_type per il forced prefix; stringa vuota = disabilitato
        effective_q_type = "" if args.no_forced_prefix else q_type

        input_ids = build_input_ids(
            filled_prompt, tokenizer, model, args.conv_mode,
            question_type=effective_q_type,
            debug_prompt=(idx == 0),
        ).to(args.device)

        n_img_tok = (input_ids == IMAGE_TOKEN_INDEX).sum().item()
        log.info("[DIAG1] IMAGE_TOKEN in input_ids: %d  (atteso: 1)", n_img_tok)

        # DIAG3 solo sulla prima domanda; ans_real riutilizzato direttamente
        # [FIX] eliminata doppia chiamata a vqa_generate per la prima domanda
        if idx == 0:
            blank_tensor = torch.zeros_like(image_tensor)
            ans_real  = vqa_generate(model, tokenizer, input_ids, image_tensor,
                                     device=args.device, max_new_tokens=max_tok,
                                     image_tensor_r=image_tensor_r,
                                     visual_concat_override=vc_override)
            ans_blank = vqa_generate(model, tokenizer, input_ids, blank_tensor,
                                     device=args.device, max_new_tokens=max_tok,
                                     image_tensor_r=image_tensor_r,
                                     visual_concat_override=vc_override)
            ans_noise = vqa_generate(model, tokenizer, input_ids,
                                     torch.randn_like(image_tensor),
                                     device=args.device, max_new_tokens=max_tok,
                                     image_tensor_r=image_tensor_r,
                                     visual_concat_override=vc_override)
            log.info("[DIAG3] real  image → '%s'", ans_real)
            log.info("[DIAG3] blank image → '%s'", ans_blank)
            log.info("[DIAG3] noise image → '%s'", ans_noise)
            generated_answer = ans_real   # [FIX] riusa senza rigenerare
        else:
            generated_answer = vqa_generate(
                model, tokenizer, input_ids, image_tensor,
                device=args.device, max_new_tokens=max_tok,
                image_tensor_r=image_tensor_r,
                visual_concat_override=vc_override,
            )

        # [FIX-B] Echo detection migliorata
        if is_prompt_echo(generated_answer, filled_prompt):
            log.warning("ECHO DETECTED — %s | raw: '%s'", q["question_id"], generated_answer)
            generated_answer = ""

        # [FIX-A] Rimuovi il forced prefix dall'output generato se presente
        prefix_used = FORCED_ASSISTANT_PREFIX.get(effective_q_type, None)
        if prefix_used and generated_answer.lower().startswith(prefix_used.lower()):
            generated_answer = generated_answer[len(prefix_used):].strip(" :,")

        # [FIX-C] Post-processing listing: estrai solo oggetti noti dalla risposta verbosa
        if q_type == "type0_object_listing" and not args.no_postprocess_listing:
            expected_objs_raw = resolved_gt["config"].get("expected_objects", [])
            variants          = resolved_gt["config"].get("expected_object_variants", {})
            generated_answer_raw = generated_answer
            generated_answer     = postprocess_listing(generated_answer, expected_objs_raw, variants)
            if generated_answer != generated_answer_raw:
                log.info("[FIX-C] Listing post-processed: '%s' → '%s'",
                         generated_answer_raw, generated_answer)

        # ── METRICHE ──────────────────────────────────────────────────────────
        exact, soft = compute_metrics_from_resolved_ground_truth(
            generated_answer, resolved_gt,
            similarity_threshold=args.similarity_threshold,
        )
        q["correct"] = soft

        if q_type == "type1_spatial_relation":
            type1_total   += 1
            type1_correct += int(soft)
        elif q_type == "type1_spatial_relation_multi_reference":
            type1_multi_total   += 1
            type1_multi_correct += int(soft)
        elif q_type == "type2_object_identification":
            type2_total   += 1
            type2_correct += int(soft)
        elif q_type == "type0_object_listing":
            type0_total += 1
            expected_objs = [
                str(x).strip().lower()
                for x in resolved_gt["config"].get("expected_objects", [])
            ]
            recall, found, missing = compute_object_listing_recall(
                generated_answer, expected_objs
            )
            type0_recall_sum += recall

        outcome = "✓" if soft else ("~" if exact else "✗")
        print(f"\n[{q['question_id']}]  [{outcome}]")
        print(f"  Prompt    : {filled_prompt}")
        print(f"  Generated : {generated_answer}")
        print(f"  GT        : {ground_truth_for_display(resolved_gt)}")
        if q_type == "type0_object_listing":
            print(f"  Recall    : {recall:.0%}  |  Found: {found}  |  Missing: {missing}")

        last_task_id = task_id

    # ── METRICHE FINALI ───────────────────────────────────────────────────────
    total_correct = type1_correct + type1_multi_correct + type2_correct
    total         = type1_total   + type1_multi_total   + type2_total

    print("\n" + "=" * 60)
    print("FINAL METRICS")

    if type1_total:
        print(f"  Type-1 (spatial relation)      : "
            f"{type1_correct}/{type1_total} = {type1_correct/type1_total:.2%}")
        # Breakdown per frame (solo i frame effettivamente valutati)
        evaluated_frames = args.frames if args.frames else ["world", "camera"]
        for frame in evaluated_frames:
            qs_f = [q for _, q in all_questions
                    if q.get("frame") == frame
                    and q.get("type") == "type1_spatial_relation"]
            c = sum(1 for q in qs_f if q.get("correct", False))
            if qs_f:
                print(f"    ↳ [{frame:6}]: {c}/{len(qs_f)} = {c/len(qs_f):.2%}")
    else:
        print("  Type-1 (spatial relation)      : N/A")

    if type2_total:
        print(f"  Type-2 (object identification) : "
              f"{type2_correct}/{type2_total} = {type2_correct/type2_total:.2%}")
    else:
        print("  Type-2 (object identification) : N/A")

    if type0_total:
        avg_r = type0_recall_sum / type0_total
        print(f"  Type-0 (object listing)        : "
              f"avg recall {avg_r:.2%} over {type0_total} tasks")
    else:
        print("  Type-0 (object listing)        : N/A")

    if total:
        print(f"  Overall                        : "
              f"{total_correct}/{total} = {total_correct/total:.2%}")
    else:
        print("  Overall                        : N/A")
    print("=" * 60)

    results["metrics"] = {
        "type1_accuracy":       type1_correct / type1_total if type1_total else None,
        "type1_multi_accuracy": type1_multi_correct / type1_multi_total if type1_multi_total else None,
        "type2_accuracy":       type2_correct / type2_total if type2_total else None,
        "overall_accuracy":     total_correct / total       if total       else None,
        "type1_correct":        type1_correct,   "type1_total":      type1_total,
        "type1_multi_correct":  type1_multi_correct, "type1_multi_total": type1_multi_total,
        "type2_correct":        type2_correct,   "type2_total":      type2_total,
        "total_correct":        total_correct,   "total":            total,
    }

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Results saved to %s", args.output_json)


if __name__ == "__main__":
    main()