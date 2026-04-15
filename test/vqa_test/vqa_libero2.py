#!/usr/bin/env python3
"""
libero_spatial_benchmark_gen.py
Genera domande spaziali in DUE sistemi di riferimento:
  - world:  coordinate MuJoCo (robot-centrico)
  - camera: frame della agentview (quello che vede il VLM)
Solo per coppie di oggetti entro NEARBY_DIST_MAX metri l'una dall'altra.
"""

import argparse, json, os, sys
import numpy as np
import cv2

from utils.libero_utils import get_libero_env
from libero.libero import benchmark as B

# ─────────────────────────────────────────────────────────────────────────────
# Distanza massima (euclidea 3D) perché due oggetti siano considerati "vicini"
NEARBY_DIST_MAX = 0.65   # metri

SKIP_KEYWORDS = {
    "robot", "base", "link", "joint", "table", "floor", "wall",
    "mount", "gripper", "finger", "camera", "world", "unnamed",
    "visual", "collision", "site", "sensor",
    "flat_stove_1_burner", "flat_stove_1_burner_plate",
    "flat_stove_1_button", "wooden_cabinet_1_cabinet_bottom",
    "wooden_cabinet_1_cabinet_middle",
    "wooden_cabinet_1_cabinet_top",
}


# Task commands (original + variation) that require additional questions.
TASK_VARIATION_EXTRA_QUERIES = [
    {
        "original": "Open the middle layer of the drawer",
        "variation": "Open the layer of the cabinet located between the top and bottom",
    },
    {
        "original": "Put the bowl on the stove",
        "variation": "Put the object between the wine bottle and the cream cheese on the stove",
        "extra_questions": [
            {
                "target": "bowl",
                "references": ["wine bottle", "cream cheese"],
            }
        ],
    },
    {
        "original": "Put the wine bottle on the top of the cabinet",
        "variation": "Put the object behind the bowl on the top of the cabinet",
    },
    {
        "original": "Open the top layer of the drawer and put the bowl inside",
        "variation": "Open the top layer of the drawer and put the object between the plate and the cream cheese inside",
    },
    {
        "original": "Put the bowl on the top of the cabinet",
        "variation": "Put the object between the wine bottle and the cream cheese on the top of the cabinet",
    },
    {
        "original": "Push the plate to the front of the stove",
        "variation": "Push the object in front of the drawer to the front of the stove",
    },
    {
        "original": "Put the cream cheese on the bowl",
        "variation": "Put the object in front of the stove on the bowl",
    },
    {
        "original": "Turn on the stove",
        "variation": "Turn on the the object behind the cream cheese",
    },
    {
        "original": "Put the bowl on the plate",
        "variation": "Put the object between the wine bottle and the cream cheese on the plate",
    },
    {
        "original": "Put the wine bottle on the rack",
        "variation": "Put the object behind the bowl on the rack",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. RELAZIONI SPAZIALI
# ─────────────────────────────────────────────────────────────────────────────

def axis_relation_world(pos_a, pos_b, ambiguity_thresh=0.04):
    """
    Sistema di riferimento MuJoCo:
      x → destra robot   y → avanti robot   z → su
    """
    d = pos_a - pos_b
    abs_d = np.abs(d)
    if abs_d.max() < ambiguity_thresh:
        return "next to", 0.0
    dom = np.argmax(abs_d)
    if dom == 0:
        return ("right of" if d[0] > 0 else "left of"), abs(d[0])
    elif dom == 1:
        return ("in front of" if d[1] > 0 else "behind"), abs(d[1])
    else:
        return ("above" if d[2] > 0 else "below"), abs(d[2])


def get_camera_axes(env):
    """
    Legge gli assi della camera 'agentview' dal simulatore MuJoCo.
    cam_xmat è (3,3):
      riga 0 = asse RIGHT  della camera nel world frame
      riga 1 = asse UP     della camera nel world frame
      riga 2 = asse BACK   della camera nel world frame
    """
    sim = getattr(env, "sim", None) or getattr(env.env, "sim", None)
    cam_id = sim.model.camera_name2id("agentview")
    cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)
    cam_right = cam_mat[0]
    cam_up = cam_mat[1]
    print(f"  cam_right = {cam_right.round(3)}")
    print(f"  cam_up    = {cam_up.round(3)}")
    return cam_right, cam_up   # cam_right, cam_up in world coords


def axis_relation_camera(pos_a, pos_b, cam_right, cam_up,
                         ambiguity_thresh=0.04):
    """
    Sistema di riferimento camera (frame immagine).

    u = dot(d, cam_right) → u>0: A è a DESTRA di B nell'immagine
    v = dot(d, cam_up)    → v>0: A è più lontano (= visivamente più in alto
                                  nell'immagine capovolta = "behind")
    dz = d[2]             → above/below rimane world-z
    """
    d = pos_a - pos_b
    u  = float(np.dot(d, cam_right))
    v  = float(np.dot(d, cam_up))
    dz = float(d[2])

    mags = np.abs([u, v, dz])
    if mags.max() < ambiguity_thresh:
        return "next to", 0.0

    dom = np.argmax(mags)
    if dom == 0:
        return ("right of" if u > 0 else "left of"), abs(u)
    elif dom == 1:
        # v>0: A è nella direzione cam_up → visivamente "in alto" nell'img
        # immagine capovolta (MuJoCo OpenGL): "in alto" visivo = più lontano = behind
        return ("behind" if v > 0 else "in front of"), abs(v)
    else:
        return ("above" if dz > 0 else "below"), abs(dz)


def inverse_relation(rel):
    return {
        "left of": "right of", "right of": "left of",
        "in front of": "behind", "behind": "in front of",
        "above": "below", "below": "above", "next to": "next to",
    }.get(rel, rel)


def canonicalize_task_command(text):
    clean = (
        text.replace("\n", " ")
        .replace('"', " ")
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .lower()
    )
    return " ".join(clean.split())


def canonicalize_object_name(text):
    clean = text.lower().replace("_", " ").replace("-", " ").strip()
    return " ".join(clean.split())


def pretty_object_name(name):
    base = name
    base = base.replace("_main", "")
    if "_" in base and base.split("_")[-1].isdigit():
        base = "_".join(base.split("_")[:-1])
    clean = canonicalize_object_name(base)
    if clean in {"flat stove", "flat stove 1"}:
        return "stove"
    return clean


def build_task_query_index():
    idx = {}
    for item in TASK_VARIATION_EXTRA_QUERIES:
        query_cfg = item.get("extra_questions", [])
        if not query_cfg:
            continue
        idx[canonicalize_task_command(item["original"])] = query_cfg
        idx[canonicalize_task_command(item["variation"])] = query_cfg
    return idx


TASK_QUERY_INDEX = build_task_query_index()


def find_scene_object(scene_names, query_name):
    q = canonicalize_object_name(query_name)
    for scene_name in scene_names:
        if canonicalize_object_name(scene_name) == q:
            return scene_name
    for scene_name in scene_names:
        if pretty_object_name(scene_name) == q:
            return scene_name
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. POSIZIONI OGGETTI
# ─────────────────────────────────────────────────────────────────────────────

def get_object_positions(env):
    sim = getattr(env, "sim", None) or getattr(env.env, "sim", None)
    if sim is None:
        raise RuntimeError("Impossibile trovare env.sim")
    positions = {}
    for body_id in range(sim.model.nbody):
        name = sim.model.body_id2name(body_id)
        if any(kw in name.lower() for kw in SKIP_KEYWORDS):
            continue
        positions[name] = sim.data.body_xpos[body_id].copy()
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# 3. GENERAZIONE DOMANDE
# ─────────────────────────────────────────────────────────────────────────────

def make_type1_question(q_id, obj_a, obj_b, relation, objects_list, frame):
    objects_str = ", ".join(objects_list)
    prompt = (
        f"The image shows a robotic manipulation scene in which the following "
        f"objects are present: {objects_str}. "
        f"Could you describe ONLY the position of {obj_a} with respect to {obj_b}? "
        f"Use ONLY these position attributes: left of, right of, behind, in front of, above, below."
    )
    return {
        "question_id": q_id,
        "type": "type1_spatial_relation",
        "frame": frame,                    # "world" o "camera"
        "filled_prompt": prompt,
        "target_object": obj_a,
        "reference_object": obj_b,
        "objects_list": objects_str,
        "ground_truth": {
            "type1_spatial_relation": {
                "reference_answer": f"The {obj_a} is {relation} the {obj_b}",
                "keywords": [relation, obj_b],
                "keyword_match_threshold": 2,
                "_exact_relation": relation,
                "_obj_a": obj_a,
                "_obj_b": obj_b,
            }
        },
    }


def make_type2_question(q_id, target_obj, spatial_desc, objects_list, frame):
    objects_str = ", ".join(objects_list)
    prompt = (
        f"The image shows a robotic manipulation scene in which the following "
        f"objects are present: {objects_str}. "
        f"Which ONE of these objects is {spatial_desc}? "
        f"Answer with ONLY the object name from the list above, nothing else."
    )
    return {
        "question_id": q_id,
        "type": "type2_object_identification",
        "frame": frame,
        "filled_prompt": prompt,
        "spatial_description": spatial_desc,
        "objects_list": objects_str,
        "ground_truth": {
            "type2_object_identification": {
                "valid_answers": [target_obj],
                "primary_answer": target_obj,
            }
        },
    }


def make_type1_dual_ref_question(q_id, obj_a, ref_b, ref_c, relation,
                                 objects_list, frame):
    objects_str = ", ".join(objects_list)
    prompt = (
        f"The image shows a robotic manipulation scene in which the following "
        f"objects are present: {objects_str}. "
        f"Could you describe ONLY the position of {obj_a} with respect to the "
        f"{ref_b} and the {ref_c}? "
        f"Use ONLY these position attributes: left of, right of, behind, in front of, above, below."
    )
    return {
        "question_id": q_id,
        "type": "type1_spatial_relation_multi_reference",
        "frame": frame,
        "filled_prompt": prompt,
        "target_object": obj_a,
        "reference_objects": [ref_b, ref_c],
        "objects_list": objects_str,
        "ground_truth": {
            "type1_spatial_relation_multi_reference": {
                "reference_answer": (
                    f"The {obj_a} is {relation} the midpoint between "
                    f"the {ref_b} and the {ref_c}"
                ),
                "keywords": [relation, ref_b, ref_c],
                "keyword_match_threshold": 3,
                "_exact_relation": relation,
                "_obj_a": obj_a,
                "_obj_b": ref_b,
                "_obj_c": ref_c,
            }
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENTRY PER SINGOLO TASK
# ─────────────────────────────────────────────────────────────────────────────

def build_task_entry(task_name, env, obs, output_dir, task_idx,
                     max_pairs=6, nearby_dist=NEARBY_DIST_MAX):

    # ── frame ────────────────────────────────────────────────────────────
    frame_rgb = obs["agentview_image"]
    if frame_rgb.ndim == 4:
        frame_rgb = frame_rgb[0]
    frame_bgr = cv2.cvtColor(frame_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    frame_dir = os.path.join(output_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    frame_path = os.path.join(frame_dir, f"task{task_idx:02d}_first_frame.png")
    cv2.imwrite(frame_path, frame_bgr)

    # ── posizioni ─────────────────────────────────────────────────────────
    positions = get_object_positions(env)
    obj_names = sorted(positions.keys())
    if len(obj_names) < 2:
        print(f"  [WARN] task {task_idx}: solo {len(obj_names)} oggetti → skip")
        return None

    print(f"  [DEBUG] distanze inter-oggetto per task {task_idx}:")
    all_pairs_debug = []
    for i, a in enumerate(obj_names):
        for j, b in enumerate(obj_names):
            if i >= j:
                continue
            dist = float(np.linalg.norm(positions[a] - positions[b]))
            all_pairs_debug.append((dist, a.split("_1")[0], b.split("_1")[0]))
    all_pairs_debug.sort()
    for dist, a, b in all_pairs_debug[:10]:   # stampa le 10 più vicine
        print(f"    {dist:.3f}m  {a} ↔ {b}")

    pos_dir = os.path.join(output_dir, "positions")
    os.makedirs(pos_dir, exist_ok=True)
    with open(os.path.join(pos_dir, f"task{task_idx:02d}_positions.json"), "w") as f:
        json.dump({k: v.tolist() for k, v in positions.items()}, f, indent=2)

    # ── assi camera ───────────────────────────────────────────────────────
    cam_right, cam_up = get_camera_axes(env)

    # ── coppie VICINE ────────────────────────────────────────────────────
    # Filtra solo coppie con distanza euclidea 3D < nearby_dist
    # Ordina per distanza crescente (più vicine = più affidabili visivamente)
    nearby_pairs = []
    for i, a in enumerate(obj_names):
        for j, b in enumerate(obj_names):
            if i >= j:
                continue
            dist = float(np.linalg.norm(positions[a] - positions[b]))
            if dist < nearby_dist:
                nearby_pairs.append((dist, a, b))
    nearby_pairs.sort()  # crescente per distanza

    if not nearby_pairs:
        print(f"  [WARN] task {task_idx}: nessuna coppia entro {nearby_dist}m → skip")
        return None

    # ── genera domande per entrambi i frame ──────────────────────────────
    questions = []
    for q_idx, (dist, a, b) in enumerate(nearby_pairs[:max_pairs]):

        # Relazione nel world frame
        rel_w, _ = axis_relation_world(positions[a], positions[b])
        # Relazione nel camera frame
        rel_c, _ = axis_relation_camera(positions[a], positions[b],
                                         cam_right, cam_up)

        if rel_w == "next to" and rel_c == "next to":
            continue

        base_id = f"task{task_idx}_q{q_idx}"

        if rel_w != "next to":
            questions.append(make_type1_question(
                f"{base_id}_world_type1", a, b, rel_w, obj_names, "world"))
            questions.append(make_type1_question(
                f"{base_id}_world_inv_type1", b, a,
                inverse_relation(rel_w), obj_names, "world"))
            questions.append(make_type2_question(
                f"{base_id}_world_type2", a,
                f"{rel_w} the {b}", obj_names, "world"))

        if rel_c != "next to":
            questions.append(make_type1_question(
                f"{base_id}_cam_type1", a, b, rel_c, obj_names, "camera"))
            questions.append(make_type1_question(
                f"{base_id}_cam_inv_type1", b, a,
                inverse_relation(rel_c), obj_names, "camera"))
            questions.append(make_type2_question(
                f"{base_id}_cam_type2", a,
                f"{rel_c} the {b}", obj_names, "camera"))

        agreement = "✅ concordano" if rel_w == rel_c else f"⚠️  world={rel_w} cam={rel_c}"
        print(f"     [{q_idx}] {a.split('_1')[0]} → {b.split('_1')[0]}"
              f"  dist={dist:.3f}m  {agreement}")

    # ── domande extra per task/variation specifici ──────────────────────
    task_key = canonicalize_task_command(task_name)
    extra_queries = TASK_QUERY_INDEX.get(task_key, [])
    if extra_queries:
        pretty_objects = sorted({pretty_object_name(name) for name in obj_names})
        for eq_idx, cfg in enumerate(extra_queries):
            target = find_scene_object(obj_names, cfg["target"])
            ref1 = find_scene_object(obj_names, cfg["references"][0])
            ref2 = find_scene_object(obj_names, cfg["references"][1])

            if not target or not ref1 or not ref2:
                print(
                    f"  [WARN] task {task_idx}: oggetti non trovati per extra query "
                    f"{cfg['target']} / {cfg['references']}"
                )
                continue

            midpoint = (positions[ref1] + positions[ref2]) / 2.0
            rel_w, _ = axis_relation_world(positions[target], midpoint)
            rel_c, _ = axis_relation_camera(positions[target], midpoint,
                                            cam_right, cam_up)

            if rel_w != "next to":
                questions.append(make_type1_dual_ref_question(
                    f"task{task_idx}_extra{eq_idx}_world_type1_multi",
                    pretty_object_name(target),
                    pretty_object_name(ref1),
                    pretty_object_name(ref2),
                    rel_w,
                    pretty_objects,
                    "world",
                ))

            if rel_c != "next to":
                questions.append(make_type1_dual_ref_question(
                    f"task{task_idx}_extra{eq_idx}_cam_type1_multi",
                    pretty_object_name(target),
                    pretty_object_name(ref1),
                    pretty_object_name(ref2),
                    rel_c,
                    pretty_objects,
                    "camera",
                ))

            print(
                f"     [extra{eq_idx}] {pretty_object_name(target)} wrt "
                f"{pretty_object_name(ref1)}+{pretty_object_name(ref2)} "
                f"(world={rel_w}, camera={rel_c})"
            )

    return {
        "task_id": task_idx,
        "task_name": task_name,
        "first_frame_path": frame_path,
        "objects_in_scene": obj_names,
        "nearby_pairs_used": [
            {"a": a, "b": b, "dist_m": round(dist, 4)}
            for dist, a, b in nearby_pairs[:max_pairs]
        ],
        "questions": questions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--libero_path", required=True)
    p.add_argument("--output_dir",  default="./spatial_benchmark")
    p.add_argument("--suite",       default="libero_goal",
                   choices=["libero_goal", "libero_10", "libero_spatial", "libero_object"])
    p.add_argument("--max_pairs",   type=int, default=6)
    p.add_argument("--nearby_dist", type=float, default=NEARBY_DIST_MAX,
                   help="Distanza euclidea 3D massima (metri) per considerare due oggetti vicini")
    return p.parse_args()


def main():
    args = parse_args()

    if args.libero_path not in sys.path:
        sys.path.insert(0, args.libero_path)

    benchmark_dict = B.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    n_tasks = task_suite.n_tasks
    print(f"{n_tasks} tasks in {args.suite}  (nearby_dist={args.nearby_dist}m)")
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = []
    for task_idx in range(n_tasks):
        task = task_suite.get_task(task_idx)
        print(f"\n[{task_idx:02d}/{n_tasks}] {task.name}")

        env, task_description, _ = get_libero_env(task, "tinyvla", resolution=256)
        try:
            obs = env.reset()
            entry = build_task_entry(task.name, env, obs, args.output_dir,
                                     task_idx, args.max_pairs, args.nearby_dist)
            if entry:
                tasks.append(entry)
                n_world  = sum(1 for q in entry["questions"] if q["frame"] == "world")
                n_camera = sum(1 for q in entry["questions"] if q["frame"] == "camera")
                print(f"     → {len(entry['questions'])} domande totali"
                      f"  (world={n_world}, camera={n_camera})")
        except Exception as e:
            print(f"     [ERRORE] {e}")
        finally:
            env.close()

    out = os.path.join(args.output_dir, "spatial_benchmark.json")
    with open(out, "w") as f:
        json.dump({
            "metadata": {
                "suite": args.suite,
                "generator": "libero_spatial_benchmark_gen.py",
                "nearby_dist_max_m": args.nearby_dist,
                "frames": {
                    "world":  "MuJoCo world frame: x=right, y=forward(robot), z=up",
                    "camera": "agentview camera frame: u=img_right, v=cam_up(behind+), z=world_up"
                },
            },
            "tasks": tasks,
        }, f, indent=2, ensure_ascii=False)

    total_q = sum(len(t["questions"]) for t in tasks)
    print(f"\nDone → {len(tasks)} task, {total_q} domande totali")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
