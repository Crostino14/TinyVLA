#!/usr/bin/env python3
import json
import shutil
import re

JSON_PATH = "/home/A.CARDAMONE7/outputs/vqa_test/spatial_benchmark.json"
OUTPUT_PATH = "/home/A.CARDAMONE7/outputs/vqa_test/spatial_benchmark_patched.json"
shutil.copy(JSON_PATH, JSON_PATH + ".bak2")

ALL_RELATION_OPTIONS = [
    "left of",
    "right of",
    "behind",
    "in front of",
    "above",
    "below",
    "next to",
]


def format_relation_options(options: list[str]) -> str:
    return " / ".join(options)


def humanize_name(value: str) -> str:
    """Convert internal MuJoCo-like object names to readable labels."""
    if not isinstance(value, str):
        return value
    x = value.strip()
    if not x:
        return x

    x = x.replace("_main", "")
    x = re.sub(r"_\d+\b", "", x)
    x = x.replace("_", " ")
    x = " ".join(x.split())
    if x in {"flat stove", "flat stove 1"}:
        return "stove"
    return x


def humanize_list(values):
    if not isinstance(values, list):
        return values
    return [humanize_name(v) if isinstance(v, str) else v for v in values]


def humanize_csv(values: str) -> str:
    if not isinstance(values, str):
        return values
    parts = [p.strip() for p in values.split(",") if p.strip()]
    return ", ".join(humanize_name(p) for p in parts)


def humanize_text(value: str) -> str:
    if not isinstance(value, str):
        return value
    return humanize_name(value)


with open(JSON_PATH, "r") as f:
    data = json.load(f)

patched = 0
for task in data.get("tasks", []):
    if isinstance(task.get("objects_in_scene"), list):
        task["objects_in_scene"] = humanize_list(task["objects_in_scene"])

    for pair in task.get("nearby_pairs_used", []):
        if isinstance(pair, dict):
            if "a" in pair:
                pair["a"] = humanize_name(pair["a"])
            if "b" in pair:
                pair["b"] = humanize_name(pair["b"])

    for q in task.get("questions", []):
        if "filled_prompt" in q:
            q["filled_prompt"] = humanize_text(q["filled_prompt"])

        # Humanize common object-bearing fields for every question type.
        if "target_object" in q:
            q["target_object"] = humanize_name(q["target_object"])
        if "reference_object" in q:
            q["reference_object"] = humanize_name(q["reference_object"])
        if isinstance(q.get("reference_objects"), list):
            q["reference_objects"] = humanize_list(q["reference_objects"])
        if "objects_list" in q:
            q["objects_list"] = humanize_csv(q["objects_list"])
        if "spatial_description" in q:
            q["spatial_description"] = humanize_name(q["spatial_description"])

        gt = q.get("ground_truth", {})
        if isinstance(gt, dict):
            t0 = gt.get("type0_object_listing")
            if isinstance(t0, dict) and isinstance(t0.get("expected_objects"), list):
                t0["expected_objects"] = humanize_list(t0["expected_objects"])

            t1 = gt.get("type1_spatial_relation")
            if isinstance(t1, dict):
                if "reference_answer" in t1:
                    t1["reference_answer"] = humanize_name(t1["reference_answer"])
                if isinstance(t1.get("keywords"), list):
                    t1["keywords"] = humanize_list(t1["keywords"])
                for key in ["_obj_a", "_obj_b", "_obj_c"]:
                    if key in t1:
                        t1[key] = humanize_name(t1[key])

            t1m = gt.get("type1_spatial_relation_multi_reference")
            if isinstance(t1m, dict):
                if "reference_answer" in t1m:
                    t1m["reference_answer"] = humanize_name(t1m["reference_answer"])
                if isinstance(t1m.get("keywords"), list):
                    t1m["keywords"] = humanize_list(t1m["keywords"])
                for key in ["_obj_a", "_obj_b", "_obj_c"]:
                    if key in t1m:
                        t1m[key] = humanize_name(t1m[key])

            t2 = gt.get("type2_object_identification")
            if isinstance(t2, dict):
                if isinstance(t2.get("valid_answers"), list):
                    t2["valid_answers"] = humanize_list(t2["valid_answers"])
                if "primary_answer" in t2:
                    t2["primary_answer"] = humanize_name(t2["primary_answer"])

        q_type = q.get("type")
        if q_type not in {
            "type1_spatial_relation",
            "type1_spatial_relation_multi_reference",
        }:
            continue

        tgt = q.get("target_object", "")
        obj_list = q.get("objects_list", "")
        if not tgt:
            print(f"SKIP {q.get('question_id', '?')}: missing target")
            continue

        rel_opts = format_relation_options(ALL_RELATION_OPTIONS)

        if q_type == "type1_spatial_relation":
            ref = q.get("reference_object", "")
            if not ref:
                print(f"SKIP {q.get('question_id', '?')}: missing reference_object")
                continue
            new_prompt = (
                f"The image shows a robotic manipulation scene. "
                f"Objects present: {obj_list}. "
                f"Complete this sentence using ONLY one of: {rel_opts}. "
                f"The {tgt} is ___ the {ref}. "
                f"Answer with ONLY the missing relation phrase."
            )
        else:
            refs = q.get("reference_objects", [])
            if not isinstance(refs, list) or len(refs) < 2:
                print(f"SKIP {q.get('question_id', '?')}: missing reference_objects")
                continue
            ref_a, ref_b = refs[0], refs[1]
            new_prompt = (
                f"The image shows a robotic manipulation scene. "
                f"Objects present: {obj_list}. "
                f"Complete this sentence using ONLY one of: {rel_opts}. "
                f"The {tgt} is ___ the midpoint between the {ref_a} and the {ref_b}. "
                f"Answer with ONLY the missing relation phrase."
            )

        q["filled_prompt"] = new_prompt
        patched += 1

with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(
    f"Done. Patched {patched} type1 prompts (single + multi reference). "
    f"Output: {OUTPUT_PATH}"
)
