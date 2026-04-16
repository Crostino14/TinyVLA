import json

NAME_MAP = {
    "akita_black_bowl_1_main":         "bowl",
    "cream_cheese_1_main":             "cream cheese",
    "flat_stove_1_burner":             None,   # ← escludi
    "flat_stove_1_burner_plate":       None,   # ← escludi
    "flat_stove_1_button":             None,   # ← escludi
    "flat_stove_1_main":               "flat stove",
    "plate_1_main":                    "plate",
    "wine_bottle_1_main":              "wine bottle",
    "wine_rack_1_main":                "wine rack",
    "wooden_cabinet_1_cabinet_bottom": "cabinet bottom drawer",
    "wooden_cabinet_1_cabinet_middle": "cabinet middle drawer",
    "wooden_cabinet_1_cabinet_top":    "cabinet top drawer",
    "wooden_cabinet_1_main":           "wooden cabinet",
}

# Nomi MuJoCo da escludere completamente
EXCLUDED_MUJOCO = {k for k, v in NAME_MAP.items() if v is None}

def clean(text):
    for k, v in NAME_MAP.items():
        if v is not None:
            text = text.replace(k, v)
    return text

def question_involves_excluded(q):
    """Restituisce True se la domanda coinvolge un oggetto escluso."""
    for field in ["target_object", "reference_object", "spatial_description",
                  "_obj_a", "_obj_b"]:
        val = q.get(field, "") or ""
        if any(ex in val for ex in EXCLUDED_MUJOCO):
            return True
    gt = q.get("ground_truth", {})
    for cfg in gt.values():
        for field in ["_obj_a", "_obj_b", "primary_answer"]:
            if any(ex in str(cfg.get(field, "")) for ex in EXCLUDED_MUJOCO):
                return True
        for ans in cfg.get("valid_answers", []):
            if any(ex in ans for ex in EXCLUDED_MUJOCO):
                return True
    return False

with open("/home/A.CARDAMONE7/outputs/vqa_test/spatial_benchmark.json") as f:
    data = json.load(f)

for task in data["tasks"]:
    # Rimuovi oggetti esclusi da objects_in_scene
    task["objects_in_scene"] = [
        NAME_MAP.get(o, o)
        for o in task["objects_in_scene"]
        if NAME_MAP.get(o, o) is not None
    ]

    # Filtra domande che coinvolgono oggetti esclusi
    filtered_questions = []
    for q in task["questions"]:
        if question_involves_excluded(q):
            continue
        # Pulisci i nomi nei campi testo
        for field in ["filled_prompt", "objects_list", "target_object",
                      "reference_object", "spatial_description"]:
            if field in q:
                q[field] = clean(q[field])
        gt = q.get("ground_truth", {})
        for qtype in gt:
            cfg = gt[qtype]
            for field in ["reference_answer", "keywords", "_obj_a", "_obj_b",
                          "valid_answers", "primary_answer"]:
                if field in cfg:
                    v = cfg[field]
                    cfg[field] = [clean(x) for x in v] if isinstance(v, list) else clean(str(v))
        filtered_questions.append(q)

    removed = len(task["questions"]) - len(filtered_questions)
    if removed:
        print(f"  task {task['task_id']}: rimossa/e {removed} domanda/e con oggetti esclusi")
    task["questions"] = filtered_questions

with open("/home/A.CARDAMONE7/outputs/vqa_test/spatial_benchmark_clean.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Done → spatial_benchmark_clean.json")