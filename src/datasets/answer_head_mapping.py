# ------------------------------------------------------------------
# Multi-Head Answer Mapping for NuScenes-QA
#
# Maps the global 30-answer vocabulary to per-question-type heads.
# Each head has its own small answer space, eliminating cross-type
# confusion (e.g., a count question can never predict "car").
# ------------------------------------------------------------------

# Question type indices (must match QTYPE_MAP in nuscenes_qa.py)
QTYPE_NAMES = ['exist', 'count', 'object', 'status', 'comparison']
QTYPE_TO_IDX = {name: i for i, name in enumerate(QTYPE_NAMES)}

# Per-head answer vocabularies (global_answer_string → local_head_index)
HEAD_ANSWERS = {
    'exist': {
        'no': 0,
        'yes': 1,
    },
    'count': {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
        '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    },
    'object': {
        'barrier': 0,
        'bicycle': 1,
        'bus': 2,
        'car': 3,
        'construction vehicle': 4,
        'motorcycle': 5,
        'pedestrian': 6,
        'traffic cone': 7,
        'trailer': 8,
        'truck': 9,
    },
    'status': {
        'moving': 0,
        'not standing': 1,
        'parked': 2,
        'standing': 3,
        'stopped': 4,
        'with rider': 5,
        'without rider': 6,
    },
    'comparison': {
        'no': 0,
        'yes': 1,
    },
}

# Number of classes per head
HEAD_SIZES = {name: len(answers) for name, answers in HEAD_ANSWERS.items()}
# exist=2, count=11, object=10, status=7, comparison=2

# Reverse mapping: per-head local index → global answer string
HEAD_IDX_TO_ANS = {
    qtype: {idx: ans for ans, idx in answers.items()}
    for qtype, answers in HEAD_ANSWERS.items()
}

# Build global_answer_index → (qtype_idx, local_head_index) mapping
# This is computed lazily because it needs the global ans2ix dict
def build_global_to_local(ans2ix):
    """
    Given the global ans2ix dict (from answer_dict.json),
    returns a dict: global_ans_idx → (qtype_idx, local_head_idx)
    
    Also returns local_to_global: (qtype_idx, local_head_idx) → global_ans_idx
    """
    global_to_local = {}
    local_to_global = {}
    
    for qtype_name, answers in HEAD_ANSWERS.items():
        qtype_idx = QTYPE_TO_IDX[qtype_name]
        for ans_str, local_idx in answers.items():
            if ans_str in ans2ix:
                global_idx = ans2ix[ans_str]
                global_to_local[global_idx] = (qtype_idx, local_idx)
                local_to_global[(qtype_idx, local_idx)] = global_idx
    
    return global_to_local, local_to_global
