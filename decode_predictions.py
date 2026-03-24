import json

with open("prediction_analysis.json") as f:
    data = json.load(f)

ans2ix, ix2ans = json.load(open("src/datasets/answer_dict.json"))

for d in data[:20]:
    gt_word = ix2ans[str(d["gt"])]
    pred_word = ix2ans[str(d["pred"])]

    print("\nQuestion: ", d["question"])
    print("GT: ", gt_word)
    print("Pred: ", pred_word)