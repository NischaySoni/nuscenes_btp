import json
import numpy as np
import matplotlib.pyplot as plt
import os

ATT_DIR = "outputs/attention"
PRED_FILE = "prediction_analysis.json"

# ------------------------------------------------
# Load predictions
# ------------------------------------------------

if not os.path.exists(PRED_FILE):
    print("prediction_analysis.json not found. Run evaluation first.")
    exit()

with open(PRED_FILE) as f:
    results = json.load(f)

print("Total samples:", len(results))

# ------------------------------------------------
# Find count errors
# ------------------------------------------------

count_errors = []

for i, r in enumerate(results):

    question = r["question"].lower()

    if "how many" in question:

        if r["gt"] != r["pred"]:

            count_errors.append((i, r))

print("Total count errors:", len(count_errors))

# ------------------------------------------------
# Load attention files
# ------------------------------------------------

if not os.path.exists(ATT_DIR):
    print("Attention folder not found.")
    exit()

att_files = sorted(os.listdir(ATT_DIR))

sample_idx = 0
error_idx = 0

for att_file in att_files:

    att_path = os.path.join(ATT_DIR, att_file)

    att = np.load(att_path)

    for b in range(att.shape[0]):

        if sample_idx >= len(results):
            break

        r = results[sample_idx]

        question = r["question"].lower()

        if "how many" in question and r["gt"] != r["pred"]:

            attention = att[b,:,0]   # first glimpse

            attention = attention / (attention.max() + 1e-8)

            num_tokens = len(attention)

            grid_size = int(np.sqrt(num_tokens))

            if grid_size * grid_size != num_tokens:
                grid_size = int(np.ceil(np.sqrt(num_tokens)))
                padded = np.zeros(grid_size * grid_size)
                padded[:num_tokens] = attention
                attention = padded

            grid = attention.reshape(grid_size, grid_size)

            plt.figure(figsize=(6,6))

            plt.imshow(grid, cmap="hot")

            plt.colorbar()

            plt.title(
                f"COUNT ERROR\nGT={r['gt']}  Pred={r['pred']}\n{r['question']}"
            )

            plt.xlabel("BEV X")

            plt.ylabel("BEV Y")

            plt.show()

            error_idx += 1

        sample_idx += 1

print("Visualized count errors:", error_idx)
