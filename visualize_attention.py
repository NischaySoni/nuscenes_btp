import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import os

# ---------------------------------------------------
# Files
# ---------------------------------------------------

ATT_DIR = "outputs/attention"
PRED_FILE = "prediction_analysis.json"

MAX_PLOTS = 20
plot_count = 0

# ---------------------------------------------------
# Load predictions
# ---------------------------------------------------

if not os.path.exists(PRED_FILE):
    print("prediction_analysis.json not found. Run evaluation first.")
    exit()

with open(PRED_FILE) as f:
    results = json.load(f)

print("Total predictions:", len(results))

# ---------------------------------------------------
# Load attention files
# ---------------------------------------------------

if not os.path.exists(ATT_DIR):
    print("Attention folder not found. Run evaluation first.")
    exit()

att_files = sorted(os.listdir(ATT_DIR))

print("Attention files found:", len(att_files))

# ---------------------------------------------------
# Visualization
# ---------------------------------------------------

sample_idx = 0

for att_file in att_files:

    if plot_count >= MAX_PLOTS:
        break

    att_path = os.path.join(ATT_DIR, att_file)

    att = np.load(att_path)

    # shape: [batch_size, num_objects, glimpses]

    for b in range(att.shape[0]):

        if sample_idx >= len(results):
            break

        r = results[sample_idx]

        question = r["question"]
        gt = r["gt"]
        pred = r["pred"]

        # first glimpse
        attention = att[b,:,0]

        # normalize attention
        attention = attention / (attention.max() + 1e-8)

        # reshape tokens into grid
        grid_size = int(np.sqrt(len(attention)))

        if grid_size * grid_size != len(attention):
            # fallback if not perfect square
            grid_size = int(np.ceil(np.sqrt(len(attention))))
            padded = np.zeros(grid_size * grid_size)
            padded[:len(attention)] = attention
            attention = padded

        grid = attention.reshape(grid_size, grid_size)

        # ---------------------------------------------------
        # Plot
        # ---------------------------------------------------

        plt.figure(figsize=(6,6))

        plt.imshow(grid, cmap="hot")

        plt.colorbar()

        plt.title(
            f"GT={gt}  Pred={pred}\n{question}"
        )

        plt.xlabel("BEV X")

        plt.ylabel("BEV Y")

        plt.show(block=True)
        plot_count += 1
        plt.close()

        sample_idx += 1
        
