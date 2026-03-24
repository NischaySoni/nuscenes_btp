import torch
from torch.utils.data import DataLoader
from datasets.nuscenes_bev_dataset import NuScenesBEVQADataset
from models.mcan import MCAN   # existing MCAN implementation
from tqdm import tqdm

# ================= CONFIG =================

QA_JSON = "data/nuscenes_qa_train.json"
FEAT_DIR = "bev_features"
MEAN_PATH = "stats/bev_mean.npy"
STD_PATH = "stats/bev_std.npy"

BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= DATA =================

dataset = NuScenesBEVQADataset(
    qa_json=QA_JSON,
    feat_dir=FEAT_DIR,
    mean_path=MEAN_PATH,
    std_path=STD_PATH
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

# ================= MODEL =================

model = MCAN(
    visual_dim=69,      # 🔥 IMPORTANT
    num_objects=80
)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# ================= TRAIN =================

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        visual_feats = batch["visual_feats"].to(DEVICE)
        questions = batch["question"]
        answers = batch["answer"].to(DEVICE)

        logits = model(visual_feats, questions)
        loss = criterion(logits, answers)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

torch.save(model.state_dict(), "mcan_image_radar.pth")
print("✅ Training complete")
