import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# =========================================================
# PATHS  (LOCAL PATHS REMOVED – USER MUST SET THESE)
# =========================================================

# Set this to the folder where Industrial Visualizer saves JSON files
JSON_FOLDER = r"PATH_TO_JSON_FOLDER"

# Set this to your trained model file
MODEL_PATH  = r"PATH_TO_final_pointnet_tnet.pth"

# =========================================================
# PARAMETERS (MATCH TRAINING)
# =========================================================
WINDOW_SIZE = 10
STEP_SIZE   = 10
N_POINTS    = 256

CLASS_NAMES = ["EMPTY", "HUMAN", "ANIMAL"]

# Guard logic parameters
HISTORY_LEN = 5
CONF_THRESH = 0.60

# =========================================================
# POINTNET + INPUT T-NET (MATCH TRAINING EXACTLY)
# =========================================================
class InputTNet(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, k * k)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k)
        x = x + identity.repeat(B, 1)
        return x.view(B, self.k, self.k)


class PointNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.input_tnet = InputTNet(k=5)
        self.conv1 = nn.Conv1d(5, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)          # (B,5,N)
        trans = self.input_tnet(x)
        x = torch.bmm(trans, x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# =========================================================
# LOAD MODEL
# =========================================================
device = torch.device("cpu")

if not os.path.exists(MODEL_PATH):
    print("❌ MODEL_PATH invalid. Please set correct path.")
    exit()

model = PointNetClassifier(num_classes=3).to(device)
ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("✅ PointNet model loaded. Live system running.\n", flush=True)

# =========================================================
# SAFE JSON → FRAMES LOADER
# =========================================================
def load_frames(json_path, retries=10, delay=0.2):
    for _ in range(retries):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            break
        except json.JSONDecodeError:
            time.sleep(delay)
        except Exception:
            return []
    else:
        return []

    frames = []
    for entry in data.get("data", []):
        pc = entry.get("frameData", {}).get("pointCloud", [])
        if pc:
            frames.append(pc)
    return frames

# =========================================================
# FIXED-N SAMPLING (256)
# =========================================================
def fix_n_points(points, n=N_POINTS):
    points = np.array(points, dtype=np.float32)

    if len(points) == 0:
        return np.zeros((n, 5), dtype=np.float32)

    if len(points) >= n:
        idx = np.random.choice(len(points), n, replace=False)
        return points[idx, :5]

    pad = np.zeros((n - len(points), 5), dtype=np.float32)
    return np.vstack((points[:, :5], pad))

# =========================================================
# MAIN REAL-TIME LOOP
# =========================================================
processed_files = set()
pred_history = deque(maxlen=HISTORY_LEN)

if not os.path.exists(JSON_FOLDER):
    print("❌ JSON_FOLDER invalid. Please set correct path.")
    exit()

while True:
    try:
        json_files = sorted(f for f in os.listdir(JSON_FOLDER) if f.endswith(".json"))

        for jf in json_files:
            json_path = os.path.join(JSON_FOLDER, jf)

            if json_path in processed_files:
                continue

            frames = load_frames(json_path)
            if len(frames) < WINDOW_SIZE:
                continue

            window_preds = []

            for start in range(0, len(frames) - WINDOW_SIZE + 1, STEP_SIZE):
                window = frames[start:start + WINDOW_SIZE]

                all_points = []
                for frame in window:
                    for p in frame:
                        all_points.append(p[:5])   # x,y,z,v,snr

                cloud = fix_n_points(all_points)
                cloud = torch.tensor(cloud).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(cloud).squeeze()
                    probs = torch.softmax(logits, dim=0)

                pred_class = torch.argmax(probs).item()
                confidence = probs[pred_class].item()

                pred_history.append(pred_class)

                final_label = CLASS_NAMES[pred_class]

                # ================= GUARD LOGIC =================
                if final_label == "ANIMAL":
                    human_count = pred_history.count(CLASS_NAMES.index("HUMAN"))

                    if human_count >= 3:
                        final_label = "HUMAN (guarded)"
                    elif confidence < CONF_THRESH:
                        final_label = "ANIMAL / HUMAN (uncertain)"
                # =================================================

                window_preds.append(final_label)

            if window_preds:
                final_decision = max(set(window_preds), key=window_preds.count)
                print(f"[{jf}] → {final_decision}", flush=True)
                processed_files.add(json_path)

    except Exception as e:
        print("⚠ Runtime issue, continuing:", str(e), flush=True)

    time.sleep(1)
