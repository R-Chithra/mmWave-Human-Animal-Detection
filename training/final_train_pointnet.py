import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report

# ===================== CONFIG =====================

# Set this to your prepared fixed-N dataset directory
# Structure expected:
# DATA_DIR/
#    ├── empty/
#    ├── human/
#    └── animal/
DATA_DIR = r"PATH_TO_fixed_n_data"

# Set this to where you want to save the trained model
SAVE_PATH = r"PATH_TO_SAVE_final_pointnet_tnet.pth"

BATCH_SIZE = 16
LR = 1e-3
MAX_EPOCHS = 50
PATIENCE = 5
NUM_CLASSES = 3

CLASS_MAP = {
    "empty": 0,
    "human": 1,
    "animal": 2
}
CLASS_NAMES = ["empty", "human", "animal"]
# =================================================


# ===================== DATASET =====================
class RadarPointCloudDataset(Dataset):
    def __init__(self, root_dir):
        if not os.path.exists(root_dir):
            raise ValueError("DATA_DIR invalid. Please set correct dataset path.")

        self.samples = []
        for cls, label in CLASS_MAP.items():
            cls_dir = os.path.join(root_dir, cls)

            if not os.path.exists(cls_dir):
                raise ValueError(f"Missing class folder: {cls_dir}")

            for f in os.listdir(cls_dir):
                if f.endswith(".npy"):
                    self.samples.append((os.path.join(cls_dir, f), label))

        print(f"Total samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        points = np.load(path).astype(np.float32)   # (256,5)
        return torch.tensor(points), torch.tensor(label)


# ===================== INPUT T-NET =====================
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


# ===================== POINTNET =====================
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
        x = x.transpose(1, 2)          # (B, 5, N)

        trans = self.input_tnet(x)
        x = torch.bmm(trans, x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.max(x, 2)[0]         # (B, 256)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ===================== EARLY STOPPING =====================
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# ===================== MAIN TRAINING =====================
def main():

    if not os.path.exists(DATA_DIR):
        print("❌ DATA_DIR invalid. Please set correct dataset path.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = RadarPointCloudDataset(DATA_DIR)

    # ---- Train / Validation split ----
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = PointNetClassifier(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    early_stopper = EarlyStopping(patience=PATIENCE)

    # ---- Training loop ----
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0

        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(device), labels.to(device)
                outputs = model(points)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if early_stopper.step(val_loss, model):
            print("Early stopping triggered.")
            break

    # ---- Restore best model ----
    model.load_state_dict(early_stopper.best_state)

    # ---- Validation Metrics ----
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for points, labels in val_loader:
            points = points.to(device)
            outputs = model(points)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100
    print(f"\nValidation Accuracy: {accuracy:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # ---- Save final model ----
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_val_loss": early_stopper.best_loss,
        "classes": CLASS_NAMES
    }, SAVE_PATH)

    print(f"\nFinal model saved at:\n{SAVE_PATH}")
    print("Training + Validation evaluation COMPLETE.")


if __name__ == "__main__":
    main()
