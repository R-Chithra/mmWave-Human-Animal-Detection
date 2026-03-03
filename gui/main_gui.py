import sys
import os
import time
import json
import cv2
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox
)
from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap

# =========================================================
# CONFIG  (LOCAL PATHS REMOVED – USER MUST SET THESE)
# =========================================================

# Set this to the folder where Industrial Visualizer saves JSON files
BASE_JSON_PATH = r"YOUR_JSON_OUTPUT_FOLDER_PATH_HERE"

# Optional: If you want auto-launch of Industrial Visualizer, set full path here.
# Otherwise leave as empty string "" and it will not launch.
TI_VISUALIZER_EXE = r""  # Example: r"C:\ti\radar_toolbox\...\Industrial_Visualizer.exe"

# Set this to your trained model file path
MODEL_PATH = r"PATH_TO_final_pointnet_tnet.pth"

WINDOW_SIZE = 10
STEP_SIZE = 10
N_POINTS = 256

CLASS_NAMES = ["EMPTY", "HUMAN", "ANIMAL"]
CONF_THRESH = 0.60
HISTORY_LEN = 5

# =========================================================
# POINTNET (EXACT MATCH)
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
        ident = torch.eye(self.k, device=x.device).view(1, self.k * self.k)
        x = x + ident.repeat(B, 1)
        return x.view(B, self.k, self.k)

class PointNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_tnet = InputTNet(k=5)
        self.conv1 = nn.Conv1d(5, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = x.transpose(1, 2)
        trans = self.input_tnet(x)
        x = torch.bmm(trans, x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# =========================================================
# UTILS
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

def load_frames(json_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return []
    frames = []
    for entry in data.get("data", []):
        pc = entry.get("frameData", {}).get("pointCloud", [])
        if pc:
            frames.append(pc)
    return frames

# =========================================================
# INFERENCE THREAD
# =========================================================
class InferenceThread(QThread):
    update_signal = Signal(float, float, float, str)

    def __init__(self, json_folder):
        super().__init__()
        self.json_folder = json_folder
        self.running = True

    def run(self):
        device = torch.device("cpu")
        model = PointNetClassifier().to(device)

        if not os.path.exists(MODEL_PATH):
            print("MODEL PATH INVALID. Set MODEL_PATH correctly.")
            return

        ckpt = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        pred_history = deque(maxlen=HISTORY_LEN)
        processed = set()

        while self.running:
            if not os.path.exists(self.json_folder):
                time.sleep(1)
                continue

            files = sorted(f for f in os.listdir(self.json_folder) if f.endswith(".json"))
            for jf in files:
                path = os.path.join(self.json_folder, jf)
                if path in processed:
                    continue

                frames = load_frames(path)
                if len(frames) < WINDOW_SIZE:
                    continue

                window_preds = []
                probs_accum = []

                for i in range(0, len(frames) - WINDOW_SIZE + 1, STEP_SIZE):
                    window = frames[i:i + WINDOW_SIZE]
                    pts = []
                    for fr in window:
                        for p in fr:
                            pts.append(p[:5])

                    cloud = fix_n_points(pts)
                    cloud = torch.tensor(cloud).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(cloud).squeeze()
                        probs = torch.softmax(logits, dim=0).cpu().numpy()

                    pred = int(np.argmax(probs))
                    conf = probs[pred]

                    pred_history.append(pred)
                    final_label = CLASS_NAMES[pred]

                    if final_label == "ANIMAL":
                        if pred_history.count(CLASS_NAMES.index("HUMAN")) >= 3:
                            final_label = "HUMAN (guarded)"
                        elif conf < CONF_THRESH:
                            final_label = "ANIMAL / HUMAN (uncertain)"

                    window_preds.append(final_label)
                    probs_accum.append(probs)

                if window_preds:
                    final_decision = max(set(window_preds), key=window_preds.count)
                    avg_probs = np.mean(probs_accum, axis=0)
                    self.update_signal.emit(
                        avg_probs[1], avg_probs[2], avg_probs[0], final_decision
                    )
                    processed.add(path)

            time.sleep(1)

    def stop(self):
        self.running = False

# =========================================================
# CAMERA WIDGET
# =========================================================
class CameraWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.setAlignment(Qt.AlignCenter)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(img).scaled(self.width(), self.height(), Qt.KeepAspectRatio))

    def release(self):
        self.cap.release()

# =========================================================
# GUI
# =========================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mmWave Human–Animal Detection System")
        self.resize(1200, 800)

        # Launch Visualizer only if path is provided
        if TI_VISUALIZER_EXE and os.path.exists(TI_VISUALIZER_EXE):
            subprocess.Popen([TI_VISUALIZER_EXE], shell=False)

        self.thread = None
        layout = QVBoxLayout(self)

        cam_box = QGroupBox("Live Camera")
        cam_layout = QVBoxLayout()
        self.cam = CameraWidget()
        cam_layout.addWidget(self.cam)
        cam_box.setLayout(cam_layout)
        layout.addWidget(cam_box, stretch=3)

        bottom = QHBoxLayout()

        path_box = QGroupBox("Invoke JSON Path")
        path_layout = QVBoxLayout()
        self.base_lbl = QLabel(BASE_JSON_PATH)
        self.base_lbl.setWordWrap(True)
        self.ts_edit = QLineEdit("ENTER_SUBFOLDER_NAME")
        self.run_btn = QPushButton("RUN")
        self.run_btn.clicked.connect(self.start)
        path_layout.addWidget(self.base_lbl)
        path_layout.addWidget(self.ts_edit)
        path_layout.addWidget(self.run_btn)
        path_box.setLayout(path_layout)
        bottom.addWidget(path_box, 1)

        pred_box = QGroupBox("Prediction")
        pred_layout = QVBoxLayout()
        self.h_lbl = QLabel("HUMAN : --")
        self.a_lbl = QLabel("ANIMAL : --")
        self.e_lbl = QLabel("EMPTY : --")
        self.f_lbl = QLabel("FINAL : ---")
        self.f_lbl.setStyleSheet("font-size:28px;font-weight:bold")
        pred_layout.addWidget(self.h_lbl)
        pred_layout.addWidget(self.a_lbl)
        pred_layout.addWidget(self.e_lbl)
        pred_layout.addWidget(self.f_lbl)
        pred_box.setLayout(pred_layout)
        bottom.addWidget(pred_box, 3)

        layout.addLayout(bottom, stretch=2)

    def start(self):
        if self.thread:
            return
        folder = os.path.join(BASE_JSON_PATH, self.ts_edit.text())
        if not os.path.exists(folder):
            self.f_lbl.setText("FINAL : INVALID PATH")
            return
        self.thread = InferenceThread(folder)
        self.thread.update_signal.connect(self.update_pred)
        self.thread.start()

    def update_pred(self, h, a, e, f):
        self.h_lbl.setText(f"HUMAN : {h:.2f}")
        self.a_lbl.setText(f"ANIMAL : {a:.2f}")
        self.e_lbl.setText(f"EMPTY : {e:.2f}")
        self.f_lbl.setText(f"FINAL : {f}")

    def closeEvent(self, e):
        if self.thread:
            self.thread.stop()
            self.thread.quit()
            self.thread.wait()
        self.cam.release()
        e.accept()

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
