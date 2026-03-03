import json
import os
import numpy as np

# ================= CONFIG =================

# Set this to your replay JSON directory
# Expected structure:
# RAW_DATA_DIR/
#     ├── human/
#     ├── animal/
#     └── empty/
RAW_DATA_DIR = r"PATH_TO_Replay_Data"

# Set this to where windowed data should be saved
OUT_DATA_DIR = r"PATH_TO_windowed_data"

WINDOW_SIZE = 10
USE_IDXS = [0, 1, 2, 3, 4]   # x, y, z, velocity, SNR
# =========================================


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_points_from_frame(frame):
    frameData = frame.get("frameData", {})

    if "pointCloud" not in frameData:
        return []   # no points in this frame

    pc = frameData["pointCloud"]
    return [[p[i] for i in USE_IDXS] for p in pc]


def window_one_json(json_path):
    """Convert one replay.json → list of window point clouds"""

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        print(f"⚠ Failed to read {json_path}")
        return []

    frames = data.get("data", [])
    windows = []

    for start in range(0, len(frames), WINDOW_SIZE):
        end = start + WINDOW_SIZE
        if end > len(frames):
            break

        all_points = []
        for frame in frames[start:end]:
            all_points.extend(extract_points_from_frame(frame))

        windows.append(np.array(all_points, dtype=np.float32))

    return windows


def process_class_folder(class_name):
    in_dir = os.path.join(RAW_DATA_DIR, class_name)
    out_dir = os.path.join(OUT_DATA_DIR, class_name)

    if not os.path.exists(in_dir):
        print(f"⚠ Skipping {class_name} (folder not found: {in_dir})")
        return

    ensure_dir(out_dir)

    json_files = [f for f in os.listdir(in_dir) if f.endswith(".json")]

    print(f"\nProcessing class: {class_name}")
    print(f"Found {len(json_files)} JSON files")

    window_count = 0

    for jf in json_files:
        json_path = os.path.join(in_dir, jf)
        windows = window_one_json(json_path)

        base = os.path.splitext(jf)[0]

        for i, w in enumerate(windows):
            out_path = os.path.join(out_dir, f"{base}_win{i}.npy")
            np.save(out_path, w)
            window_count += 1

    print(f"Saved {window_count} windows for class '{class_name}'")


def main():

    if not os.path.exists(RAW_DATA_DIR):
        print("❌ RAW_DATA_DIR invalid. Please set correct path.")
        return

    ensure_dir(OUT_DATA_DIR)

    for cls in ["human", "animal", "empty"]:
        process_class_folder(cls)

    print("\nWindowing complete.")


if __name__ == "__main__":
    main()
