import os
import numpy as np
import random

# ===== CONFIG =====

# Set this to your windowed data directory
# Expected structure:
# WINDOWED_DIR/
#     ├── empty/
#     ├── human/
#     └── animal/
WINDOWED_DIR = r"PATH_TO_windowed_data"

# Set this to where you want fixed-N data saved
OUT_DIR = r"PATH_TO_fixed_n_data"

N_POINTS = 256

CLASSES = {
    "empty": 0,
    "human": 1,
    "animal": 2
}
# ==================


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def fix_n_points(points, n=N_POINTS):
    """
    points: (Ni, 5) OR (0, ?)
    returns: (N, 5)
    """
    # Case 1: completely empty window
    if points.size == 0:
        return np.zeros((n, 5), dtype=np.float32)

    Ni = points.shape[0]

    # Case 2: enough points
    if Ni >= n:
        idx = np.random.choice(Ni, n, replace=False)
        return points[idx]

    # Case 3: pad
    pad = np.zeros((n - Ni, 5), dtype=np.float32)
    return np.vstack((points, pad))


def process_class(cls_name, label):
    in_dir = os.path.join(WINDOWED_DIR, cls_name)
    out_dir = os.path.join(OUT_DIR, cls_name)

    if not os.path.exists(in_dir):
        print(f"⚠ Skipping {cls_name} (folder not found: {in_dir})")
        return

    ensure_dir(out_dir)

    files = [f for f in os.listdir(in_dir) if f.endswith(".npy")]
    print(f"\nProcessing {cls_name}: {len(files)} windows")

    for f in files:
        path = os.path.join(in_dir, f)
        points = np.load(path)           # (Ni, 5)
        fixed = fix_n_points(points)     # (256, 5)

        out_path = os.path.join(out_dir, f)
        np.save(out_path, fixed)

    print(f"Saved fixed-N samples for {cls_name}")


def main():
    if not os.path.exists(WINDOWED_DIR):
        print("❌ WINDOWED_DIR invalid. Please set correct path.")
        return

    ensure_dir(OUT_DIR)

    for cls, label in CLASSES.items():
        process_class(cls, label)

    print("\nFixed-N preparation complete.")


if __name__ == "__main__":
    main()
