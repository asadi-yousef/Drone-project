import os
import glob
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

# IMPORTANT:
# Save your pasted estimator code as: motion_estimator.py
# so this import works.
from motion_estimator import MotionEstimator

# =========================
# USER CONFIGURATION
# =========================

FRAMES_DIR = "images"   # folder with 000000.png, 000001.png, ...
GT_PATH = "camera_poses.txt"  # ground-truth poses file

START_IDX = 0            # first frame index to use
END_IDX = None           # last frame index (None = use all)

USE_GT_SCALE = False      # False = direction-only VO (no scale)
VERBOSE = True           # print debug info
MAX_PAIRS = None       # limit number of pairs processed (None = all)

def load_gt_poses(gt_path: str):
    """
    Loads GT poses from a whitespace/csv-like text file.

    Expected columns (at least):
      frame x y z roll pitch yaw

    Returns:
      frames (N,), positions (N,3)
    """
    # Try whitespace-delimited with header
    data = np.genfromtxt(gt_path, dtype=None, encoding=None, names=True)

    # Normalize common column name variants
    colnames = set(data.dtype.names)

    def pick(name_options):
        for n in name_options:
            if n in colnames:
                return n
        raise ValueError(f"Could not find any of columns: {name_options} in {colnames}")

    c_frame = pick(["frame", "Frame", "idx", "index"])
    c_x = pick(["x", "X"])
    c_y = pick(["y", "Y"])
    c_z = pick(["z", "Z"])

    frames = np.asarray(data[c_frame], dtype=int)
    positions = np.vstack([data[c_x], data[c_y], data[c_z]]).T.astype(np.float64)
    return frames, positions


def list_frames(frames_dir: str, ext: str = "png"):
    files = sorted(glob.glob(os.path.join(frames_dir, f"*.{ext}")))
    if len(files) < 2:
        raise ValueError(f"Need at least 2 frames in {frames_dir}, found {len(files)}")
    return files


def build_camera_matrix_from_first_frame(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    h, w = img.shape[:2]
    f = 0.9 * max(w, h)  # same rough guess style you used
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float64)
    return K


def integrate_trajectory(estimator: MotionEstimator, frame_files, gt_frames, gt_positions,
                         use_gt_scale=True, max_pairs=None, verbose=False):
    """
    Integrates relative motions into a global camera trajectory.

    Assumption (standard OpenCV convention):
      estimator returns relative pose cam1 -> cam2 such that: X2 = R * X1 + t
    """

    gt_map = {int(f): gt_positions[i] for i, f in enumerate(gt_frames)}

    # World-from-camera rotation and camera position in world
    R_wc = np.eye(3, dtype=np.float64)
    p_w  = np.zeros(3, dtype=np.float64)

    est_positions = [p_w.copy()]

    num_pairs = len(frame_files) - 1
    
    if max_pairs is not None:
        num_pairs = min(num_pairs, int(max_pairs))

    for i in range(num_pairs):
        f1 = frame_files[i]
        f2 = frame_files[i + 1]

        img1 = cv2.imread(f1)
        img2 = cv2.imread(f2)
        if img1 is None or img2 is None:
            if verbose:
                print(f"[WARN] Skipping unreadable pair: {f1}, {f2}")
            est_positions.append(p_w.copy())
            continue

        try:
            result = estimator.estimate_motion(img1, img2)
        except Exception as e:
            if verbose:
                print(f"[WARN] Motion estimation failed on pair {i}: {e}")
            est_positions.append(p_w.copy())
            continue

        # Relative motion cam1 -> cam2
        R_rel = result["rotation_matrix"].astype(np.float64)
        t_rel = result["translation_vector"].reshape(3).astype(np.float64)

        # --- Scale using GT step length (optional but needed for metric comparison) ---
        scale = 1.0
        if use_gt_scale:
            idx1 = int(os.path.splitext(os.path.basename(f1))[0])
            idx2 = int(os.path.splitext(os.path.basename(f2))[0])
            if idx1 in gt_map and idx2 in gt_map:
                gt_step = np.linalg.norm(gt_map[idx2] - gt_map[idx1])
                scale = float(gt_step) if gt_step > 1e-12 else 1.0

        t_rel = t_rel * scale

        # --- Heuristic to reduce sign flip in mostly-forward motion sequences ---
        # If your camera "forward" is +Z (common OpenCV camera coords), enforce it.
        # If this makes it worse, delete these 2 lines.
        if t_rel[2] < 0:
            t_rel = -t_rel

        # --- Compose into world ---
        # IMPORTANT: t_rel is in camera-1 coordinates -> rotate it into world
        p_w = p_w + (R_wc @ t_rel)

        # Update world-from-camera rotation
        R_wc = R_wc @ R_rel

        est_positions.append(p_w.copy())

        if verbose and (i % 20 == 0):
            print(f"[INFO] pair {i}/{num_pairs}: "
                  f"inliers={result.get('num_inliers')} matches={result.get('total_matches')} "
                  f"scale={scale:.6f}")

    est_positions = np.vstack(est_positions)

    # --- Build comparable GT segment aligned to processed frames ---
    gt_used = []
    for k in range(num_pairs + 1):
        idx = int(os.path.splitext(os.path.basename(frame_files[k]))[0])
        if idx in gt_map:
            gt_used.append(gt_map[idx])
        else:
            gt_used.append(gt_used[-1] if gt_used else np.zeros(3, dtype=np.float64))
    gt_used = np.vstack(gt_used)

    # Align both so they start at origin
    gt_used = gt_used - gt_used[0]
    est_positions = est_positions - est_positions[0]

    return est_positions, gt_used


def plot_3d(gt_xyz, est_xyz, title="Trajectory: Estimated vs Ground Truth"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="Ground Truth")
    ax.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2], label="Estimated (VO)")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    gt_frames, gt_positions = load_gt_poses(GT_PATH)
    frame_files = list_frames(FRAMES_DIR, ext="png")
    print("GT rows:", len(gt_frames))
    print("GT frame min/max:", int(np.min(gt_frames)), int(np.max(gt_frames)))
    print("GT first/last positions:", gt_positions[0], gt_positions[-1])

    def idx_from_path(p):
      return int(os.path.splitext(os.path.basename(p))[0])

    img_idxs = np.array([idx_from_path(f) for f in frame_files], dtype=int)
    print("IMG count:", len(frame_files))
    print("IMG idx min/max:", int(img_idxs.min()), int(img_idxs.max()))
    gt_set = set(map(int, gt_frames))
    missing = [i for i in img_idxs[:200] if i not in gt_set]  # sample first 200
    print("Missing GT (sample first 200):", missing[:20], "count:", len(missing))

    K = build_camera_matrix_from_first_frame(frame_files[0])
    estimator = MotionEstimator(camera_matrix=K)

    est_xyz, gt_xyz = integrate_trajectory(
        estimator,
        frame_files,
        gt_frames,
        gt_positions,
        use_gt_scale=(not USE_GT_SCALE),
        max_pairs=MAX_PAIRS,
        verbose=VERBOSE
    )

    # Save for later inspection
    out_est = os.path.join(FRAMES_DIR, "estimated_trajectory_xyz.txt")
    out_gt = os.path.join(FRAMES_DIR, "gt_trajectory_xyz_aligned.txt")
    np.savetxt(out_est, est_xyz, fmt="%.9f", header="x y z")
    np.savetxt(out_gt, gt_xyz, fmt="%.9f", header="x y z")
    print(f"[OK] Saved:\n  {out_est}\n  {out_gt}")

    plot_3d(gt_xyz, est_xyz)


if __name__ == "__main__":
    main()
