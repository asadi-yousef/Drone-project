import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# CONFIG SECTION (EDIT THIS)
# ==========================

VIDEO_PATH = "test.mp4"     # <-- path to your video
MAX_FRAMES = 0                     # 0 = process all frames
FRAME_SKIP = 2                    # use every Nth frame
ARUCO_ID = 0                       # marker id
ARUCO_CELL_CM = 2.5                # each cell length in cm
ARUCO_TOTAL_CELLS = 6              # 4 inner + 2 border (standard assumption)

# ==========================
# IMPORT YOUR ESTIMATOR
# ==========================

from motion_estimator import MotionEstimator


def build_rough_camera_matrix(frame_shape):
    h, w = frame_shape[:2]
    f = 0.9 * max(w, h)
    cx, cy = w / 2.0, h / 2.0
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]], dtype=np.float64)


def se3_to_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def T_inv(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def camera_center_from_marker_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    t = tvec.reshape(3, 1)
    return (-R.T @ t).reshape(3)


def detect_aruco_camera_center(frame, aruco_detector, K, dist, marker_length_m, wanted_id):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is None:
        return None

    ids = ids.flatten()
    idxs = np.where(ids == wanted_id)[0]
    if len(idxs) == 0:
        return None

    i = int(idxs[0])

    # ArUco returns corners as (1,4,2) for each marker in Python
    corner_2d = corners[i].reshape(4, 2)

    rvec, tvec = estimate_pose_square_marker_solvepnp(corner_2d, marker_length_m, K, dist)
    if rvec is None:
        return None

    return camera_center_from_marker_pose(rvec, tvec)


def estimate_pose_square_marker_solvepnp(corner_2d, marker_length_m, K, dist):
    """
    corner_2d: (4,2) image points in the same order as returned by ArUco (tl, tr, br, bl).
    Returns rvec, tvec for marker->camera:
      X_cam = R * X_marker + t
    """
    # 3D marker corners in marker frame (origin at marker center)
    half = marker_length_m / 2.0
    objp = np.array([
        [-half,  half, 0.0],  # top-left
        [ half,  half, 0.0],  # top-right
        [ half, -half, 0.0],  # bottom-right
        [-half, -half, 0.0],  # bottom-left
    ], dtype=np.float32)

    imgp = np.asarray(corner_2d, dtype=np.float32).reshape(-1, 2)

    # SOLVEPNP_IPPE_SQUARE is designed for square planar markers (good for ArUco).
    # If your OpenCV build doesn't support it, fallback to SOLVEPNP_ITERATIVE. :contentReference[oaicite:2]{index=2}
    try:
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except Exception:
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)

    if not ok:
        return None, None
    return rvec, tvec


def plot_trajectories(vo, aruco):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    if len(vo) > 0:
        ax.plot(vo[:, 0], vo[:, 1], vo[:, 2], label="VO (ORB + Essential)")
        ax.scatter(vo[0, 0], vo[0, 1], vo[0, 2], s=60)

    if len(aruco) > 0:
        ax.plot(aruco[:, 0], aruco[:, 1], aruco[:, 2], label="ArUco (metric)")
        ax.scatter(aruco[0, 0], aruco[0, 1], aruco[0, 2], s=60)

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# ==========================
# MAIN PIPELINE
# ==========================

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(VIDEO_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

# 1. Read the very first frame just to get camera matrix dimensions
ok, first_frame_ref = cap.read()
if not ok:
    raise RuntimeError("Could not read video")

K = build_rough_camera_matrix(first_frame_ref.shape)
dist = np.zeros((5, 1))

estimator = MotionEstimator(camera_matrix=K)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

marker_length_m = (ARUCO_TOTAL_CELLS * ARUCO_CELL_CM) / 100.0

# Reset video to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Variables for the main loop
T_w_c = np.eye(4)
vo_positions = []
aruco_positions = []

prev_frame = None
prev_aruco = None
last_scale = 1.0   # Default scale if we can't find ArUco immediately

frame_counter = 0
used_frames = 0

print("Initializing: Searching for first ArUco marker to align origins...")

# --- PHASE 1: INITIALIZATION LOOP ---
# We read frames until we find the first ArUco marker.
# This sets the starting point for both VO and ArUco paths.
start_frame_idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Error: Video ended before finding any ArUco marker.")
        exit()
        
    aruco_pos = detect_aruco_camera_center(frame, aruco_detector, K, dist, marker_length_m, wanted_id=ARUCO_ID)
    
    if aruco_pos is not None:
        print(f"Locked on ArUco at frame {start_frame_idx}. Starting VO...")
        
        # Initialize VO start point to match the specific ArUco location
        T_w_c[:3, 3] = aruco_pos
        
        # Save first points
        vo_positions.append(aruco_pos)
        aruco_positions.append(aruco_pos)
        
        # Set previous state for the next iteration
        prev_frame = frame
        prev_aruco = aruco_pos
        
        frame_counter = start_frame_idx
        break
        
    start_frame_idx += 1


# --- PHASE 2: MAIN TRACKING LOOP ---
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_counter += 1
    
    # Skip frames if configured
    if FRAME_SKIP > 1 and (frame_counter - start_frame_idx) % FRAME_SKIP != 0:
        continue

    # 1. Detect ArUco (Ground Truth)
    aruco_pos = detect_aruco_camera_center(frame, aruco_detector, K, dist, marker_length_m, wanted_id=ARUCO_ID)
    
    if aruco_pos is not None:
        aruco_positions.append(aruco_pos)
    else:
        # If we lose the marker, we just visualize a gap in the orange line, 
        # or you could append the last known position (optional)
        pass

    # 2. Estimate Visual Odometry (VO)
    try:
        result = estimator.estimate_motion(prev_frame, frame)
    except Exception as e:
        print(f"VO Failed at frame {frame_counter}: {e}")
        prev_frame = frame
        prev_aruco = aruco_pos if aruco_pos is not None else prev_aruco
        continue

    R = result["rotation_matrix"]
    t = result["translation_vector"].reshape(3, 1)
    
    # Normalize translation direction (since monocular VO has no scale)
    t_dir = t / (np.linalg.norm(t) + 1e-9)

    # 3. Calculate Scale intelligently
    # Only update scale if the drone actually moved enough to get a clean reading.
    # Threshold: 2 cm (0.02 meters)
    if prev_aruco is not None and aruco_pos is not None:
        dist_moved = np.linalg.norm(aruco_pos - prev_aruco)
        if dist_moved > 0.02:
            last_scale = dist_moved

    # Apply the scale (either fresh or carried over)
    t_scaled = last_scale * t_dir

    # 4. Update Pose
    # FIX: Use simple multiplication (T_rel), NOT inverse
    T_rel = se3_to_T(R, t_scaled)
    T_w_c = T_w_c @ T_rel

    vo_positions.append(T_w_c[:3, 3].copy())

    # 5. Prepare for next step
    prev_frame = frame
    
    # Only update prev_aruco if we actually saw it this frame
    if aruco_pos is not None:
        prev_aruco = aruco_pos
        
    used_frames += 1
    if MAX_FRAMES > 0 and used_frames >= MAX_FRAMES:
        break

cap.release()

vo_positions = np.asarray(vo_positions)
aruco_positions = np.asarray(aruco_positions)

print("Processing Complete.")
print(f"VO points: {len(vo_positions)}")
print(f"ArUco points: {len(aruco_positions)}")

plot_trajectories(vo_positions, aruco_positions)
