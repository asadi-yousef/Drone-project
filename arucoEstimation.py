import cv2
import numpy as np

# ============================================================
# SE(3) helpers
# ============================================================
def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert (rvec,tvec) to 4x4 homogeneous transform using Rodrigues."""
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T

def invert_T(T: np.ndarray) -> np.ndarray:
    """Inverse of SE(3) transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv

def rotation_angle_deg(R: np.ndarray) -> float:
    """Angle of rotation (deg) from a rotation matrix."""
    tr = (np.trace(R) - 1.0) / 2.0
    tr = float(np.clip(tr, -1.0, 1.0))
    return float(np.degrees(np.arccos(tr)))

def euler_zyx_deg(R: np.ndarray) -> np.ndarray:
    """Yaw-পিচ-রول (ZYX) in degrees (roll, pitch, yaw)."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll  = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0.0
    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))

# ============================================================
# Calibration loaders
# ============================================================
def load_camera_matrix(path: str) -> np.ndarray:
    K = np.load(path)
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3x3, got {K.shape}")
    return K

def load_dist_coeffs_or_zeros(path: str | None) -> np.ndarray:
    """
    If you have dist_coeffs.npy, load it.
    Otherwise return zeros.
    """
    if path is None:
        return np.zeros((5,), dtype=np.float64)

    dist = np.load(path)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)

    # Common sizes: 4, 5, 8, 12, 14
    if dist.size not in (4, 5, 8, 12, 14):
        raise ValueError(f"Unexpected distortion coeff length: {dist.size}")
    return dist

# ============================================================
# ArUco board pose from one image
# ============================================================
def make_grid_board(
    dict_name: str,
    markersX: int,
    markersY: int,
    markerLength_m: float,
    markerSeparation_m: float,
    ids: list[int] | None = None,
):
    """
    Creates an ArUco GridBoard that matches your printed board.
    markersX, markersY define grid size (e.g., 3x2 for 6 markers).
    ids (optional) should match the IDs used in your print (e.g., [0,1,2,3,4,5]).
    """
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))

    ids_arr = None
    if ids is not None:
        ids_arr = np.array(ids, dtype=np.int32).reshape(-1, 1)

    # OpenCV API differs by version
    if hasattr(aruco, "GridBoard_create"):
        # Older-style factory
        board = aruco.GridBoard_create(
            markersX, markersY,
            markerLength_m, markerSeparation_m,
            dictionary,
            ids_arr
        )
    else:
        # Newer class constructor
        board = aruco.GridBoard(
            (markersX, markersY),
            markerLength_m,
            markerSeparation_m,
            dictionary,
            ids_arr
        )

    return board, dictionary

def detect_markers(gray: np.ndarray, dictionary, params):
    aruco = cv2.aruco
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=params)
    return corners, ids, rejected

def estimate_board_pose_single_image(
    img_bgr: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    board,
    dictionary,
    min_markers: int = 1,
):
    """
    Returns (rvec, tvec, ids, corners) for ONE image, using the whole board pose.
    """
    aruco = cv2.aruco
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    params = aruco.DetectorParameters()

    corners, ids, _ = detect_markers(gray, dictionary, params)

    if ids is None or len(ids) == 0:
        raise ValueError("No ArUco markers detected in this image.")

    if len(ids) < min_markers:
        raise ValueError(f"Detected only {len(ids)} markers, need >= {min_markers} for stable board pose.")

    # estimatePoseBoard returns (retval, rvec, tvec)
    retval, rvec, tvec = aruco.estimatePoseBoard(
        corners, ids, board, K, dist, None, None
    )

    if retval <= 0 or rvec is None or tvec is None:
        raise ValueError("Board pose could not be estimated (not enough inliers/markers).")

    return rvec.reshape(3), tvec.reshape(3), ids.flatten(), corners

# ============================================================
# Two-image relative motion from board poses
# ============================================================
def aruco_board_ground_truth_motion_two_images(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    dict_name: str,
    markersX: int,
    markersY: int,
    markerLength_m: float,
    markerSeparation_m: float,
    board_ids: list[int] | None = None,
    min_markers: int = 2,
):
    """
    Computes ΔT, ΔR, Δt between img1 -> img2 using the BOARD pose in each image.
    """
    board, dictionary = make_grid_board(
        dict_name=dict_name,
        markersX=markersX,
        markersY=markersY,
        markerLength_m=markerLength_m,
        markerSeparation_m=markerSeparation_m,
        ids=board_ids
    )

    rvec1, tvec1, ids1, corners1 = estimate_board_pose_single_image(
        img1_bgr, K, dist, board, dictionary, min_markers=min_markers
    )
    rvec2, tvec2, ids2, corners2 = estimate_board_pose_single_image(
        img2_bgr, K, dist, board, dictionary, min_markers=min_markers
    )

    T1 = rvec_tvec_to_T(rvec1, tvec1)
    T2 = rvec_tvec_to_T(rvec2, tvec2)

    dT = invert_T(T1) @ T2
    dR = dT[:3, :3]
    dt = dT[:3, 3]

    return {
        "T1": T1, "T2": T2,
        "dT": dT,
        "dR": dR,
        "dt": dt,
        "euler_zyx_deg": euler_zyx_deg(dR),
        "rot_angle_deg": rotation_angle_deg(dR),
        "detected_ids_img1": ids1,
        "detected_ids_img2": ids2,
        "corners_img1": corners1,
        "corners_img2": corners2,
    }

# ============================================================
# Optional comparison vs your estimator (monocular -> scale ambiguous)
# ============================================================
def compare_to_orb_estimator(aruco_gt: dict, orb_result: dict):
    """
    Compares ArUco GT ΔR, Δt to your project result:
      - rotation error (deg)
      - translation direction error (deg)
    """
    R_gt = aruco_gt["dR"]
    t_gt = aruco_gt["dt"].reshape(3)

    R_est = orb_result["rotation_matrix"]
    t_est = orb_result["translation_vector"].reshape(3)

    # rotation error: R_err = R_gt^T R_est
    R_err = R_gt.T @ R_est
    rot_err_deg = rotation_angle_deg(R_err)

    # translation direction error
    t_gt_u = t_gt / (np.linalg.norm(t_gt) + 1e-9)
    t_est_u = t_est / (np.linalg.norm(t_est) + 1e-9)
    cosang = float(np.clip(np.dot(t_gt_u, t_est_u), -1.0, 1.0))
    t_dir_err_deg = float(np.degrees(np.arccos(cosang)))

    return {"rot_err_deg": rot_err_deg, "t_dir_err_deg": t_dir_err_deg}

# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    # --- paths ---
    img1_path = "images/IMG_0209.jpeg"
    img2_path = "images/IMG_0210.jpeg"

    # Your calibration files:
    K_path = "camera_matrix.npy"
    dist_path = None  # e.g. "dist_coeffs.npy" if you have it

    # --- load images ---
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise RuntimeError("Failed to load images. Check the paths.")

    # --- load calibration ---
    K = load_camera_matrix(K_path)
    dist = load_dist_coeffs_or_zeros(dist_path)

    print("Loaded K:\n", K)
    print("Loaded dist:", dist)

    # ========================================================
    # IMPORTANT: Set these to match YOUR PRINTED 6-marker board
    # ========================================================
    dict_name = "DICT_6X6_250"  # must match what you printed

    # If your board is 3 columns x 2 rows = 6 markers:
    markersX = 3
    markersY = 2

    # Measure your print (meters):
    markerLength_m = 0.05       # marker side length (e.g., 5cm)
    markerSeparation_m = 0.015   # gap between markers (e.g., 1cm)

    # IDs used in the board.
    # If you printed the OpenCV grid board with ids 0..5, keep this:
    board_ids = [0, 1, 2, 3, 4, 5]
    # If you’re not sure, set board_ids=None (it may still work, but best to match exactly)

    # Minimum number of detected markers required to accept a pose
    min_markers = 2

    gt = aruco_board_ground_truth_motion_two_images(
        img1, img2,
        K=K, dist=dist,
        dict_name=dict_name,
        markersX=markersX, markersY=markersY,
        markerLength_m=markerLength_m,
        markerSeparation_m=markerSeparation_m,
        board_ids=board_ids,
        min_markers=min_markers
    )

    print("\n=== ArUco BOARD GT relative motion (img1 -> img2) ===")
    print("Detected IDs img1:", gt["detected_ids_img1"])
    print("Detected IDs img2:", gt["detected_ids_img2"])
    print("Rotation angle (deg):", gt["rot_angle_deg"])
    print("Euler ZYX (deg) [roll,pitch,yaw]:", gt["euler_zyx_deg"])
    print("dt (meters):", gt["dt"])
    print("dT:\n", gt["dT"])

    # If you want to compare to your MotionEstimator:
    # from your_project_file import MotionEstimator
    # estimator = MotionEstimator(camera_matrix=K)
    # orb_result = estimator.estimate_motion(img1, img2)
    # cmp = compare_to_orb_estimator(gt, orb_result)
    # print("\n=== Compare ORB estimator vs ArUco GT ===")
    # print("Rotation error (deg):", cmp["rot_err_deg"])
    # print("Translation direction error (deg):", cmp["t_dir_err_deg"])
