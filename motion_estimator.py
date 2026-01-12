import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional

# ==========================================
# PART 1: MOTION ESTIMATOR CLASS
# ==========================================

class MotionEstimator:
    """
    Estimates 6DOF motion (translation and rotation) between two drone frames
    using feature matching and essential matrix decomposition.
    """
    
    def __init__(self, camera_matrix=None, focal_length=800, principal_point=None):
        self.camera_matrix = camera_matrix
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.detector = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def _get_camera_matrix(self, img_shape):
        if self.camera_matrix is not None:
            return self.camera_matrix
        
        h, w = img_shape[:2]
        if self.principal_point is None:
            cx, cy = w / 2, h / 2
        else:
            cx, cy = self.principal_point
        
        K = np.array([
            [self.focal_length, 0, cx],
            [0, self.focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        return K
    
    def detect_and_match_features(self, img1, img2, ratio_thresh=0.75):
        if len(img1.shape) == 3: gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else: gray1 = img1
            
        if len(img2.shape) == 3: gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else: gray2 = img2
        
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            raise ValueError("Not enough features detected in one or both images")
        
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 8:
            raise ValueError(f"Not enough good matches found: {len(good_matches)}")
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        return pts1, pts2, good_matches
    
    def estimate_motion(self, img1, img2) -> Dict:
        K = self._get_camera_matrix(img1.shape)
        pts1, pts2, matches = self.detect_and_match_features(img1, img2)
        
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )
        
        num_inliers = int((mask.ravel() == 1).sum())
        pts1_inliers = pts1[mask.ravel() == 1]
        pts2_inliers = pts2[mask.ravel() == 1]
        
        _, R, t, pose_mask = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)
        
        euler_angles = self._rotation_matrix_to_euler_y_up(R.T)
        
        result = {
            'rotation_matrix': R,
            'translation_vector': t,
            'euler_angles': euler_angles,
            'num_inliers': num_inliers,
            'total_matches': len(matches),
            'matched_points': (pts1, pts2),
            'inlier_points': (pts1_inliers, pts2_inliers)
        }
        return result
    
    def _rotation_matrix_to_euler(self, R):
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        return np.degrees([roll, pitch, yaw])

    def _rotation_matrix_to_euler_y_up(self, R):
        S = np.diag([1, -1, 1])
        R2 = S @ R @ S
        return self._rotation_matrix_to_euler(R2)

    def visualize_matches(self, img1, img2, pts1, pts2, num_display=50):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2)
        w = w1 + w2
        
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        if len(img1.shape) == 2: vis[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        else: vis[:h1, :w1] = img1
            
        if len(img2.shape) == 2: vis[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        else: vis[:h2, w1:w1+w2] = img2
        
        num_matches = min(num_display, len(pts1))
        for i in range(num_matches):
            pt1 = tuple(pts1[i].astype(int))
            pt2 = tuple((pts2[i] + [w1, 0]).astype(int))
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(vis, pt1, 5, color, -1)
            cv2.circle(vis, pt2, 5, color, -1)
            cv2.line(vis, pt1, pt2, color, 1)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f'Feature Matches (showing {num_matches} of {len(pts1)})')
        plt.axis('off')
        plt.show()

# ==========================================
# PART 2: ARUCO & MATH HELPER FUNCTIONS
# ==========================================

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

def estimate_pose_square_marker_solvepnp(corner_2d, marker_length_m, K, dist):
    half = marker_length_m / 2.0
    objp = np.array([
        [-half,  half, 0.0], [ half,  half, 0.0],
        [ half, -half, 0.0], [-half, -half, 0.0],
    ], dtype=np.float32)
    imgp = np.asarray(corner_2d, dtype=np.float32).reshape(-1, 2)
    
    try:
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except Exception:
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        
    if not ok: return None, None
    return rvec, tvec

def get_camera_pose_from_aruco(frame, aruco_detector, K, dist, marker_length_m, wanted_id):
    """
    Returns the 4x4 transformation matrix T_world_cam.
    Initializes world frame to match ArUco marker frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    
    if ids is None: return None, None
    ids = ids.flatten()
    if wanted_id not in ids: return None, None

    idx = np.where(ids == wanted_id)[0][0]
    corner_2d = corners[idx].reshape(4, 2)

    rvec, tvec = estimate_pose_square_marker_solvepnp(corner_2d, marker_length_m, K, dist)
    if rvec is None: return None, None

    # Marker to Camera transform
    R_cm, _ = cv2.Rodrigues(rvec)
    T_cm = np.eye(4)
    T_cm[:3, :3] = R_cm
    T_cm[:3, 3] = tvec.flatten()

    # Camera in World (inverted)
    T_wc = T_inv(T_cm)
    
    # Also return raw position for scale calculation
    cam_center = T_wc[:3, 3]
    
    return T_wc, cam_center

# ==========================================
# PART 3: PLOTTING & DISPLAY
# ==========================================

def plot_trajectories(vo, aruco):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    if len(vo) > 0:
        ax.plot(vo[:, 0], vo[:, 1], vo[:, 2], label="VO (Estimated)", color='tab:blue')
        ax.scatter(vo[0, 0], vo[0, 1], vo[0, 2], s=60, color='tab:blue', marker='^')

    if len(aruco) > 0:
        ax.plot(aruco[:, 0], aruco[:, 1], aruco[:, 2], label="ArUco (Ground Truth)", color='tab:orange')
        ax.scatter(aruco[0, 0], aruco[0, 1], aruco[0, 2], s=60, color='tab:orange', marker='o')

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters)")
    ax.set_title("Drone Trajectory Comparison")
    ax.legend()
    ax.grid(True)
    plt.show()

def print_motion_results(result: Dict, frame_idx=None):
    if frame_idx is not None:
        print(f"\n--- Frame {frame_idx} Motion Results ---")
    else:
        print("\n" + "=" * 60)
        print(" MOTION ESTIMATION RESULTS ")
        print("=" * 60)
        
    print(f" Matches: {result['total_matches']} (Inliers: {result['num_inliers']})")
    
    # 1. Output Rotation Matrix
    print(" Rotation Matrix (3x3):")
    print(result['rotation_matrix'])
    
    # 2. Output Translation Vector
    t = result['translation_vector'].ravel()
    print(f" Translation Vector: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    
    if frame_idx is None:
        print("=" * 60 + "\n")

# ==========================================
# PART 4: USER INTERFACE & LOGIC
# ==========================================

def get_user_camera_matrix():
    print("\n--- Camera Calibration Input ---")
    print("Enter camera parameters (press Enter to skip/auto-estimate)")
    
    f_str = input(" Focal Length (pixels): ").strip()
    cx_str = input(" Principal Point X (cx): ").strip()
    cy_str = input(" Principal Point Y (cy): ").strip()
    
    if not f_str:
        print(" > No input. Will estimate K automatically from image size.")
        return None
    
    f = float(f_str)
    cx = float(cx_str) if cx_str else 0.0
    cy = float(cy_str) if cy_str else 0.0
    
    return {'f': f, 'cx': cx, 'cy': cy}

def run_images_mode(k_params):
    print("\n--- Image Mode ---")
    path1 = input("Enter path for Image 1: ").strip().strip('"')
    path2 = input("Enter path for Image 2: ").strip().strip('"')
    
    if not os.path.exists(path1) or not os.path.exists(path2):
        print("Error: One or both files not found!")
        return

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    
    # Build K
    h, w = img1.shape[:2]
    if k_params:
        K = np.array([
            [k_params['f'], 0, k_params['cx'] if k_params['cx'] else w/2],
            [0, k_params['f'], k_params['cy'] if k_params['cy'] else h/2],
            [0, 0, 1]
        ])
    else:
        f = 0.9 * max(w, h)
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])

    estimator = MotionEstimator(camera_matrix=K)
    
    try:
        result = estimator.estimate_motion(img1, img2)
        print_motion_results(result)
        
        pts1, pts2 = result['inlier_points']
        estimator.visualize_matches(img1, img2, pts1, pts2)
        
    except ValueError as e:
        print(f"Error: {e}")

def run_video_mode(k_params):
    print("\n--- Video Mode ---")
    video_path = input("Enter video path: ").strip().strip('"')
    
    if not os.path.exists(video_path):
        print("Error: Video file not found!")
        return

    # ArUco Config
    ARUCO_ID = 0
    ARUCO_CELL_CM = 2.5
    ARUCO_TOTAL_CELLS = 6
    marker_length_m = (ARUCO_TOTAL_CELLS * ARUCO_CELL_CM) / 100.0
    
    cap = cv2.VideoCapture(video_path)
    ok, first = cap.read()
    if not ok: return
    
    # Build K
    h, w = first.shape[:2]
    if k_params:
        K = np.array([
            [k_params['f'], 0, k_params['cx'] if k_params['cx'] else w/2],
            [0, k_params['f'], k_params['cy'] if k_params['cy'] else h/2],
            [0, 0, 1]
        ], dtype=np.float64)
    else:
        f = 0.9 * max(w, h)
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)
        
    dist = np.zeros((5, 1))
    
    estimator = MotionEstimator(camera_matrix=K)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Initialize Loop
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    T_w_c = np.eye(4)
    vo_positions = []
    aruco_positions = []
    
    prev_frame = None
    prev_aruco = None
    last_scale = 1.0
    frame_counter = 0
    start_frame_idx = 0
    
    print("Searching for initial ArUco marker to align coordinates...")
    
    # 1. Alignment Loop
    while True:
        ok, frame = cap.read()
        if not ok: 
            print("Video ended without finding marker.")
            return
            
        T_start, aruco_pos = get_camera_pose_from_aruco(frame, aruco_detector, K, dist, marker_length_m, ARUCO_ID)
        
        if T_start is not None:
            print(f"Alignment Successful at frame {start_frame_idx}!")
            T_w_c = T_start
            vo_positions.append(T_w_c[:3, 3].copy())
            aruco_positions.append(T_w_c[:3, 3].copy())
            
            prev_frame = frame
            prev_aruco = aruco_pos
            frame_counter = start_frame_idx
            break
        start_frame_idx += 1
        
    # 2. Main Loop
    print("Processing video frames (Press Ctrl+C to stop early)...")
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            
            frame_counter += 1
            if frame_counter % 2 != 0: continue # Skip every other frame for speed

            # Track ArUco
            _, aruco_pos = get_camera_pose_from_aruco(frame, aruco_detector, K, dist, marker_length_m, ARUCO_ID)
            
            if aruco_pos is not None:
                aruco_positions.append(aruco_pos)

            # VO Estimation
            try:
                result = estimator.estimate_motion(prev_frame, frame)
                
                # OUTPUT ROTATION & TRANSLATION FOR THIS FRAME
                print_motion_results(result, frame_idx=frame_counter)

            except Exception:
                prev_frame = frame
                prev_aruco = aruco_pos if aruco_pos is not None else prev_aruco
                continue
                
            R = result["rotation_matrix"]
            t = result["translation_vector"].reshape(3, 1)
            t_dir = t / (np.linalg.norm(t) + 1e-9)
            
            # Scale Calculation
            if prev_aruco is not None and aruco_pos is not None:
                dist_moved = np.linalg.norm(aruco_pos - prev_aruco)
                if dist_moved > 0.02: # 2cm threshold
                    last_scale = dist_moved
            
            # Apply Scale & Direction Fix (-1.0 flip)
            t_scaled = -1.0 * last_scale * t_dir
            
            T_rel = se3_to_T(R, t_scaled)
            T_w_c = T_w_c @ T_rel
            
            vo_positions.append(T_w_c[:3, 3].copy())
            
            prev_frame = frame
            if aruco_pos is not None: prev_aruco = aruco_pos
            
    except KeyboardInterrupt:
        print("Stopping early...")
        
    print(f"Done. Processed {len(vo_positions)} frames.")
    
    vo_arr = np.array(vo_positions)
    aruco_arr = np.array(aruco_positions)
    
    plot_trajectories(vo_arr, aruco_arr)

# ==========================================
# PART 5: MAIN ENTRY POINT
# ==========================================

if __name__ == "__main__":
    print("="*40)
    print("  DRONE MOTION ESTIMATION TOOL")
    print("="*40)
    
    while True:
        mode = input("Select Mode:\n 1. Video Analysis (VO vs ArUco)\n 2. Two Image Comparison\n > ").strip()
        if mode in ['1', '2']: break
        print("Invalid selection.")
        
    k_params = get_user_camera_matrix()
    
    if mode == '1':
        run_video_mode(k_params)
    else:
        run_images_mode(k_params)