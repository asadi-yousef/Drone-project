import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- הגדרות מצלמה (iPhone 13 Pro Max) ---
# אם צילמת ב-4K, הפוקוס הוא ~3050. אם ב-1080p, חלק ב-2.

focal_length = 1525.0 # חצי מ-3050 כי זה HD ולא 4K
cx = 540  # מרכז רוחב (1080/2)
cy = 960  # מרכז גובה (1920/2)

K = np.array([[focal_length, 0, cx],
              [0, focal_length, cy],
              [0,    0,    1]], dtype=np.float32)

def find_matches(img1, img2):
    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return kp1, kp2, good_matches

# --- פונקציית ArUco מעודכנת ---
def get_aruco_pose(frame, marker_size=0.025): # 2.5 cm = 0.025 m
    # שימוש ב-4X4 כפי שמופיע בסרטון שלך
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    corners, ids, _ = detector.detectMarkers(frame)
    
    if ids is not None:
        obj_points = np.array([[-marker_size/2,  marker_size/2, 0],
                              [ marker_size/2,  marker_size/2, 0],
                              [ marker_size/2, -marker_size/2, 0],
                              [-marker_size/2, -marker_size/2, 0]], dtype=np.float32)
        
        # חישוב יציב יותר בעזרת solvePnP
        _, rvec, tvec = cv2.solvePnP(obj_points, corners[0], K, None)
        return tvec
    return None

class DualTracker:
    def __init__(self):
        self.R_global = np.eye(3)
        self.t_global = np.zeros((3, 1))
        self.path_sift = []
        self.path_aruco = []
        self.scale_factor = 0.01  # תיקון קנה מידה ראשוני

    def update_sift(self, R_rel, t_rel):
        # ננרמל את t_rel אם הוא קטן מדי
        norm_t = np.linalg.norm(t_rel)
        
        # אם התנועה היא "רעש" (0 מוחלט), נתעלם. אם היא סבירה, נכניס לגרף.
        if norm_t < 0.0001: 
            return

        # עדכון המיקום הגלובלי (הגדלנו מעט את ה-scale_factor ליציבות)
        self.t_global = self.t_global + self.R_global @ (t_rel * 0.05)
        self.R_global = R_rel @ self.R_global
        
        # שמירת הנתיב
        self.path_sift.append([self.t_global[0,0], self.t_global[2,0], -self.t_global[1,0]])

    def update_aruco(self, tvec):
        if tvec is not None:
            # ArUco נותן לנו את ה-Scale האמיתי. 
            # נשתמש בו כדי לעדכן את ה-scale_factor של ה-SIFT בזמן אמת
            self.path_aruco.append([tvec[0,0], tvec[2,0], -tvec[1,0]])

def main(video_path, aruco = False):
    cap = cv2.VideoCapture(video_path)
    tracker = DualTracker()
    
    frames_buffer = [] # בופר לשמירת פריימים קודמים

    while True:
        ret, frame = cap.read()
        if not ret: break

        # הוספת פריים לבופר
        frames_buffer.append(frame)
        if len(frames_buffer) > 5: # נשווה תמיד לפריים שהיה לפני 5 פריימים
            prev_frame = frames_buffer.pop(0)
            
            # 1. חישוב SIFT
            kp1, kp2, matches = find_matches(prev_frame, frame)
            if len(matches) > 15: # הורדנו מעט את רף המאצ'ים
                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
                E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None and E.shape == (3,3):
                    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
                    tracker.update_sift(R, t)

        # 2. חישוב ArUco (תמיד, בכל פריים)
        if aruco:
            t_aruco = get_aruco_pose(frame)
            if t_aruco is not None:
                tracker.update_aruco(t_aruco)

        cv2.imshow('Tracking...', cv2.resize(frame, (480, 850)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    
    # שינוי כאן: מציירים אם יש לפחות סוג אחד של נתונים
    if len(tracker.path_sift) > 1 or len(tracker.path_aruco) > 1:
        plot_comparison(tracker.path_sift, tracker.path_aruco)
    else:
        print("Still no data. Try moving the camera more clearly.")

def plot_comparison(sift_path, aruco_path):
    sift_path = np.array(sift_path)
    aruco_path = np.array(aruco_path)

    # סינון רעשים בסיסי - ממוצע נע (Moving Average)
    def smooth(data, window=5):
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='same')

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    if len(sift_path) > 1:
        # החלקת הצירים בנפרד
        xs = smooth(sift_path[:, 0])
        ys = smooth(sift_path[:, 1])
        zs = smooth(sift_path[:, 2])
        ax.plot(xs, zs, -ys, label='SIFT (Smoothed)', color='blue', linewidth=2)

    if len(aruco_path) > 1:
        # ArUco הוא ה-Ground Truth שלנו
        ax.plot(aruco_path[:, 0], aruco_path[:, 2], -aruco_path[:, 1], 
                label='ArUco (Ground Truth)', color='red', linestyle='--', linewidth=2)
        # הוספת נקודות לציון מקום המרקר
        ax.scatter(aruco_path[0,0], aruco_path[0,2], -aruco_path[0,1], color='red', s=50)

    ax.set_xlabel('X (Right/Left)')
    ax.set_ylabel('Z (Forward/Back)')
    ax.set_zlabel('Y (Up/Down)')
    ax.set_title("Drone Motion Comparison: 2.5cm Marker Analysis")
    ax.legend()
    
    # הבטחת קנה מידה שווה כדי שהגרף לא ייראה מעוות
    all_data = np.vstack([sift_path, aruco_path]) if len(aruco_path) > 0 else sift_path
    max_range = np.ptp(all_data, axis=0).max() / 2.0
    mid = all_data.mean(axis=0)
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[2]-max_range, mid[2]+max_range)
    ax.set_zlim(-mid[1]-max_range, -mid[1]+max_range)
    
    plt.show()

if __name__ == "__main__":
    main('test.mp4')