import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict

class MotionEstimator:
    """
    Estimates 6DOF motion (translation and rotation) between two drone frames
    using feature matching and essential matrix decomposition.
    """
    
    def __init__(self, camera_matrix=None, focal_length=800, principal_point=None):
        """
        Initialize the motion estimator with camera intrinsic parameters.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix (if None, will be estimated)
            focal_length: focal length in pixels (used if camera_matrix is None)
            principal_point: (cx, cy) principal point (if None, uses image center)
        """
        self.camera_matrix = camera_matrix
        self.focal_length = focal_length
        self.principal_point = principal_point
        
        # Initialize feature detector (ORB is fast, SIFT is more accurate)
        self.detector = cv2.ORB_create(nfeatures=2000)
        # Alternative: self.detector = cv2.SIFT_create()
        
        # Initialize feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # For SIFT, use: cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def _get_camera_matrix(self, img_shape):
        """Create camera matrix if not provided."""
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
        """
        Detect features in both images and match them.
        
        Args:
            img1: First image (grayscale or BGR)
            img2: Second image (grayscale or BGR)
            ratio_thresh: Lowe's ratio test threshold
            
        Returns:
            pts1: Matched points in image 1 (Nx2)
            pts2: Matched points in image 2 (Nx2)
            matches: List of good matches
        """
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        # Detect keypoints and compute descriptors
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            raise ValueError("Not enough features detected in one or both images")
        
        # Match features using KNN (k=2 for ratio test)
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 8:
            raise ValueError(f"Not enough good matches found: {len(good_matches)}")
        
        # Extract matched point coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        return pts1, pts2, good_matches
    
    def estimate_motion(self, img1, img2) -> Dict:
        """
        Estimate 6DOF motion between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Dictionary containing:
                - 'rotation_matrix': 3x3 rotation matrix
                - 'translation_vector': 3x1 translation vector (direction, not scale)
                - 'euler_angles': (roll, pitch, yaw) in degrees
                - 'num_inliers': number of inlier matches
                - 'matched_points': (pts1, pts2) matched point coordinates
        """
        # Get camera matrix
        K = self._get_camera_matrix(img1.shape)
        
        # Detect and match features
        pts1, pts2, matches = self.detect_and_match_features(img1, img2)
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )
        
        # Count inliers
        num_inliers = np.sum(mask)
        
        # Filter points to only inliers
        pts1_inliers = pts1[mask.ravel() == 1]
        pts2_inliers = pts2[mask.ravel() == 1]
        
        # Recover pose from essential matrix
        _, R, t, pose_mask = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)
        
        # Convert rotation matrix to Euler angles (roll, pitch, yaw)
        euler_angles = self._rotation_matrix_to_euler(R)
        
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
        """
        Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees.
        Uses ZYX convention (yaw-pitch-roll).
        """
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
    
    def visualize_matches(self, img1, img2, pts1, pts2, num_display=50):
        """
        Visualize feature matches between two images.
        
        Args:
            img1: First image
            img2: Second image
            pts1: Points in first image
            pts2: Points in second image
            num_display: Number of matches to display
        """
        # Create side-by-side image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2)
        w = w1 + w2
        
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Copy images
        if len(img1.shape) == 2:
            vis[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        else:
            vis[:h1, :w1] = img1
            
        if len(img2.shape) == 2:
            vis[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        else:
            vis[:h2, w1:w1+w2] = img2
        
        # Draw matches
        num_matches = min(num_display, len(pts1))
        for i in range(num_matches):
            pt1 = tuple(pts1[i].astype(int))
            pt2 = tuple((pts2[i] + [w1, 0]).astype(int))
            
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(vis, pt1, 5, color, -1)
            cv2.circle(vis, pt2, 5, color, -1)
            cv2.line(vis, pt1, pt2, color, 1)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f'Feature Matches (showing {num_matches} of {len(pts1)})')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def print_motion_results(result: Dict):
    """Pretty print the motion estimation results."""
    print("=" * 60)
    print("DRONE MOTION ESTIMATION RESULTS")
    print("=" * 60)
    print(f"\nFeature Matching:")
    print(f"  Total matches: {result['total_matches']}")
    print(f"  Inliers: {result['num_inliers']}")
    print(f"  Inlier ratio: {result['num_inliers']/result['total_matches']:.2%}")
    
    print(f"\nRotation (Euler Angles):")
    roll, pitch, yaw = result['euler_angles']
    print(f"  Roll:  {roll:8.3f}°")
    print(f"  Pitch: {pitch:8.3f}°")
    print(f"  Yaw:   {yaw:8.3f}°")
    
    print(f"\nRotation Matrix:")
    print(result['rotation_matrix'])
    
    print(f"\nTranslation Vector (normalized direction):")
    print(result['translation_vector'].ravel())
    print("\nNote: Translation magnitude cannot be determined from monocular vision")
    print("      without scale information (depth or known object size)")
    print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Load two consecutive frames
    img1 = cv2.imread('images/frame1.jpeg')
    img2 = cv2.imread('images/frame2.jpeg')
    
    if img1 is None or img2 is None:
        print("Error: Could not load images. Please provide 'frame1.jpg' and 'frame2.jpg'")
        print("\nCreating synthetic example for demonstration...")
        
        # Create synthetic images for demo
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img2 = img1.copy()
        # Add some transformations for demo
        M = cv2.getRotationMatrix2D((320, 240), 5, 1.0)
        img2 = cv2.warpAffine(img2, M, (640, 480))
    
    # Initialize estimator
    estimator = MotionEstimator(focal_length=800)
    
    # Estimate motion
    try:
        result = estimator.estimate_motion(img1, img2)
        
        # Print results
        print_motion_results(result)
        
        # Visualize matches
        pts1, pts2 = result['inlier_points']
        estimator.visualize_matches(img1, img2, pts1, pts2, num_display=50)
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure the images have overlapping content with visible features.")