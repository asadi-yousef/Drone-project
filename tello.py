# tello_vo_driver.py

import cv2
import numpy as np
import time
from djitellopy import Tello
# Import your class and function from the motion_estimator.py file
from motion_estimator import MotionEstimator, print_motion_results

# --- 1. Tello Camera Calibration ---
# The Tello resolution is typically 960x720. 
# Using a good estimate for the intrinsic matrix (K) is crucial.
# Replace these values if you perform your own calibration!
K_TELLO_960x720 = np.array([
    [921.0, 0.0, 480.0],  # fx (focal length), 0, cx (principal point x)
    [0.0, 921.0, 360.0],  # 0, fy, cy (principal point y)
    [0.0, 0.0, 1.0]
], dtype=np.float64)


def run_visual_odometry_test():
    """Connects to Tello, captures frames, and estimates motion."""
    print("Initializing Motion Estimator with Tello Camera Matrix...")
    estimator = MotionEstimator(camera_matrix=K_TELLO_960x720)
    tello = Tello()

    try:
        # 1. Connect to Tello
        tello.connect()
        print(f"Tello connected. Battery: {tello.get_battery()}%")
        tello.streamon()
        time.sleep(1) # Wait for the stream to start

        # 2. Capture Reference Frame (Frame 1)
        img1 = tello.get_frame_read().frame
        if img1 is None:
            raise Exception("Could not get initial frame from Tello stream.")

        # Ensure image resolution matches the K matrix for accuracy
        # Resize if necessary (e.g., if K is 960x720 but stream is 640x480)
        if img1.shape[1] != 960 or img1.shape[0] != 720:
             # NOTE: For real testing, you should calibrate to the actual stream resolution.
             print(f"WARNING: Image size {img1.shape[:2]} does not match K matrix (720, 960).")


        # 3. Execute Movement
        tello.takeoff()
        print("Tello took off. Moving forward 50 cm...")
        tello.move_forward(50) 
        time.sleep(3) # Wait for movement to stabilize

        # 4. Capture Current Frame (Frame 2)
        img2 = tello.get_frame_read().frame
        if img2 is None:
            raise Exception("Could not get second frame after movement.")

        # 5. Estimate Motion
        print("Frames captured. Estimating 6DOF motion...")
        result = estimator.estimate_motion(img1, img2)

        # 6. Display Results
        print_motion_results(result)
        
        # Visualize matches (Optional, but requires a screen/environment to display Matplotlib)
        pts1, pts2 = result['inlier_points']
        estimator.visualize_matches(img1, img2, pts1, pts2, num_display=50)

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An error occurred during Tello operation or motion estimation: {e}")
    
    finally:
        # 7. Safety Landing
        print("\nTest complete. Landing Tello...")
        tello.land()
        tello.streamoff()
        tello.end()

if __name__ == "__main__":
    run_visual_odometry_test()