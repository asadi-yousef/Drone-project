import argparse
import cv2
#import mediapipe as mp # This is imported but not used in the core logic below
import time
import socket
import threading
import numpy as np
from collections import deque, defaultdict
from math import hypot
import os

# ---------- CONFIG ----------
LOCAL_UDP_PORT = 9000
TELLO_IP = "192.168.10.1"
TELLO_CMD_PORT = 8889

# --- NEW VIDEO CONFIG ---
TELLO_VIDEO_PORT = 11111
# OpenCV URL format for reading the UDP H.264 stream from the Tello
# 0.0.0.0 tells the computer to listen on all interfaces for data coming to this port.
VIDEO_STREAM_URL = f'udp://@0.0.0.0:{TELLO_VIDEO_PORT}' 
# ------------------------

# 360° scan settings
SECTOR_SIZE_DEG = 30 # 12 sectors of 30°
INCREMENTAL_ANGLE = 30 # degrees per rotation step
INCREMENTAL_PAUSE = 1.5 # sec between steps
ROTATION_RETRY_ATTEMPTS = 2
MAPPING_MIN_SECTORS = 10  # require >=10/12 sectors covered
MAPPING_MIN_SAMPLES_PER_SECTOR = 2  # samples per sector

# Command behavior
COMMAND_COOLDOWN = 2.5
MIN_BATTERY_LEVEL = 20
MOVEMENT_DISTANCE_CM = 20 # safer indoors (forward/back)
ALTITUDE_STEP_CM = 20 # up/down step
TELLO_SPEED_CM_S = 10 

class TelloController:
    def __init__(self, local_port=LOCAL_UDP_PORT, tello_ip=TELLO_IP, tello_port=TELLO_CMD_PORT):
        self.tello_ip = tello_ip
        self.tello_port = tello_port
        self.tello_address = (self.tello_ip, self.tello_port)
        self.local_ip = ''
        self.local_port = local_port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.local_ip, self.local_port))
        self.response = None
        self.stop_event = threading.Event()
        self.response_thread = threading.Thread(target=self._receive_response, daemon=True)
        self.response_thread.start()

        self.in_flight = False
        self.last_error = None
        self.motor_fault = False
        self.imu_fault = False
        self.last_motor_fault_time = 0.0
        print("✅ TelloController initialized.")

    def _receive_response(self):
        while not self.stop_event.is_set():
            try:
                self.sock.settimeout(1.0)
                data, _ = self.sock.recvfrom(1024)
                text = data.decode(errors='ignore').strip()
                if not text:
                    continue
                self.response = text
                print(f"<- Received: {text}")
                lower = text.lower()
                if 'motor stop' in lower:
                    self.motor_fault = True
                    self.in_flight = False
                    self.last_motor_fault_time = time.time()
                    self.last_error = text
                    print("⚠ Motor fault detected: marking drone as not flying.")
                if 'no valid imu' in lower:
                    self.imu_fault = True
                    self.in_flight = False
                    self.last_error = text
                    print("⚠ IMU fault detected.")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"❌ Receive thread error: {e}")
                break

    def send_command(self, command, timeout=7.0, retries=1):
        """
        Sends command; returns (ok: bool, resp: str|None).
        Query commands (ending with '?') return the raw string response.
        """
        if not command:
            return True, None

        cmd_root = command.split()[0]
        is_query = command.strip().endswith('?')

        if self.motor_fault and cmd_root not in ('land', 'battery?'):
            since = time.time() - self.last_motor_fault_time
            if since < 6.0:
                print(f"⚠ Suppressed movement command '{command}' due to recent motor fault ({since:.1f}s)")
                return False, "Motor Fault"
            else:
                print("Clearing stale motor fault.")
                self.motor_fault = False

        for i in range(retries + 1):
            print(f"-> Sending: {command}")
            try:
                self.sock.sendto(command.encode(), self.tello_address)
            except Exception as e:
                print(f"❌ UDP send failed: {e}")
                return False, f"UDP send failed: {e}"

            start = time.time()
            while time.time() - start < timeout:
                if self.response is not None:
                    text = self.response
                    self.response = None

                    if is_query and text:
                        return True, text

                    if text == 'ok':
                        if cmd_root == 'takeoff':
                            self.in_flight = True
                            print("✅ Tello state: in_flight = True")
                            time.sleep(4)
                        if cmd_root == 'land':
                            self.in_flight = False
                            print("✅ Tello state: in_flight = False")
                            time.sleep(3)
                        return True, "ok"
                    elif 'error' in text.lower():
                        self.last_error = text
                        if 'not joystick' in text.lower() and i < retries:
                            print("⚠ 'Not joystick' error. Re-entering SDK mode and retrying command.")
                            self.send_command('command')
                            time.sleep(1.0)
                            break
                        return False, text
                time.sleep(0.05)

            if i < retries:
                print(f"⚠ Timeout or error on attempt {i+1}. Retrying...")
                time.sleep(0.5)

        print(f"⚠ Command '{command}' failed after all retries.")
        return False, self.last_error or "Timeout"

    def safe_takeoff(self):
        if not self.in_flight:
            ok, _ = self.send_command("takeoff")
            return ok
        return False

    def safe_land(self):
        if self.in_flight:
            ok, _ = self.send_command("land")
            return ok
        return False

    def start_sdk_mode(self):
        ok, _ = self.send_command('command')
        if not ok:
            print("❌ Failed to enter SDK mode")
            return False

        ok, battery_level = self.send_command('battery?', timeout=3.5)
        if ok and battery_level and battery_level.strip().isdigit():
            battery = int(battery_level.strip())
            print(f"🔋 Battery level: {battery}%")
            if battery < MIN_BATTERY_LEVEL:
                print(f"🚨 LOW BATTERY WARNING: {battery}% (min {MIN_BATTERY_LEVEL}%).")
        else:
            print("⚠ Could not retrieve battery level.")

        # slow speed for mapping
        self.send_command(f"speed {TELLO_SPEED_CM_S}")

        # keep the drone stream on 
        self.send_command('streamoff')
        time.sleep(0.3)
        ok, _ = self.send_command('streamon')
        if not ok:
            print("❌ Failed to enable video stream")
            return False
        print("✅ SDK mode active and video stream is on.")
        time.sleep(1.5)
        return True

    def stop(self):
        print("Stopping Tello controller...")
        self.stop_event.set()
        if self.response_thread.is_alive():
            self.response_thread.join(timeout=1.0)
        self.sock.close()
        print("Tello controller stopped.")

    def safe_rotate_cw(self, angle):
        ok, _ = self.send_command(f"cw {angle}", retries=ROTATION_RETRY_ATTEMPTS)
        if ok:
            time.sleep(1.8)
        return ok

# ----------------------------------------------------------------------
#                             MAIN FUNCTION
# ----------------------------------------------------------------------

def main():
    """Initializes the Tello controller, starts the video stream, and displays it."""
    
    # 1. Initialize the Tello controller (handles command/response on UDP 9000)
    tello = TelloController()
    
    # 2. Enter SDK Mode and send 'streamon' command
    # This prepares the drone to send video data to the default port 11111
    if not tello.start_sdk_mode():
        print("Fatal error: Could not establish communication with Tello. Aborting.")
        tello.stop()
        return

    # 3. Initialize OpenCV Video Capture
    # We use cv2.CAP_FFMPEG as the backend for decoding H.264 streams
    cap = cv2.VideoCapture(VIDEO_STREAM_URL, cv2.CAP_FFMPEG)
    
    # Set buffer size to a low value to reduce stream latency (lag)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) 

    if not cap.isOpened():
        print("-" * 50)
        print(f"❌ Error: Failed to open video stream on {VIDEO_STREAM_URL}.")
        print("Check if port 11111 is blocked by your Windows Firewall or Antivirus.")
        print("Also ensure the drone is on and you are connected to its Wi-Fi.")
        print("-" * 50)
        tello.stop()
        return

    try:
        print("▶️ Live video stream active. Press 'q' in the video window to quit.")
        
        while True:
            # 4. Read the frame from the stream
            ret, frame = cap.read()

            if not ret:
                # If frame read fails, print a warning and continue trying
                # NOTE: For Tello, sometimes the first few frames fail to decode.
                print("⚠️ Frame read failed. Retrying...")
                time.sleep(0.01)
                continue

            # 5. Display the frame
            # Resizing to a manageable size (e.g., 480x360)
            display_frame = cv2.resize(frame, (480, 360))
            cv2.imshow("Tello Live Video Feed (Press 'q' to quit)", display_frame)

            # 6. Handle user input
            key = cv2.waitKey(1) & 0xFF
            
            # Quitting command
            if key == ord('q'):
                break

            # Small delay to yield CPU time
            time.sleep(1/60)

    except KeyboardInterrupt:
        print("Program interrupted by user (Ctrl+C).")
    finally:
        # 7. Cleanup
        print("Stopping video stream and controller...")
        tello.send_command('streamoff')
        cap.release()
        cv2.destroyAllWindows()
        tello.stop()
        print("Program finished.")

if __name__ == '__main__':
    # Ensure all necessary folders/files for your ORB-SLAM log exist if you plan to use it later
    # if not os.path.exists(os.path.dirname(ORB_SLAM_LOG_FILE)):
    #     os.makedirs(os.path.dirname(ORB_SLAM_LOG_FILE))
        
    main()