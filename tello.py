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
import keyPressModule as kp

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
    # Inside the TelloController class:
    def send_rc_control(self, lr, fb, ud, yv):
        """
        Sends the 'rc a b c d' command for continuous movement.
        lr: Left/Right (-100 to 100)
        fb: Forward/Backward (-100 to 100)
        ud: Up/Down (-100 to 100)
        yv: Yaw velocity (-100 to 100)
        """
        if not self.in_flight:
            # We should not send movement commands if not in flight
            return
            
        command = f"rc {lr} {fb} {ud} {yv}"
        # For RC commands, we don't wait for 'ok', just send and forget (non-blocking)
        try:
            self.sock.sendto(command.encode(), self.tello_address)
        except Exception as e:
            # Silently fail for continuous RC commands to avoid excessive console output
            pass

# ----------------------------------------------------------------------
#                             MAIN FUNCTION
# ----------------------------------------------------------------------

def get_keyboard_input(tello_controller):
    """Checks for key presses and sends commands to the TelloController."""
    lr, fb, ud, yv = 0, 0, 0, 0
    # Use a safe speed for manual control (10-100)
    speed = 50 
    
    # Check for Takeoff/Land
    if kp.getkey("h"): # 'h' for Hover/Takeoff
        if not tello_controller.in_flight:
            print("🚀 Taking off...")
            tello_controller.safe_takeoff()
    elif kp.getkey("l"): # 'l' for Land
        if tello_controller.in_flight:
            print("🛬 Landing...")
            tello_controller.safe_land()
            
    if not tello_controller.in_flight:
        return [0, 0, 0, 0] # Drone must be in flight for RC control
    
    # Movement: Left/Right (lr)
    if kp.getkey("LEFT"): 
        lr = -speed
    elif kp.getkey("RIGHT"): 
        lr = speed

    # Movement: Forward/Backward (fb)
    if kp.getkey("UP"): 
        fb = speed
    elif kp.getkey("DOWN"): 
        fb = -speed

    # Movement: Up/Down (ud)
    if kp.getkey("w"): 
        ud = speed
    elif kp.getkey("s"): 
        ud = -speed

    # Movement: Yaw Velocity (yv)
    if kp.getkey("a"): 
        yv = -speed
    elif kp.getkey("d"): 
        yv = speed

    # Flips (requires good battery and altitude)
    if kp.getkey("f"):
        # Example flip: forward
        print("🤸 Performing forward flip...")
        tello_controller.send_command('flip f')
        time.sleep(COMMAND_COOLDOWN) # Wait for flip to complete

    # Return the four RC control values
    return [lr, fb, ud, yv]
    
# ----------------------------------------------------------------------
#                                MAIN FUNCTION
# ----------------------------------------------------------------------

def main():
    """Initializes the Tello controller, starts the video stream, and displays it with keyboard control."""
    
    # 1. Initialize the Tello controller
    tello = TelloController()
    
    # 1b. Initialize Pygame for key presses
    kp.init() 
    
    # 2. Enter SDK Mode and send 'streamon' command
    if not tello.start_sdk_mode():
        print("Fatal error: Could not establish communication with Tello. Aborting.")
        tello.stop()
        return

    # 3. Initialize OpenCV Video Capture
    cap = cv2.VideoCapture(VIDEO_STREAM_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) 

    if not cap.isOpened():
        print("-" * 50)
        print(f"❌ Error: Failed to open video stream on {VIDEO_STREAM_URL}.")
        print("Check Tello connection and firewall.")
        print("-" * 50)
        tello.stop()
        return

    # Image Capture setup
    image_count = 0
    capture_folder = "tello_captures"
    os.makedirs(capture_folder, exist_ok=True)
    last_rc_command_time = time.time()

    try:
        print("▶️ Live video stream active. Press 'q' to quit, 'h' to takeoff, 'l' to land, 'p' to capture image.")
        
        while True:
            # Read the frame
            ret, frame = cap.read()

            if not ret:
                time.sleep(0.01)
                continue

            # --- KEYBOARD CONTROL AND COMMANDS ---
            
            # Get control values and handle takeoff/land/flip
            rc_values = get_keyboard_input(tello) 
            
            # Send RC Control command (non-blocking)
            lr, fb, ud, yv = rc_values
            tello.send_rc_control(lr, fb, ud, yv)
            last_rc_command_time = time.time()
            
            # Image Capture (Key 'p')
            if kp.getkey("p"):
                filename = os.path.join(capture_folder, f"tello_capture_{image_count:04d}.png")
                cv2.imwrite(filename, frame)
                print(f"📸 Image captured: {filename}")
                image_count += 1
                time.sleep(0.3) # Cooldown to avoid multiple captures on a single press

            # --- VIDEO DISPLAY ---
            
            # Display current Tello state on the frame
            status_text = f"Flight: {'In Air' if tello.in_flight else 'Landed'} | CMD: rc {lr} {fb} {ud} {yv}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            display_frame = cv2.resize(frame, (480, 360))
            cv2.imshow("Tello Live Video Feed (Press 'q' to quit)", display_frame)

            # Handle user input (OpenCV window 'q' key)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            time.sleep(1/60) # Frame rate governor

    except KeyboardInterrupt:
        print("Program interrupted by user (Ctrl+C).")
    finally:
        # 7. Cleanup
        print("Stopping video stream and controller...")
        tello.send_command('streamoff')
        # Ensure the drone lands if it's still flying
        if tello.in_flight:
            tello.safe_land() 
        cap.release()
        cv2.destroyAllWindows()
        tello.stop()
        print("Program finished.")

if __name__ == '__main__':

    main()