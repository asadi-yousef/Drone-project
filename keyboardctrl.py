import cv2
from djitellopy import tello
import keyPressModule as kp
from time import sleep
import numpy as np
import math
import socket
import threading

######### PARAMETERS ###########
fSpeed = 117/10  # forward speed in cm/s
aSpeed = 360/10  # angular speed deg/s
interval = 0.25

dInterval = fSpeed * interval
aInterval = aSpeed * interval
###########################################

x, y = 500, 500
a = 0
yaw = 0

kp.init()
me = tello.Tello()
me.connect()

# -------- VIDEO STREAM SETUP (UDP / OpenCV) --------
me.send_command_without_return("command")
sleep(0.5)
me.send_command_without_return("streamoff")
sleep(0.5)
me.send_command_without_return("streamon")
sleep(1.2)

VIDEO_STREAM_URL = "udp://@0.0.0.0:11111"
cap = cv2.VideoCapture(VIDEO_STREAM_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

if not cap.isOpened():
    print("❌ Video stream failed to open")
    exit()
else:
    print("✅ Video stream active")
# --------------------------------------------------

# -------- VIDEO RECORDING SETUP --------
fourcc = cv2.VideoWriter_fourcc(*'XVID')
tello_video = cv2.VideoWriter(
    'tello_camera.avi',
    fourcc,
    20,
    (360, 240)
)
# --------------------------------------

print("Battery:", me.get_battery())
points = []

# -------- VIDEO THREAD --------
stop_video = False

def video_loop():
    while not stop_video:
        ret, frame = cap.read()
        if ret and frame is not None:
            frame = cv2.resize(frame, (360, 240))
            cv2.imshow("Tello Camera", frame)
            tello_video.write(frame)
        cv2.waitKey(1)
# ------------------------------

video_thread = threading.Thread(target=video_loop, daemon=True)
video_thread.start()

# -------- CONTROL FUNCTIONS --------
def getkeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 15
    aspeed = 50
    global x, y, yaw, a
    d = 0

    if kp.getkey("LEFT"):
        lr = -speed
        d = dInterval
        a = -180

    elif kp.getkey("RIGHT"):
        lr = speed
        d = -dInterval
        a = 180

    elif kp.getkey("UP"):
        fb = speed
        d = dInterval
        a = 270

    elif kp.getkey("DOWN"):
        fb = -speed
        d = -dInterval
        a = -90

    elif kp.getkey("w"):
        ud = speed
    elif kp.getkey("s"):
        ud = -speed

    elif kp.getkey("a"):
        yv = -aspeed
        yaw -= aInterval

    elif kp.getkey("d"):
        yv = aspeed
        yaw += aInterval

    sleep(0.25)

    a += yaw
    x += int(d * math.cos(math.radians(a)))
    y += int(d * math.sin(math.radians(a)))

    return [lr, fb, ud, yv, x, y]


def take_off_land():
    if kp.getkey("t"):
        me.takeoff()
    elif kp.getkey("l"):
        me.land()


def drawPoints(img, points):
    for p in points:
        cv2.circle(img, p, 5, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, points[-1], 8, (0, 255, 0), cv2.FILLED)
    cv2.putText(
        img,
        f'({(points[-1][0] - 500) / 100:.2f},{(points[-1][1] - 500) / 100:.2f}) m',
        (points[-1][0] + 10, points[-1][1] + 30),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 225, 125),
        2
    )

# -------- MAIN LOOP --------
while True:
    take_off_land()
    values = getkeyboardInput()
    me.send_rc_control(values[0], values[1], values[2], values[3])
    sleep(0.05)

    img = np.zeros([1000, 1000, 3], np.uint8)
    points.append((values[4], values[5]))
    drawPoints(img, points)
    cv2.imshow("output", img)

    if kp.getkey("q"):
        break

# -------- CLEANUP --------
stop_video = True
video_thread.join()

tello_video.release()
cap.release()
cv2.destroyAllWindows()
me.land()
me.end()