from djitellopy import tello
import keyPressModule as kp
from time import  sleep


kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())

def getkeyboardInput():
    lr,fb,ud,yv = 0, 0 ,0 ,0
    speed = 40

    if kp.getkey("LEFT"): lr = -speed
    elif kp.getkey("RIGHT"): lr = speed

    elif kp.getkey("UP"): fb = speed
    elif kp.getkey("DOWN"): fb = -speed

    elif kp.getkey("w"): ud = speed
    elif kp.getkey("s"): ud = -speed

    elif kp.getkey("a"): yv = -speed
    elif kp.getkey("d"): yv = speed

    return [lr,fb,ud,yv]


def take_off_land():
    if kp.getkey("h"):
        me.takeoff()
    elif kp.getkey("l"):
        me.land()

while True:
    take_off_land()
    values = getkeyboardInput()
    me.send_rc_control(values[0],values[1],values[2],values[3])
    sleep(0.05)