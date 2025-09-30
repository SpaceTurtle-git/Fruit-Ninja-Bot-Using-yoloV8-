from ultralytics import YOLO
import cv2
import numpy 
import pyautogui
import keyboard
import time
import mss 
import torch
import threading


# GPU or CPU
if torch.cuda.is_available():
    device = 0
    print(f"GPU detected: {torch.cuda.get_device_name(0)} — Using device {device}")
else:
    device = "cpu"
    print("No GPU detected — Using CPU")


# ===== CONFIG =====
model_path = r"C:\Users\rockf\Documents\dumbshi\goon_data\yolov8 v2\runs\detect\train\weights\best.pt"
bot = YOLO(model_path)
sct = mss.mss()
top_right,top_left,width,height = 0,0,1280,720
frame_counter = 0
x11,y11,x22,y22 = 0,0,0,0
coords_lock = threading.Lock()
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

paused = False  # PAUSE FLAG


# Video writer
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("fruit_ninja_ai.avi", fourcc, 20, (width, height))


# Functions
def screenshotMachine():
    monitor = {"top": top_right, "left": top_left, "width": width, "height": height}
    img = numpy.array(sct.grab(monitor))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return frame


def slashMachine():
    global x11,y11,x22,y22
    while not keyboard.is_pressed('q'):
        if paused:  # Pause control
            time.sleep(0.1)
            continue
        with coords_lock:
            if x11 != 0 and y11 != 0 and x22 != 0 and y22 != 0:
                pyautogui.moveTo(x11,y11)
                pyautogui.mouseDown()
                pyautogui.moveTo(x22,y22,duration=0.15)
                pyautogui.moveTo(x22,y11,duration=0.1)
                pyautogui.moveTo(x11,y22,duration=0.1)


# ======== Main Program ========
threading.Thread(target=slashMachine, daemon=True).start()

while not keyboard.is_pressed('q'):

    # Toggle pause
    if keyboard.is_pressed('p'):
        paused = not paused
        print(f"Paused = {paused}")
        time.sleep(0.3)  # Prevent fast toggling

    if paused:
        time.sleep(0.1)
        continue

    frame = screenshotMachine()
    results = bot(frame, imgsz=640, device=device)

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            label = bot.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label != 'bomb':
                x11,y11,x22,y22 = x1,y1,x2,y2

            color = (0,255,0) if label != "bomb" else (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Fruit Ninja AI", frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()
