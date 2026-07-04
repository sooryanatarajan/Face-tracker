import cv2
import mediapipe as mp
import torch
import pyautogui
import time
import math
import csv
import numpy as np
import screen_brightness_control as sbc

from pycaw.pycaw import AudioUtilities
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
from models.combined import FaceCursorModel


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0
screen_w, screen_h = pyautogui.size()

speakers = AudioUtilities.GetSpeakers()
volume = speakers.EndpointVolume

def set_volume(delta):
    v = volume.GetMasterVolumeLevelScalar()
    v = max(0.0, min(1.0, v + delta))
    volume.SetMasterVolumeLevelScalar(v, None)

model = FaceCursorModel()
model.load_state_dict(torch.load("models/face_cursor.pth", map_location="cpu"))
model.eval()

edges = torch.tensor(
    [[a, b] for a, b in FACEMESH_TESSELATION],
    dtype=torch.long
).t().contiguous()

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


cursor_mode = False
scroll_mode = False
volume_mode = False
brightness_mode = False

cursor_x, cursor_y = screen_w // 2, screen_h // 2
prev_nose = None
prev_scroll_nose_y = None
fx = fy = 0.0


RAW_GAIN = 500
MODEL_GAIN = 0.6
LOWPASS_ALPHA = 0.35
DEADZONE = 1.2
NOSE_EPS = 0.0008

ml_bias_x = ml_bias_y = 0.0
ML_BIAS_ALPHA = 0.002

MOUTH_OPEN_THRESH = 0.028
CLICK_COOLDOWN = 0.22
HEAD_RIGHT_THRESH = 0.020
last_click_time = 0.0
mouth_ready = True
head_center_x = None
HEAD_CENTER_ALPHA = 0.05

SCROLL_GAIN = 1600
SCROLL_DEADZONE = 0.0018
SCROLL_ALPHA = 0.25
SCROLL_MAX_SPEED = 120
SCROLL_COOLDOWN = 0.02
scroll_v = 0.0
last_scroll_time = 0.0

last_volume_pinch = None
last_brightness_pinch = None
VOLUME_STEP = 0.04
BRIGHTNESS_STEP = 4


log = open("cursor_log.csv", "w", newline="")
writer = csv.writer(log)
writer.writerow(["t", "x", "y"])
start_time = time.time()

def log_cursor():
    t = time.time() - start_time
    writer.writerow([t, cursor_x, cursor_y])


def nonlinear_gain(v, base=1.0, accel=2.2, max_gain=6.0):
    return v * min(base + accel * abs(v), max_gain)

def edge_boost(x, y):
    nx = abs(x - screen_w / 2) / (screen_w / 2)
    ny = abs(y - screen_h / 2) / (screen_h / 2)
    return 1.0 + 0.6 * max(nx, ny)

print("""
E = Toggle Cursor
S = Toggle Scroll
V = Toggle Volume (right hand pinch)
B = Toggle Brightness (left hand pinch)
D = Disable all
Q = Quit
""")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_res = face_mesh.process(rgb)
    hand_res = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('e'):
        cursor_mode = not cursor_mode
    elif key == ord('s'):
        scroll_mode = not scroll_mode
    elif key == ord('v'):
        volume_mode = not volume_mode
        brightness_mode = False
    elif key == ord('b'):
        brightness_mode = not brightness_mode
        volume_mode = False
    elif key == ord('d'):
        cursor_mode = scroll_mode = volume_mode = brightness_mode = False
    elif key == ord('q'):
        break

    
    if face_res.multi_face_landmarks and not (volume_mode or brightness_mode):
        face = face_res.multi_face_landmarks[0]
        nose = face.landmark[1]
        now = time.time()

        if head_center_x is None:
            head_center_x = nose.x
        else:
            head_center_x = (1-HEAD_CENTER_ALPHA)*head_center_x + HEAD_CENTER_ALPHA*nose.x

        mouth_open = abs(face.landmark[13].y - face.landmark[14].y)
        if mouth_open > MOUTH_OPEN_THRESH:
            if mouth_ready and cursor_mode and now-last_click_time > CLICK_COOLDOWN:
                pyautogui.rightClick() if nose.x-head_center_x > HEAD_RIGHT_THRESH else pyautogui.click()
                last_click_time = now
                mouth_ready = False
        else:
            mouth_ready = True

        if scroll_mode:
            if prev_scroll_nose_y is None:
                prev_scroll_nose_y = nose.y
            else:
                dy = nose.y - prev_scroll_nose_y
                prev_scroll_nose_y = nose.y
                if abs(dy) > SCROLL_DEADZONE:
                    scroll_v += dy * SCROLL_GAIN

            scroll_v *= (1 - SCROLL_ALPHA)
            scroll_v = max(-SCROLL_MAX_SPEED, min(SCROLL_MAX_SPEED, scroll_v))

            if abs(scroll_v) > 1 and now-last_scroll_time > SCROLL_COOLDOWN:
                pyautogui.scroll(int(-scroll_v))
                last_scroll_time = now

        if cursor_mode:
            if prev_nose is None:
                prev_nose = nose
            else:
                dxn = nose.x - prev_nose.x
                dyn = nose.y - prev_nose.y
                if abs(dxn) > NOSE_EPS or abs(dyn) > NOSE_EPS:
                    prev_nose = nose

                    dx_raw = dxn * RAW_GAIN
                    dy_raw = dyn * RAW_GAIN

                    x = torch.tensor([[p.x,p.y,p.z] for p in face.landmark], dtype=torch.float)
                    with torch.no_grad():
                        corr = model(x, edges)

                    ml_bias_x = (1-ML_BIAS_ALPHA)*ml_bias_x + ML_BIAS_ALPHA*corr[0].item()
                    ml_bias_y = (1-ML_BIAS_ALPHA)*ml_bias_y + ML_BIAS_ALPHA*corr[1].item()

                    dx = nonlinear_gain(dx_raw + (corr[0].item()-ml_bias_x)*MODEL_GAIN)
                    dy = nonlinear_gain(dy_raw + (corr[1].item()-ml_bias_y)*MODEL_GAIN)

                    boost = edge_boost(cursor_x, cursor_y)
                    fx = fx*(1-LOWPASS_ALPHA) + dx*LOWPASS_ALPHA*boost
                    fy = fy*(1-LOWPASS_ALPHA) + dy*LOWPASS_ALPHA*boost

                    if abs(fx) < DEADZONE: fx = 0
                    if abs(fy) < DEADZONE: fy = 0

                    cursor_x = max(0, min(screen_w, cursor_x + fx))
                    cursor_y = max(0, min(screen_h, cursor_y + fy))
                    pyautogui.moveTo(cursor_x, cursor_y)

    if hand_res.multi_hand_landmarks:
        for i, hand in enumerate(hand_res.multi_hand_landmarks):
            label = hand_res.multi_handedness[i].classification[0].label
            thumb = hand.landmark[4]
            index = hand.landmark[8]
            pinch = math.dist((thumb.x, thumb.y), (index.x, index.y))

            if volume_mode and label == "Right":
                if last_volume_pinch is not None:
                    if pinch > last_volume_pinch + 0.01:
                        set_volume(+VOLUME_STEP)
                    elif pinch < last_volume_pinch - 0.01:
                        set_volume(-VOLUME_STEP)
                last_volume_pinch = pinch

            if brightness_mode and label == "Left":
                if last_brightness_pinch is not None:
                    try:
                        if pinch > last_brightness_pinch + 0.01:
                            sbc.set_brightness(f"+{BRIGHTNESS_STEP}")
                        elif pinch < last_brightness_pinch - 0.01:
                            sbc.set_brightness(f"-{BRIGHTNESS_STEP}")
                    except Exception:
                        pass
                last_brightness_pinch = pinch

    log_cursor()

    cv2.imshow("Head Cursor Controller", frame)

cap.release()
cv2.destroyAllWindows()
log.close()

print("Cursor data saved to cursor_log.csv")

import numpy as np

print("\n=== Cursor Control Metrics ===")

data = np.genfromtxt("cursor_log.csv", delimiter=",", skip_header=1)

t = data[:, 0]
x = data[:, 1]
y = data[:, 2]

dt = np.diff(t)
dx = np.diff(x)
dy = np.diff(y)


vx = dx / dt
vy = dy / dt
speed = np.sqrt(vx**2 + vy**2)


if len(speed) > 0:
    jitter_rms = np.sqrt(np.mean(speed**2))
else:
    jitter_rms = 0.0

print(f"Jitter (RMS velocity): {jitter_rms:.6f} px/s")


total_time = t[-1] - t[0]
total_disp = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)

drift = total_disp / total_time if total_time > 0 else 0.0
print(f"Drift: {drift:.6f} px/s")


TARGET_X = x[0]
TARGET_Y = y[0]

dist = np.sqrt((x - TARGET_X)**2 + (y - TARGET_Y)**2)

if dist[0] > 1e-6:
    overshoot = max(0.0, (dist[0] - np.min(dist)) / dist[0]) * 100
    print(f"Overshoot: {overshoot:.2f} %")
else:
    print("Overshoot: N/A (no initial movement)")

# Settling time
EPS = 10  # pixels
settled = np.where(dist < EPS)[0]

if len(settled) > 0:
    settling_time = t[settled[0]]
    print(f"Settling time: {settling_time:.3f} s")
else:
    print("Settling time: Not settled")

print("================================\n")
