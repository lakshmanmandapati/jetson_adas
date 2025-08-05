import cv2
import numpy as np
import mediapipe as mp
import time
import threading
from scipy.spatial import distance as dist
import os
import urllib.request
import simpleaudio as sa

# --- Alert Sound Setup ---
ALERT_FILE = "beep.wav"
if not os.path.exists(ALERT_FILE):
    print("[INFO] Downloading alert sound...")
    try:
        urllib.request.urlretrieve(
            "https://github.com/zaps166/media-sound-files/raw/master/wav/beep1.wav",
            ALERT_FILE
        )
    except Exception as e:
        print(f"[ERROR] Could not download alert: {e}")
        exit(1)

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Constants and Thresholds ---
HEAD_PITCH_UP = -15
HEAD_PITCH_DOWN = 25
HEAD_YAW_THRESHOLD = 25
POSE_HOLD = 1.5

EAR_THRESHOLD = 0.21
GAZE_SIDE = 0.4
GAZE_DOWN = 0.65
GAZE_HOLD = 2.0
DROWSY_HOLD = 1.5
COGNITIVE_HOLD = 8.0

BEEP_COOLDOWN = 3.0

# --- Landmark Indices ---
HEAD_IDX = [1, 152, 226, 446, 57, 287]
LEFT_EYE, RIGHT_EYE = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]
LEFT_IRIS, RIGHT_IRIS = [468, 469, 470, 471], [473, 474, 475, 476]

# --- Utility Functions ---
def euclidean(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))
def calc_ear(eye): return (euclidean(eye[1], eye[5]) + euclidean(eye[2], eye[4])) / (2 * euclidean(eye[0], eye[3]))

def play_alert():
    global last_beep_time
    now = time.time()
    if now - last_beep_time > BEEP_COOLDOWN:
        last_beep_time = now
        threading.Thread(target=sa.WaveObject.from_wave_file(ALERT_FILE).play, daemon=True).start()

def get_head_pose(pts, cam_matrix, dist_coeffs):
    model = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ], dtype=np.float64)
    success, rvec, _ = cv2.solvePnP(model, pts, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success: return None, None
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    return pitch, yaw

def get_gaze(left_eye, right_eye, left_iris, right_iris):
    def norm_pos(eye, iris):
        x_min, x_max = min(pt[0] for pt in eye), max(pt[0] for pt in eye)
        y_min, y_max = min(pt[1] for pt in eye), max(pt[1] for pt in eye)
        iris_center = np.mean(iris, axis=0)
        return (iris_center[0] - x_min) / (x_max - x_min + 1e-6), (iris_center[1] - y_min) / (y_max - y_min + 1e-6)
    xL, yL = norm_pos(left_eye, left_iris)
    xR, yR = norm_pos(right_eye, right_iris)
    x, y = (xL + xR) / 2, (yL + yR) / 2
    if y > GAZE_DOWN: return "DOWN"
    elif x < GAZE_SIDE: return "LEFT"
    elif x > 1 - GAZE_SIDE: return "RIGHT"
    return "CENTER"

# --- Initialize Video ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_beep_time = 0
calibrated = False
cam_matrix, dist_coeffs = None, None
forward_offset = None
timers = {"DROWSY": 0, "HEAD": 0, "GAZE": 0, "COGNITIVE": 0}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    text, color = "ATTENTIVE", (0, 255, 0)

    if results.multi_face_landmarks:
        mesh = np.array([(p.x * w, p.y * h) for p in results.multi_face_landmarks[0].landmark])
        if not calibrated:
            focal = w
            cam_matrix = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))
            pitch, yaw = get_head_pose(mesh[HEAD_IDX], cam_matrix, dist_coeffs)
            if pitch is not None:
                forward_offset = np.array([pitch, yaw])
                calibrated = True
                print("[INFO] Calibrated. Offset:", forward_offset.round(1))
                continue
            text, color = "CALIBRATING...", (0, 255, 255)
        else:
            pitch, yaw = get_head_pose(mesh[HEAD_IDX], cam_matrix, dist_coeffs)
            ear = (calc_ear(mesh[LEFT_EYE]) + calc_ear(mesh[RIGHT_EYE])) / 2
            gaze = get_gaze(mesh[LEFT_EYE], mesh[RIGHT_EYE], mesh[LEFT_IRIS], mesh[RIGHT_IRIS])

            distraction = None
            now = time.time()

            if ear < EAR_THRESHOLD:
                if timers["DROWSY"] == 0: timers["DROWSY"] = now
                elif now - timers["DROWSY"] > DROWSY_HOLD: distraction = "DROWSY"
            else: timers["DROWSY"] = 0

            if pitch is not None and distraction is None:
                d_pitch, d_yaw = pitch - forward_offset[0], yaw - forward_offset[1]
                if d_pitch < HEAD_PITCH_UP or d_pitch > HEAD_PITCH_DOWN or abs(d_yaw) > HEAD_YAW_THRESHOLD:
                    if timers["HEAD"] == 0: timers["HEAD"] = now
                    elif now - timers["HEAD"] > POSE_HOLD: distraction = "HEAD TURNED"
                else: timers["HEAD"] = 0

            if distraction is None and gaze != "CENTER":
                if timers["GAZE"] == 0: timers["GAZE"] = now
                elif now - timers["GAZE"] > GAZE_HOLD: distraction = f"GAZE {gaze}"
            else: timers["GAZE"] = 0

            if distraction is None:
                if ear > EAR_THRESHOLD + 0.04:
                    if timers["COGNITIVE"] == 0: timers["COGNITIVE"] = now
                    elif now - timers["COGNITIVE"] > COGNITIVE_HOLD:
                        distraction = "COGNITIVE"
                else: timers["COGNITIVE"] = 0

            if distraction:
                text, color = f"ALERT: {distraction}", (0, 0, 255)
                play_alert()

    else:
        text, color = "NO FACE", (0, 0, 255)

    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.imshow("Driver Monitor", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
