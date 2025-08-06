import cv2
import numpy as np
import mediapipe as mp
import time
import threading
from scipy.spatial import distance as dist
import os
import urllib.request
import pygame

ALERT_FILE = "beep.wav"
if not os.path.exists(ALERT_FILE):
    print("[INFO] Downloading alert sound...")
    try:
        url = "https://github.com/zaps166/media-sound-files/raw/master/wav/beep1.wav"
        urllib.request.urlretrieve(url, ALERT_FILE)
    except Exception as e:
        print(f"[WARNING] Failed to download alert sound. Error: {e}")
        print("Please manually add 'beep.wav' in the project directory.")
        exit(1)

pygame.mixer.init()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)

HEAD_PITCH_UP_THRESHOLD = -15
HEAD_PITCH_DOWN_THRESHOLD = 25
HEAD_YAW_THRESHOLD = 25
POSE_TIME_THRESHOLD = 1.5

EAR_THRESHOLD = 0.21
GAZE_SIDE_THRESHOLD = 0.4
GAZE_DOWN_THRESHOLD = 0.65
COGNITIVE_TIME_THRESHOLD = 8
DROWSY_TIME_THRESHOLD = 1.5
GAZE_TIME_THRESHOLD = 2.0

WARNING_DELAY = 4.0

HEAD_POSE_LANDMARKS = [1, 152, 226, 446, 57, 287]
LEFT_EYE, RIGHT_EYE = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]
LEFT_IRIS, RIGHT_IRIS = [468, 469, 470, 471], [473, 474, 475, 476]

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(eye):
    return (euclidean(eye[1], eye[5]) + euclidean(eye[2], eye[4])) / (2.0 * euclidean(eye[0], eye[3]))

def get_head_pose(landmarks, cam_matrix, dist_coeffs):
    model_points = np.array([(0.0, 0.0, 0.0), (0.0, -63.6, -12.5),
                             (-43.3, 32.7, -26.0), (43.3, 32.7, -26.0),
                             (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)], dtype=np.float64)
    success, rvec, _ = cv2.solvePnP(model_points, landmarks, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if success:
        R, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        if sy >= 1e-6:
            pitch = np.rad2deg(np.arctan2(R[2, 1], R[2, 2]))
            yaw = np.rad2deg(np.arctan2(-R[2, 0], sy))
            return pitch, yaw
    return None, None

def get_gaze_direction(left_eye, right_eye, left_iris, right_iris):
    def get_norm_pos(eye, iris):
        x_min, x_max = min(pt[0] for pt in eye), max(pt[0] for pt in eye)
        y_min, y_max = min(pt[1] for pt in eye), max(pt[1] for pt in eye)
        iris_center = np.mean(iris, axis=0)
        norm_x = (iris_center[0] - x_min) / (x_max - x_min + 1e-6)
        norm_y = (iris_center[1] - y_min) / (y_max - y_min + 1e-6)
        return norm_x, norm_y

    norm_x_left, norm_y_left = get_norm_pos(left_eye, left_iris)
    norm_x_right, norm_y_right = get_norm_pos(right_eye, right_iris)
    avg_norm_x = (norm_x_left + norm_x_right) / 2.0
    avg_norm_y = (norm_y_left + norm_y_right) / 2.0

    if avg_norm_y > GAZE_DOWN_THRESHOLD:
        return "DOWN"
    elif avg_norm_x < GAZE_SIDE_THRESHOLD:
        return "LEFT"
    elif avg_norm_x > 1 - GAZE_SIDE_THRESHOLD:
        return "RIGHT"
    else:
        return "CENTER"

def play_alert_sound():
    pygame.mixer.Sound(ALERT_FILE).play()

cap = cv2.VideoCapture(0)
camera_matrix, dist_coeffs = None, None
is_calibrated = False
forward_pose_offset = None
distraction_timers = { "HEAD": 0, "DROWSY": 0, "COGNITIVE": 0, "GAZE": 0 }
warning_timers = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    status_text, status_color = "ATTENTIVE", (0, 255, 0)

    if results.multi_face_landmarks:
        mesh_points = np.array([(p.x * w, p.y * h) for p in results.multi_face_landmarks[0].landmark])

        if not is_calibrated:
            status_text, status_color = "CALIBRATING... LOOK FORWARD", (0, 255, 255)
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            pitch, yaw = get_head_pose(mesh_points[HEAD_POSE_LANDMARKS].astype(np.float64), camera_matrix, dist_coeffs)
            if pitch is not None:
                forward_pose_offset = np.array([pitch, yaw])
                is_calibrated = True
                print(f"[INFO] Calibration Complete: Forward Pose Offset = {forward_pose_offset.round(1)}")
        else:
            final_distraction = None
            avg_ear = (calculate_ear(mesh_points[LEFT_EYE]) + calculate_ear(mesh_points[RIGHT_EYE])) / 2.0
            gaze_direction = get_gaze_direction(mesh_points[LEFT_EYE], mesh_points[RIGHT_EYE], mesh_points[LEFT_IRIS], mesh_points[RIGHT_IRIS])
            pitch, yaw = get_head_pose(mesh_points[HEAD_POSE_LANDMARKS].astype(np.float64), camera_matrix, dist_coeffs)

            if avg_ear < EAR_THRESHOLD:
                if distraction_timers["DROWSY"] == 0:
                    distraction_timers["DROWSY"] = time.time()
                elif time.time() - distraction_timers["DROWSY"] > DROWSY_TIME_THRESHOLD:
                    final_distraction = "DROWSY"
            else:
                distraction_timers["DROWSY"] = 0

            if final_distraction is None and pitch is not None:
                relative_pitch = pitch - forward_pose_offset[0]
                relative_yaw = yaw - forward_pose_offset[1]
                if relative_pitch < HEAD_PITCH_UP_THRESHOLD or relative_pitch > HEAD_PITCH_DOWN_THRESHOLD or abs(relative_yaw) > HEAD_YAW_THRESHOLD:
                    if distraction_timers["HEAD"] == 0:
                        distraction_timers["HEAD"] = time.time()
                    elif time.time() - distraction_timers["HEAD"] > POSE_TIME_THRESHOLD:
                        final_distraction = "HEAD TURNED AWAY"
                else:
                    distraction_timers["HEAD"] = 0

            if final_distraction is None:
                if avg_ear < EAR_THRESHOLD + 0.04:
                    distraction_timers["COGNITIVE"] = 0
                else:
                    if distraction_timers["COGNITIVE"] == 0:
                        distraction_timers["COGNITIVE"] = time.time()
                    elif time.time() - distraction_timers["COGNITIVE"] > COGNITIVE_TIME_THRESHOLD:
                        final_distraction = "COGNITIVE (Staring)"

            if final_distraction is None and gaze_direction != "CENTER":
                if distraction_timers["GAZE"] == 0:
                    distraction_timers["GAZE"] = time.time()
                elif time.time() - distraction_timers["GAZE"] > GAZE_TIME_THRESHOLD:
                    final_distraction = f"GAZE ({gaze_direction})"
            else:
                distraction_timers["GAZE"] = 0

            if final_distraction:
                now = time.time()
                if final_distraction not in warning_timers:
                    warning_timers[final_distraction] = now
                    status_text, status_color = f"WARNING: {final_distraction}", (0, 165, 255)
                elif now - warning_timers[final_distraction] < WARNING_DELAY:
                    status_text, status_color = f"WARNING: {final_distraction}", (0, 165, 255)
                else:
                    status_text, status_color = f"ALERT: {final_distraction}!!", (0, 0, 255)
                    play_alert_sound()
            else:
                warning_timers.clear()

            for pt_idx in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, tuple(mesh_points[pt_idx].astype(int)), 2, (0, 255, 0), -1)
            for pt_idx in HEAD_POSE_LANDMARKS:
                cv2.circle(frame, tuple(mesh_points[pt_idx].astype(int)), 3, (0, 255, 255), -1)

    else:
        status_text, status_color = "NO FACE DETECTED", (0, 0, 255)

    cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.imshow("Driver Monitoring System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
