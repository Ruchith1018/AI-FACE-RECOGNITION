import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

def detect_head_pose(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    lm = results.multi_face_landmarks[0].landmark
    h, w = img.shape[:2]

    nose = lm[1]
    left = lm[33]
    right = lm[263]
    forehead = lm[10]

    nx, ny = nose.x*w, nose.y*h
    lx, rx = left.x*w, right.x*w
    fx, fy = forehead.x*w, forehead.y*h

    eye_center_x = (lx + rx) / 2
    yaw = (nx - eye_center_x) / w * 100
    pitch = (ny - fy) / h * 100

    return yaw, pitch
