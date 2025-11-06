# models/posture_model.py
import mediapipe as mp
import cv2
import math
import numpy as np

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                             min_detection_confidence=0.5)

# helper: angle between three points (p1-p2-p3) in degrees
def angle_between(p1, p2, p3):
    a = np.array([p1.x - p2.x, p1.y - p2.y])
    b = np.array([p3.x - p2.x, p3.y - p2.y])
    cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    return ang

def analyze_posture(frame, debug=False):
    """
    Returns: posture_label, eye_contact_label, annotated_frame, info_dict
    info_dict contains metrics used for decision (angles, gaps, landmark counts)
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    pose_res = pose.process(img)
    face_res = face_mesh.process(img)

    posture = "Not Detected"
    eye_contact = "Unknown"
    annotated = frame.copy()
    info = {}

    # posture via torso angle: shoulders-hip line relative to vertical
    if pose_res.pose_landmarks and len(pose_res.pose_landmarks.landmark) >= 25:
        lm = pose_res.pose_landmarks.landmark
        # choose left/right shoulders and hips
        left_sh = lm[11]; right_sh = lm[12]
        left_hip = lm[23]; right_hip = lm[24]

        # midpoints
        mid_sh = mp.math_helpers._NormalizedLandmark( # fallback if not available; but easier to compute manually
            x=(left_sh.x + right_sh.x)/2, y=(left_sh.y + right_sh.y)/2
        ) if False else None

        # compute torso angle using vector from mid-shoulder to mid-hip
        mid_sh_x, mid_sh_y = (left_sh.x + right_sh.x)/2, (left_sh.y + right_sh.y)/2
        mid_hp_x, mid_hp_y = (left_hip.x + right_hip.x)/2, (left_hip.y + right_hip.y)/2
        dx = mid_hp_x - mid_sh_x
        dy = mid_hp_y - mid_sh_y
        torso_angle_deg = abs(math.degrees(math.atan2(dy, dx)))  # angle in normalized coords
        # normalized coords: vertical corresponds to ~90 degrees; compute deviation from vertical
        deviation = abs(90 - torso_angle_deg)
        # Convert to a more intuitive metric: smaller deviation -> more vertical/upright
        # Use thresholds tuned by experiment
        if deviation < 8:
            posture = "Good"
        elif deviation < 18:
            posture = "Slight slouch"
        else:
            posture = "Slouching"

        # draw shoulders/hips and line
        cv2.circle(annotated, (int(left_sh.x*w), int(left_sh.y*h)), 4, (0,255,0), -1)
        cv2.circle(annotated, (int(right_sh.x*w), int(right_sh.y*h)), 4, (0,255,0), -1)
        cv2.circle(annotated, (int(left_hip.x*w), int(left_hip.y*h)), 4, (0,255,0), -1)
        cv2.circle(annotated, (int(right_hip.x*w), int(right_hip.y*h)), 4, (0,255,0), -1)
        cv2.line(annotated, (int(mid_sh_x*w), int(mid_sh_y*h)), (int(mid_hp_x*w), int(mid_hp_y*h)), (255,0,0), 2)

        info['torso_angle_deg'] = torso_angle_deg
        info['deviation'] = deviation
    else:
        info['pose_landmarks'] = 0

    # eye contact via face center and nose orientation fallback
    if face_res.multi_face_landmarks:
        fl = face_res.multi_face_landmarks[0].landmark
        info['face_landmarks'] = len(fl)
        # prefer nose tip or midpoint of eyes to compute face center
        # Mediapipe iris landmarks 468/473 exist when refine_landmarks=True
        if len(fl) > 473:
            l_eye = fl[468]; r_eye = fl[473]
            # horizontal gap between irises — if small, face looking forward
            eye_gap = abs(l_eye.x - r_eye.x)
            # compute eye midpoint x and nose tip x
            nose_tip = fl[1]  # landmark 1 is nose_tip (approx)
            eye_mid_x = (l_eye.x + r_eye.x)/2
            # compare nose x with eye midpoint, closer to center -> forward
            nose_vs_center = abs(nose_tip.x - eye_mid_x)
            # thresholds tuned experimentally
            if eye_gap < 0.08 and nose_vs_center < 0.03:
                eye_contact = "Maintained"
            else:
                eye_contact = "Looking Away"

            # draw some markers
            cv2.circle(annotated, (int(l_eye.x*w), int(l_eye.y*h)), 2, (0,255,255), -1)
            cv2.circle(annotated, (int(r_eye.x*w), int(r_eye.y*h)), 2, (0,255,255), -1)
            cv2.circle(annotated, (int(nose_tip.x*w), int(nose_tip.y*h)), 2, (0,255,0), -1)

            info['eye_gap'] = eye_gap
            info['nose_vs_center'] = nose_vs_center
        else:
            eye_contact = "Face partly detected"
    else:
        info['face_landmarks'] = 0
        eye_contact = "Face not detected"

    # overlay labels
    cv2.putText(annotated, f"Posture: {posture}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(annotated, f"Eye: {eye_contact}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    return posture, eye_contact, annotated, info
