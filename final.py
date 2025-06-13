import cv2
import mediapipe as mp
import numpy as np
import math
import pyttsx3

# Text-to-speech setup
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
last_announced_scene = ""

# Pose setup
canvas_size = 512
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
connections = mp_pose.POSE_CONNECTIONS

# Camera setup (0 = default webcam, or replace with IP stream or index)
cap = cv2.VideoCapture(0)

#cap = cv2.VideoCapture("http://100.81.129.188:8080/video")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Discard original image for privacy
    del image_rgb
    del frame

    # Blank white canvas
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    scene = "No Person Detected"

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Normalize landmarks
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        box_width = max_x - min_x
        box_height = max_y - min_y

        scale = 0.8 * canvas_size / max(box_width, box_height)
        offset_x = (canvas_size - scale * (min_x + max_x)) / 2
        offset_y = (canvas_size - scale * (min_y + max_y)) / 2

        # Draw skeleton connections
        for start_idx, end_idx in connections:
            lm1 = landmarks[start_idx]
            lm2 = landmarks[end_idx]
            if lm1.visibility > 0.5 and lm2.visibility > 0.5:
                x1 = int(lm1.x * scale + offset_x)
                y1 = int(lm1.y * scale + offset_y)
                x2 = int(lm2.x * scale + offset_x)
                y2 = int(lm2.y * scale + offset_y)
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw joints
        for lm in landmarks:
            if lm.visibility > 0.5:
                x = int(lm.x * scale + offset_x)
                y = int(lm.y * scale + offset_y)
                cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)

        # --- Scene Detection Logic ---
        def avg(a, b): return (a + b) / 2
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        torso_x = avg(right_hip.x, left_hip.x) - avg(right_shoulder.x, left_shoulder.x)
        torso_y = avg(right_hip.y, left_hip.y) - avg(right_shoulder.y, right_shoulder.y)
        angle = abs(math.degrees(math.atan2(torso_x, torso_y)))
        if angle > 90:
            angle = 180 - angle

        z_std = np.std([lm.z for lm in landmarks])

        lying_votes = 0
        if box_width > box_height:
            lying_votes += 1
        if angle > 30:
            lying_votes += 1
        if z_std < 0.1:
            lying_votes += 1

        if lying_votes >= 2:
            scene = "Person NOT OK"
        else:
            scene = "Person OK"

    # --- Voice Alert Only for "Person NOT OK" ---
    if scene == "Person NOT OK" and scene != last_announced_scene:
        tts_engine.say("Person not okay!")
        tts_engine.runAndWait()

    last_announced_scene = scene

    # Display scene label
    cv2.putText(canvas, scene, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Show canvas
    cv2.imshow("SCENE-DETECT: Stick Figure View", canvas)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
