from ultralytics import YOLO
import cv2
import numpy as np
import pyttsx3
import math

# Initialize voice engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
last_alert = None

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")  # Use yolov8m-pose.pt or l/x for better accuracy

# Camera setup
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("https://192.168.155.55:8080/video")

canvas_size = 512

def classify_scene(kpts):
    # kpts: np.array shape (17, 3) - x, y, conf
    visible = kpts[:, 2] > 0.5
    xs = kpts[visible, 0]
    ys = kpts[visible, 1]
    zs = kpts[visible, 1]  # simulate z from y for now (YOLOv8 has no real z)

    if len(xs) == 0 or len(ys) == 0:
        return "No Person"

    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    box_width = max_x - min_x
    box_height = max_y - min_y
    z_std = np.std(zs)

    # Torso angle calculation
    try:
        # Shoulders and hips
        LS, RS, LH, RH = kpts[5], kpts[6], kpts[11], kpts[12]
        if all(p[2] > 0.5 for p in [LS, RS, LH, RH]):
            torso_x = (LH[0] + RH[0])/2 - (LS[0] + RS[0])/2
            torso_y = (LH[1] + RH[1])/2 - (LS[1] + RS[1])/2
            angle = abs(math.degrees(math.atan2(torso_x, torso_y)))
            if angle > 90:
                angle = 180 - angle
        else:
            angle = 0
    except:
        angle = 0

    # Voting
    votes = 0
    if box_width > box_height:
        votes += 1
    if angle > 30:
        votes += 1
    if z_std < 20:
        votes += 1

    return "Person NOT OK" if votes >= 2 else "Person OK"

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5, verbose=False)

    # Create blank canvas
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    scene_states = []

    for person in results[0].keypoints.data.cpu().numpy():
        kpts = person.reshape(-1, 3)
        scene = classify_scene(kpts)
        scene_states.append(scene)

        # Normalize for canvas size (scale + offset)
        xs = kpts[:, 0]
        ys = kpts[:, 1]
        conf = kpts[:, 2]

        valid = conf > 0.5
        if not np.any(valid):
            continue  # skip this person, no visible keypoints

        xs = xs[valid]
        ys = ys[valid]

        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)
        box_w = max_x - min_x
        box_h = max_y - min_y
        scale = 0.8 * canvas_size / max(box_w, box_h + 1e-6)
        offset_x = (canvas_size - scale * (min_x + max_x)) / 2
        offset_y = (canvas_size - scale * (min_y + max_y)) / 2

        # Draw keypoints and connections
        for i, (x, y, conf) in enumerate(kpts):
            if conf < 0.5:
                continue
            cx = int(x * scale + offset_x)
            cy = int(y * scale + offset_y)
            cv2.circle(canvas, (cx, cy), 3, (0, 0, 255), -1)

        # Draw connections (hardcoded 17-point COCO format)
        skeleton = [
            (5, 7), (7, 9),  # left arm
            (6, 8), (8, 10),  # right arm
            (11, 13), (13, 15),  # left leg
            (12, 14), (14, 16),  # right leg
            (5, 6), (11, 12),  # shoulders, hips
            (5, 11), (6, 12),  # torso
        ]
        for a, b in skeleton:
            if kpts[a][2] > 0.5 and kpts[b][2] > 0.5:
                x1 = int(kpts[a][0] * scale + offset_x)
                y1 = int(kpts[a][1] * scale + offset_y)
                x2 = int(kpts[b][0] * scale + offset_x)
                y2 = int(kpts[b][1] * scale + offset_y)
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Decision: if anyone is NOT OK → speak once
    if "Person NOT OK" in scene_states and last_alert != "Person NOT OK":
        tts_engine.say("Alert. Person not okay.")
        tts_engine.runAndWait()
        last_alert = "Person NOT OK"
    elif "Person NOT OK" not in scene_states:
        last_alert = "OK"

    # Display result
    label = " | ".join(scene_states) if scene_states else "No Person Detected"
    cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow("SCENE-DETECT (Multi-Person)", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
