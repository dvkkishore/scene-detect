from ultralytics import YOLO
import cv2
import numpy as np
import pyttsx3
import math

# Init voice
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
last_alert = None

# Load YOLOv8-pose model
model = YOLO("yolov8n-pose.pt")  # or yolov8m/l/x-pose.pt

# Canvas setup
canvas_size = 512
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("https://192.168.155.55:8080/video")

# Scene classifier
def classify_scene(kpts):
    visible = kpts[:, 2] > 0.5
    xs = kpts[visible, 0]
    ys = kpts[visible, 1]
    zs = kpts[visible, 1]  # simulate Z from Y

    if len(xs) == 0 or len(ys) == 0:
        return "No Person"

    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    box_width = max_x - min_x
    box_height = max_y - min_y
    z_std = np.std(zs)

    try:
        LS, RS, LH, RH = kpts[5], kpts[6], kpts[11], kpts[12]
        if all(p[2] > 0.5 for p in [LS, RS, LH, RH]):
            torso_x = (LH[0] + RH[0]) / 2 - (LS[0] + RS[0]) / 2
            torso_y = (LH[1] + RH[1]) / 2 - (LS[1] + RS[1]) / 2
            angle = abs(math.degrees(math.atan2(torso_x, torso_y)))
            if angle > 90:
                angle = 180 - angle
        else:
            angle = 0
    except:
        angle = 0

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

    # Run pose prediction
    results = model.predict(frame, conf=0.5, verbose=False)

    # Delete image immediately
    del frame
    results[0].orig_img = None  # also clear internal cached copy

    # Blank white canvas
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    scene_states = []

    # Global bounding box (for consistent scale)
    all_kpts = results[0].keypoints.data.cpu().numpy().reshape(-1, 3)
    valid_all = all_kpts[:, 2] > 0.5

    if not np.any(valid_all):
        cv2.putText(canvas, "No Person Detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imshow("SCENE-DETECT (Privacy Clean)", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    global_xs = all_kpts[valid_all, 0]
    global_ys = all_kpts[valid_all, 1]
    min_x, max_x = np.min(global_xs), np.max(global_xs)
    min_y, max_y = np.min(global_ys), np.max(global_ys)
    box_w = max_x - min_x
    box_h = max_y - min_y
    scale = 0.8 * canvas_size / max(box_w, box_h + 1e-6)
    offset_x = (canvas_size - scale * (min_x + max_x)) / 2
    offset_y = (canvas_size - scale * (min_y + max_y)) / 2

    # Process each person
    for person in results[0].keypoints.data.cpu().numpy():
        kpts = person.reshape(-1, 3)
        conf = kpts[:, 2]
        if not np.any(conf > 0.5):
            continue

        scene = classify_scene(kpts)
        scene_states.append(scene)

        # Draw joints
        for i, (x, y, c) in enumerate(kpts):
            if c < 0.5:
                continue
            cx = int(x * scale + offset_x)
            cy = int(y * scale + offset_y)
            cv2.circle(canvas, (cx, cy), 3, (0, 0, 255), -1)

        # Draw skeleton
        skeleton = [
            (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (5, 6), (11, 12), (5, 11), (6, 12)
        ]
        for a, b in skeleton:
            if kpts[a][2] > 0.5 and kpts[b][2] > 0.5:
                x1 = int(kpts[a][0] * scale + offset_x)
                y1 = int(kpts[a][1] * scale + offset_y)
                x2 = int(kpts[b][0] * scale + offset_x)
                y2 = int(kpts[b][1] * scale + offset_y)
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 📍 Label placement: torso center
        try:
            LS, RS, LH, RH = kpts[5], kpts[6], kpts[11], kpts[12]
            if all(p[2] > 0.5 for p in [LS, RS, LH, RH]):
                label_x = int(((LS[0] + RS[0] + LH[0] + RH[0]) / 4) * scale + offset_x)
                label_y = int(((LS[1] + RS[1] + LH[1] + RH[1]) / 4) * scale + offset_y)
            else:
                label_x, label_y = 20, 20
        except:
            label_x, label_y = 20, 20

        label_color = (0, 255, 0) if scene == "Person OK" else (0, 0, 255)
        cv2.putText(canvas, scene, (label_x, label_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

    # 🔈 Voice alert if any NOT OK
    if "Person NOT OK" in scene_states and last_alert != "Person NOT OK":
        tts_engine.say("Alert. Person not okay.")
        tts_engine.runAndWait()
        last_alert = "Person NOT OK"
    elif "Person NOT OK" not in scene_states:
        last_alert = "OK"

    # Show result
    cv2.imshow("SCENE-DETECT (Privacy Clean)", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
