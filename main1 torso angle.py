import cv2
import mediapipe as mp
import numpy as np
import math

# Pose Estimation Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

# Load Image
image_path = 'Person not OK/Failed 2.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Save original dimensions
h, w, _ = image.shape

# Pose Detection
results = pose.process(image_rgb)

# Discard original image (Privacy)
del image
del image_rgb

# Create Blank Canvas
canvas_size = 512
skeleton_canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255  # White background

# Draw Skeleton if Detected
if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark

    # Find bounding box
    landmark_x = [lm.x for lm in landmarks]
    landmark_y = [lm.y for lm in landmarks]
    min_x, max_x = min(landmark_x), max(landmark_x)
    min_y, max_y = min(landmark_y), max(landmark_y)

    # Calculate scale and offset
    box_width = max_x - min_x
    box_height = max_y - min_y
    scale = 0.8 * canvas_size / max(box_width, box_height)  # 80% of canvas size
    offset_x = (canvas_size - scale * (min_x + max_x)) / 2
    offset_y = (canvas_size - scale * (min_y + max_y)) / 2

    # Draw all landmarks and connections
    connections = mp_pose.POSE_CONNECTIONS

    for connection in connections:
        start_idx, end_idx = connection
        if landmarks[start_idx].visibility > 0.5 and landmarks[end_idx].visibility > 0.5:
            x_start = int(landmarks[start_idx].x * scale + offset_x)
            y_start = int(landmarks[start_idx].y * scale + offset_y)
            x_end = int(landmarks[end_idx].x * scale + offset_x)
            y_end = int(landmarks[end_idx].y * scale + offset_y)
            cv2.line(skeleton_canvas, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    for lm in landmarks:
        if lm.visibility > 0.5:
            x = int(lm.x * scale + offset_x)
            y = int(lm.y * scale + offset_y)
            cv2.circle(skeleton_canvas, (x, y), 3, (0, 0, 255), -1)

    # Scene Classification Logic (Same as Before)
    torso_vector_x = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x + landmarks[mp_pose.PoseLandmark.LEFT_HIP].x)/2 - (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x)/2
    torso_vector_y = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y + landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)/2 - (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)/2

    angle_rad = math.atan2(torso_vector_x, torso_vector_y)
    angle_deg = abs(math.degrees(angle_rad))
    if angle_deg > 90:
        angle_deg = 180 - angle_deg

    z_values = np.array([lm.z for lm in landmarks])
    z_std_dev = np.std(z_values)

    lying_votes = 0
    if box_width > box_height:
        lying_votes += 1
    if angle_deg > 30:
        lying_votes += 1
    # if z_std_dev < 0.1:
    #     lying_votes += 1

    if lying_votes >= 1:
        scene = "Person NOT OK (Lying/Collapsed)"
    else:
        scene = "Person OK (Standing/Sitting)"

else:
    scene = "No Person Detected"

# Display Scene Label
cv2.putText(skeleton_canvas, scene, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Show Image
cv2.imshow('Skeleton Scene Detection', skeleton_canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
