import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

# Load your test image
image_path = 'Person not OK/3.jpg'  # Change to your image file path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to extract pose
results = pose.process(image_rgb)
print(results)

# Draw skeleton if detected
if results.pose_landmarks:
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    landmarks = results.pose_landmarks.landmark
    
    # Bounding box calculation
    landmark_x = [lm.x for lm in landmarks]
    landmark_y = [lm.y for lm in landmarks]
    min_x, max_x = min(landmark_x), max(landmark_x)
    min_y, max_y = min(landmark_y), max(landmark_y)
    width = max_x - min_x
    height = max_y - min_y

    # Draw bounding box
    h, w, _ = image.shape
    start_point = (int(min_x * w), int(min_y * h))
    end_point = (int(max_x * w), int(max_y * h))
    cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)

    # Torso angle calculation
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    torso_vector_x = (right_hip.x + left_hip.x)/2 - (right_shoulder.x + left_shoulder.x)/2
    torso_vector_y = (right_hip.y + left_hip.y)/2 - (right_shoulder.y + left_shoulder.y)/2

    angle_rad = math.atan2(torso_vector_x, torso_vector_y)
    angle_deg = abs(math.degrees(angle_rad))
    if angle_deg > 90:
        angle_deg = 180 - angle_deg

    # Fusion decision logic
    lying_votes = 0

    # Condition 1: Bounding box ratio
    if width > height:
        lying_votes += 1

    # Condition 2: Torso angle
    if angle_deg > 30:
        lying_votes += 1

    # Final decision
    if lying_votes >= 1:
        scene = "Person NOT OK (Lying/Collapsed)"
    else:
        scene = "Person OK (Standing/Sitting)"

else:
    scene = "No Person Detected"

# Display result
cv2.putText(image, scene, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow('Scene Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
