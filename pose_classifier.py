"""Interpretable 2D posture heuristic for YOLOv8-pose keypoints.

The coordinates are (x, y, confidence). There is no depth coordinate.
Do not use this prototype's output as a medical/safety decision.
"""
import math
import numpy as np


def classify_scene(keypoints):
    """Return an illustrative posture label from one person's keypoints.

    A sideways torso and wide visible bounds jointly flag possible lying.
    This is NOT validated fall detection and cannot infer injury or distress.
    """
    points = np.asarray(keypoints)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected shape (N, 3) containing x, y, confidence")
    visible = (points[:, 2] > 0.5) & np.isfinite(points).all(axis=1)
    if not np.any(visible):
        return "No Person"
    if points.shape[0] < 13 or np.count_nonzero(visible) < 4:
        return "Insufficient Pose"

    # Shoulder/hip positions are required to estimate torso orientation.
    shoulders_and_hips = (5, 6, 11, 12)
    if not all(visible[i] for i in shoulders_and_hips):
        return "Insufficient Pose"
    xs, ys = points[visible, 0], points[visible, 1]
    width, height = float(np.ptp(xs)), float(np.ptp(ys))
    if width <= 0 and height <= 0:
        return "Insufficient Pose"

    left_shoulder, right_shoulder, left_hip, right_hip = points[list(shoulders_and_hips)]
    torso_x = (left_hip[0] + right_hip[0] - left_shoulder[0] - right_shoulder[0]) / 2
    torso_y = (left_hip[1] + right_hip[1] - left_shoulder[1] - right_shoulder[1]) / 2
    angle = abs(math.degrees(math.atan2(torso_x, torso_y)))
    if angle > 90:
        angle = 180 - angle

    # Two independent 2D cues. Do not fabricate "Z" from the image Y axis.
    possible_lying = width > height and angle > 30
    return "Person NOT OK" if possible_lying else "Person OK"
