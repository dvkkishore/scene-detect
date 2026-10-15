"""Deterministic tests for the pure posture-classification helper."""
import unittest
import numpy as np
from pose_classifier import classify_scene


def pose(shoulder_left, shoulder_right, hip_left, hip_right):
    landmarks = np.zeros((17, 3), dtype=float)
    for index, point in zip((5, 6, 11, 12), (shoulder_left, shoulder_right, hip_left, hip_right)):
        landmarks[index] = [*point, 0.99]
    return landmarks


class PoseClassificationTests(unittest.TestCase):
    def test_no_person(self):
        self.assertEqual(classify_scene(np.zeros((17, 3))), "No Person")

    def test_missing_torso_is_unclassified(self):
        landmarks = pose((80, 80), (100, 80), (80, 140), (100, 140))
        landmarks[11, 2] = 0
        self.assertEqual(classify_scene(landmarks), "Insufficient Pose")

    def test_vertical_pose_is_not_flagged(self):
        self.assertEqual(classify_scene(pose((80, 40), (120, 40), (80, 150), (120, 150))), "Person OK")

    def test_sideways_pose_is_flagged(self):
        self.assertEqual(classify_scene(pose((40, 80), (40, 120), (180, 80), (180, 120))), "Person NOT OK")


if __name__ == "__main__":
    unittest.main()
