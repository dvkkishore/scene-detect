"""Dependency-free CLI behavior checks for the single-image demonstration."""
import subprocess
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ImageCliTests(unittest.TestCase):
    def test_cli_help_does_not_require_a_camera_or_ml_dependencies(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "detection_with_images.py"), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--image", result.stdout)
        self.assertIn("--output", result.stdout)

    def test_image_is_required(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "detection_with_images.py")],
            capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--image", result.stderr)


if __name__ == "__main__":
    unittest.main()
