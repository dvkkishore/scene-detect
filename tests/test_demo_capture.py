"""Exercise replay/capture I/O only; these tests do not simulate pose inference."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from tools.capture_live_demo import capture

HAS_IO = all(importlib.util.find_spec(name) for name in ('cv2', 'pyttsx3'))


@unittest.skipUnless(HAS_IO, 'Capture I/O tests require OpenCV and pyttsx3')
class CaptureTests(unittest.TestCase):
    def test_replays_pixels_records_labels_and_mutes_speech(self):
        import cv2
        import numpy as np
        import pyttsx3
        original_capture, original_init = cv2.VideoCapture, pyttsx3.init
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, output, script = folder/'input.png', folder/'output.png', folder/'io.py'
            pixels = np.full((32, 32, 3), 127, dtype=np.uint8)
            cv2.imwrite(str(source), pixels)
            script.write_text('''import cv2, pyttsx3
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
assert ok
cv2.putText(frame, "I/O test", (0, 12), 0, 0.2, (0, 0, 0))
cv2.imshow("test", frame)
engine = pyttsx3.init()
engine.say("test speech")
engine.runAndWait()
cap.release()
''')
            result = capture(script, source, output)
            self.assertEqual(result['drawn_labels'], ['I/O test'])
            self.assertEqual(result['speech_requests_muted'], ['test speech'])
            self.assertEqual(cv2.imread(str(output)).shape, pixels.shape)
        self.assertIs(cv2.VideoCapture, original_capture)
        self.assertIs(pyttsx3.init, original_init)

    def test_no_display_fails_without_creating_an_output(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, output, script = folder/'input.png', folder/'output.png', folder/'io.py'
            cv2.imwrite(str(source), np.zeros((8, 8, 3), dtype=np.uint8))
            script.write_text('pass\n')
            with self.assertRaisesRegex(RuntimeError, 'no canvas'):
                capture(script, source, output)
            self.assertFalse(output.exists())

    def test_missing_input_fails(self):
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            capture(Path('unused.py'), Path('/missing-scene-input.png'), Path('/tmp/unused.png'))
