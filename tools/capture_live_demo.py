"""Replay a local image/video through an unchanged live script, without GUI/audio.

Only camera selection, display, key polling and speech are intercepted. Pose
inference, posture classification and drawing run in the original script.
Run from the repository root. This is a documentation capture tool, not a camera
or audio integration test. No keypoints or labels are supplied by this tool.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "mediapipe": "detection_with_live_mediapipe.py",
    "yolo": "detection_with_live_yolo.py",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(script, source, output):
    import cv2
    import pyttsx3

    source, output = Path(source).resolve(), Path(output)
    if not source.is_file():
        raise ValueError(f"Input file does not exist: {source}")
    real_capture, real_text = cv2.VideoCapture, cv2.putText
    streams, labels, speech, canvases = [], [], [], []

    def replay(_requested_source):
        stream = real_capture(str(source))
        streams.append(stream)
        if not stream.isOpened():
            raise ValueError(f"OpenCV cannot open input: {source}")
        return stream

    def text(*args, **kwargs):
        labels.append(args[1] if len(args) > 1 else kwargs["text"])
        return real_text(*args, **kwargs)

    def display(_title, canvas):
        canvases[:] = [canvas.copy()]

    class SilentSpeech:
        def setProperty(self, *_args):
            pass

        def say(self, words):
            speech.append(words)

        def runAndWait(self):
            pass

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "path", [str(ROOT), *sys.path]))
            for name, replacement in {
                "VideoCapture": replay, "imshow": display, "putText": text,
                "waitKey": lambda _delay: -1, "destroyAllWindows": lambda: None,
            }.items():
                stack.enter_context(patch.object(cv2, name, replacement))
            stack.enter_context(patch.object(pyttsx3, "init", SilentSpeech))
            runpy.run_path(str(script), run_name="__main__")
    finally:
        for stream in streams:
            stream.release()
    if not canvases:
        raise RuntimeError("The script displayed no canvas; no output was saved")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvases[-1]):
        raise OSError(f"Unable to save canvas: {output}")
    return {
        "script": Path(script).name,
        "script_sha256": sha256(script),
        "input": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else source.name,
        "input_sha256": sha256(source),
        "output": output.as_posix(),
        "output_sha256": sha256(output),
        "drawn_labels": labels,
        "speech_requests_muted": speech,
        "capture_mode": "local file replay; last displayed canvas; GUI and speech intercepted",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=SCRIPTS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, help="Optional JSON execution record")
    args = parser.parse_args()
    if Path.cwd() != ROOT:
        parser.error("Run this tool from the repository root")
    record = capture(ROOT / SCRIPTS[args.variant], args.input, args.output)
    record["command"] = shlex.join(["python", *sys.argv])
    if args.variant == "yolo":
        record["model_sha256"] = sha256(ROOT / "yolov8n-pose.pt")
    encoded = json.dumps(record, indent=2) + "\n"
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
