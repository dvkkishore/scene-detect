"""Single-image pose-visualization demo (no clinical fall detection)."""
import argparse
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Render a skeleton-only image from an explicitly selected photo."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image file")
    parser.add_argument("--output", type=Path, help="Optional PNG/JPEG output path")
    parser.add_argument("--show", action="store_true", help="Open an OpenCV preview window")
    return parser.parse_args(argv)


def process_image(image_path, output_path=None, show=False):
    import math
    import cv2
    import mediapipe as mp
    import numpy as np

    # The image and the skeleton are handled locally. This is not a privacy audit.
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Unable to read image: {image_path}")

    pose_api = mp.solutions.pose
    with pose_api.Pose(static_image_mode=True, model_complexity=2) as pose:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        del frame, rgb

    canvas_size = 512
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    label = "No Person Detected"

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        xs = [point.x for point in landmarks]
        ys = [point.y for point in landmarks]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        scale = 0.8 * canvas_size / max(width, height, 1e-6)
        offset_x = (canvas_size - scale * (min(xs) + max(xs))) / 2
        offset_y = (canvas_size - scale * (min(ys) + max(ys))) / 2

        for start, end in pose_api.POSE_CONNECTIONS:
            a, b = landmarks[start], landmarks[end]
            if a.visibility <= 0.5 or b.visibility <= 0.5:
                continue
            p1 = (int(a.x * scale + offset_x), int(a.y * scale + offset_y))
            p2 = (int(b.x * scale + offset_x), int(b.y * scale + offset_y))
            cv2.line(canvas, p1, p2, (0, 255, 0), 2)

        for point in landmarks:
            if point.visibility > 0.5:
                position = (int(point.x * scale + offset_x), int(point.y * scale + offset_y))
                cv2.circle(canvas, position, 3, (0, 0, 255), -1)

        ls, rs = landmarks[pose_api.PoseLandmark.LEFT_SHOULDER], landmarks[pose_api.PoseLandmark.RIGHT_SHOULDER]
        lh, rh = landmarks[pose_api.PoseLandmark.LEFT_HIP], landmarks[pose_api.PoseLandmark.RIGHT_HIP]
        torso_dx = (lh.x + rh.x - ls.x - rs.x) / 2
        torso_dy = (lh.y + rh.y - ls.y - rs.y) / 2
        angle = abs(math.degrees(math.atan2(torso_dx, torso_dy)))
        if angle > 90:
            angle = 180 - angle

        # Preserve this legacy image demo's OR heuristic. It differs from the
        # multi-person YOLO demo's stricter two-cue posture rule.
        sideways = width > height or angle > 30
        label = "Person NOT OK (Posture Flag)" if sideways else "Person OK (Posture Flag Clear)"

    cv2.putText(canvas, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), canvas):
            raise OSError(f"Unable to save skeleton image: {output_path}")
    if show:
        cv2.imshow("SCENE-DETECT: Skeleton Preview", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return label


def main(argv=None):
    args = parse_args(argv)
    print(process_image(args.image, args.output, args.show))


if __name__ == "__main__":
    main()
