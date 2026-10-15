# SCENE-DETECT: Pose-based scene understanding

## Problem and motivation

A smart-space interface may benefit from summarizing whether people are visible and what posture they appear to have, without presenting a full-color camera view to the operator. A difficult part of that concept is distinguishing **pose** from **well-being**: a person lying down may be fine, and an emergency may have no distinct visible posture.

This collaborative prototype explores the first part of that problem, not clinical emergency recognition.

## System overview

```text
Camera or image -> pose estimation -> keypoints -> posture heuristics
                                                |
                                      skeleton-only rendering
                                                |
                                     label / optional voice alert
```

Three demonstration paths exist:
- `detection_with_images.py`: one chosen image processed through MediaPipe and rendered to a skeleton output.
- `detection_with_live_mediapipe.py`: live single-person MediaPipe landmarks, skeleton visualization and conditional audio alert.
- `detection_with_live_yolo.py`: multi-person YOLOv8-Pose keypoints and per-person posture labels.

## Engineering decisions and trade-offs

**Keypoint view instead of RGB display.** The interface draws a skeleton, reducing visual exposure in the display. However, raw camera frames are still processed in memory, and local test images are tracked. This alone does **not** establish privacy or legal compliance.

**Interpretable rules rather than a learned emergency classifier.** Width/height and torso orientation are transparent and simple to inspect. They are highly sensitive to viewpoint, occlusion and unusual but harmless postures.

**Different variants, different rules.** The single-image demo uses a more permissive OR-style posture flag; the YOLO live classifier uses two 2D cues jointly. YOLO keypoints do not contain calibrated depth. Results from the variants should not be treated as interchangeable.

## What is observable

The repository includes sample images, live processing scripts, a CLI for one-image skeleton output, and synthetic unit tests for the isolated YOLO classifier. Passing these tests validates specified logic, **not** accuracy against real incidents.

Run one selected example with:

```bash
python detection_with_images.py --image "Person not OK/3.jpg" --output demo_frames/skeleton.png
```

A working MediaPipe environment is necessary for real-image inference. `--show` is optional.

## Evaluation required before outcome claims

An empirical study would need consented and labeled footage, subject/session-disjoint splits, a confusion matrix for each variant, false-alert rates, sensitivity/recall, and measurement under poor lighting and occlusion. See [EXPERIMENTS.md](../EXPERIMENTS.md) for the protocol; the repository does not currently establish those metrics.

## Team and individual contribution

This is a **team project**, credited to **Farrukh Rasool, Ammad Majid, Sai Pavan Bolagani, Syed Annas Jawed, Soham Shripad Kharadkar and Venkata Kishore Dannana** (as recorded in the README). The repository does not by itself specify who individually developed each subsystem. Any personal portfolio case study should accurately identify the parts you owned using commit records or team confirmation.

**Primary sources:** [README](../README.md) · [YOLO classifier](../pose_classifier.py) · [image demo](../detection_with_images.py).
