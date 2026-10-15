# Evaluation protocol (proposed, not executed)

**Status:** Proposed protocol for a research demonstration. This document reports **no empirical results or validated safety performance**.

## Define the target
Describe what the model should identify (e.g. a person appearing to lie sideways in the image), rather than equating posture with a medical emergency. A voluntary resting posture can resemble a collapse, and a real emergency can occur while someone is upright.

The code contains **different heuristics**:
- `detection_with_images.py`: single-image MediaPipe posture flag using an OR threshold.
- `detection_with_live_mediapipe.py`: single-person MediaPipe approach using multiple cues, including an uncalibrated relative z signal.
- `detection_with_live_yolo.py`: multi-person YOLOv8-pose using 2D keypoints with a width/height and torso-angle AND rule.

They must be evaluated **separately**. A reported score for one is not a score for the others.

## Recommended study plan
1. Verify rights and consent for all images and clips, document recording conditions and ensure no unnecessary personal information is uploaded.
2. Define explicit ground-truth classes for no visible person, ordinary activity, an unusual/possibly horizontal posture, and insufficient pose evidence. Specify who annotates ambiguous examples.
3. Split by **person and recording session**, not randomly by adjacent video frame, to avoid subject/session leakage.
4. Include diverse viewpoints, occlusions, seated and reclining postures, wheelchair users where appropriately represented, multiple-person scenes, and challenging lighting. Report selection and exclusion criteria.
5. Run a frozen implementation and thresholds on an untouched held-out set. Keep raw annotation labels separate from computed predictions.
6. Count full confusion matrices and report per-class precision, recall and F1, plus the share of frames with insufficient pose evidence. For an alert system, also report false alerts per hour and incident-level delay, explicitly defining an incident.
7. Assess behavior across sessions; include illustrative failure cases and confidence intervals where the sample size permits. Do not claim regulatory or clinical compliance based on a small test set.

## Metrics checklist

| Item | Result |
| --- | --- |
| Evaluated model/version | Not measured |
| Dataset source, usage rights and sample count | Not established |
| Per-subject/session split | Not established |
| Confusion matrix | Not measured |
| Precision/recall/F1 | Not measured |
| False-alert rate / missed incidents | Not measured |
| Performance under low light/occlusion | Not measured |
| Privacy/compliance review | Not completed |

## Local smoke check

```bash
python -m unittest discover -s tests -v
python detection_with_images.py --help
```

Passing these checks confirms only the synthetic classifier tests and CLI argument availability. It does not demonstrate performance on real people or emergencies. The repository's example images should be reviewed for origin and redistribution permission before use in publications.
