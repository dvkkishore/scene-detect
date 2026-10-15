
# SCENE-DETECT — Human Scene Detection for Smart Homes

**SCENE-DETECT** is a smart, privacy-conscious scene understanding system designed to assist blind and visually impaired individuals. It automatically detects what is happening in a room — specifically:

- Person is **present and OK**
- Person is **present and NOT OK** (e.g., collapsed or lying)
- **No person** in the scene

The system uses pose detection and intelligent rules to classify each situation and optionally provides a voice alert when something is wrong.

---

## Project Highlights

- **Skeleton-only live display**: the live viewer renders keypoints, not full-color camera frames; camera images are still processed in memory and sample photos are in this repository
- **Real-time** and **image-based** variants
- Works with **single or multiple persons**
- Powered by **MediaPipe** and **YOLOv8-Pose**
- Audio alerts for critical events

---

## Project Files

### `detection_with_images.py`
**Single Image Scene Detection**  
This script processes individual images from three folders:
- `/Nobody/`
- `/Person OK/`
- `/Person NOT OK/`

For each image:
- Detects pose using MediaPipe
- Classifies the scene
- Draws a clean stick-figure skeleton on a 512×512 canvas
- Labels the result and saves it

📁 Useful for: testing, dataset analysis, offline evaluation

---

### `detection_with_live_mediapipe.py`
**Live Scene Detection (Single Person Only)**  
Uses a webcam or phone camera feed and processes each frame using MediaPipe Pose.

- Only detects **one person** at a time
- MediaPipe estimates 33 landmarks including relative z values (not a calibrated physical depth measurement)
- Draws stick-figure in real time on a clean canvas
- Announces via voice: **"Person not okay!"**
- Drops local frame references after inference; this is not a formal privacy guarantee

📁 Best for: simple demos, high-accuracy single-person use

---

### `detection_with_live_yolo.py`
**Live Scene Detection (Multi-Person)**  
Uses YOLOv8-Pose for real-time detection of **multiple people**.

- Draws each person as a stick figure in **true position**
- Classifies each person individually: **OK** / **NOT OK**
- Places labels near each person
- Voice alert if **anyone** is classified "NOT OK" by the posture heuristic
- Attempts to drop raw frame references after inference; library-level retention is not verified

📁 Best for: advanced demos, multi-person environments, smart home simulation

---

## Voice Output
All live scripts support audio alerts via `pyttsx3`. The system says:  
**“Alert. Person not okay.”**  
...when a collapsed person is detected.

---

## Requirements

Install dependencies:

**For Linux**
```bash
pip install -r requirements_linux.txt
```

**For Windows**
```bash
pip install -r requirements_win.txt
```

Basic requirements:
- Python 3.8+
- OpenCV
- MediaPipe
- Ultralytics (for YOLOv8)
- pyttsx3

---

## How to Run

**Image detection:**
```bash
python detection_with_images.py
```

**Live detection (MediaPipe):**
```bash
python detection_with_live_mediapipe.py
```

**Live detection (YOLOv8 multi-person):**
```bash
python detection_with_live_yolo.py
```

---

## Safety and Privacy Limitations

This is a **research/demo posture heuristic**, not medical-grade fall detection. A person lying voluntarily may be flagged, and a genuine fall may be missed. Do not rely on these labels for emergencies or unsupervised elderly-care monitoring. The YOLOv8-pose stream returns 2D coordinates and confidence; the implementation must not treat image Y position as a genuine depth measurement.

The live interfaces show skeleton-only renderings, but original RGB frames are processed by MediaPipe/Ultralytics in memory. Deleting Python references does not prove that all internal copies are immediately erased. The repository also contains test photos. No regulatory/privacy compliance has been established by this prototype.

**Camera selection (YOLO):** the default source is camera index 0. Set `SCENE_DETECT_CAMERA` to another camera index or stream URL in the environment if required. Do not commit a credential-bearing camera URL.

**Classifier regression tests:**

```bash
python -m unittest discover -s tests -v
```

The lightweight classifier tests require NumPy and do not require a camera or pre-trained model. Real-world sensitivity and false-alert rates remain unevaluated.

---

## Use Cases

- 🏠 Smart home monitoring for elderly or visually impaired
- 🧪 Human activity classification demos
- 👩‍🏫 Teaching pose estimation + privacy-centered design
- 📊 Dataset analysis and model evaluation

---

## Authors

- **Farrukh Rasool, Ammad Majid, Sai Pavan Bolagani, Syed Annas Jawed, Soham Shripad Kharadkar, Venkata Kishore Dannana**
- Smart Scene Detect Project, OTH Amberg-Weiden
- Course: Industry 4.0 Project/ Artificial Intelligence for Industrial Applications

---