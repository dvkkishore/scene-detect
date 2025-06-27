
# SCENE-DETECT — Human Scene Detection for Smart Homes

**SCENE-DETECT** is a smart, privacy-conscious scene understanding system designed to assist blind and visually impaired individuals. It automatically detects what is happening in a room — specifically:

- Person is **present and OK**
- Person is **present and NOT OK** (e.g., collapsed or lying)
- **No person** in the scene

The system uses pose detection and intelligent rules to classify each situation and optionally provides a voice alert when something is wrong.

---

## 💡 Project Highlights

- ✅ **Privacy first**: no original images are stored or shown
- ✅ **Real-time** and **image-based** variants
- ✅ Works with **single or multiple persons**
- ✅ Powered by **MediaPipe** and **YOLOv8-Pose**
- ✅ Audio alerts for critical events

---

## 📂 Project Files

### `detection_with_images.py`
🔍 **Single Image Scene Detection**  
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
🎥 **Live Scene Detection (Single Person Only)**  
Uses a webcam or phone camera feed and processes each frame using MediaPipe Pose.

- Only detects **one person** at a time
- Very accurate pose quality (33 points, including z-depth)
- Draws stick-figure in real time on a clean canvas
- Announces via voice: **"Person not okay!"**
- Discards the original image after use (privacy maintained)

📁 Best for: simple demos, high-accuracy single-person use

---

### `detection_with_live_yolo.py`
👥 **Live Scene Detection (Multi-Person)**  
Uses YOLOv8-Pose for real-time detection of **multiple people**.

- Draws each person as a stick figure in **true position**
- Classifies each person individually: **OK** / **NOT OK**
- Places labels near each person
- Voice alert if **anyone** is "NOT OK"
- All raw image data is deleted after use to preserve privacy

📁 Best for: advanced demos, multi-person environments, smart home simulation

---

## 🔊 Voice Output
All live scripts support audio alerts via `pyttsx3`. The system says:  
**“Alert. Person not okay.”**  
...when a collapsed person is detected.

---

## 🔧 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

You may also need:
```bash
sudo apt install libespeak1 ffmpeg
```

Basic requirements:
- Python 3.8+
- OpenCV
- MediaPipe
- Ultralytics (for YOLOv8)
- pyttsx3

---

## ▶️ How to Run

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

## 🛡️ Privacy Compliance

SCENE-DETECT ensures:
- Original RGB frames are deleted immediately after pose extraction
- Output is shown as skeleton-only canvas
- Designed with assistive and ethical AI principles in mind

---

## 📌 Use Cases

- 🏠 Smart home monitoring for elderly or visually impaired
- 🧪 Human activity classification demos
- 👩‍🏫 Teaching pose estimation + privacy-centered design
- 📊 Dataset analysis and model evaluation

---

## 👤 Authors

- **Your Name**
- Smart Scene Detect Project, [Your University Name]
- Course: Medical Systems Engineering / AI for Industrial Applications

---

## 📄 License

MIT License — feel free to use, modify, and improve with attribution.
