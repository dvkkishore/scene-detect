```
# 👁️‍🗨️ SCENE-DETECT: Smart Home Scene Understanding for the Visually Impaired

This project uses AI-based human pose detection to assist visually impaired individuals by recognizing real-time scenes at home and classifying them into three categories:

- 🧍‍♂️ `Person OK` – Someone is present and everything appears normal
- 🚫 `Person Not OK` – Someone is lying, unresponsive, or without orientation
- 🚪 `Nobody` – No person is present in the scene

The system works with images or video streams and optionally provides voice alerts.

---

## 📁 Project Structure

```

SCENE-DETECT/
│
├── person\_ok/              # Images showing people in safe, normal postures
├── person\_not\_ok/          # Images showing people lying down, unconscious, etc.
├── nobody/                 # Images with no people present
│
├── final.py                # Final version: Live video detection with pose + audio
├── detection\_with\_images.py # Version for testing image folders (demo/testing)
│
├── requirements.txt        # List of Python packages
└── README.md               # This file

````

---

## ⚙️ How to Run

### 1. 📦 Install dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv env
source env/bin/activate   # or env\Scripts\activate on Windows
````

Then install requirements:

```bash
pip install -r requirements.txt
```

---

### 2. 🖼️ Run on Static Images

To run on images from the folders:

```bash
python detection_with_images.py
```

This will:

* Load each image from the three folders
* Analyze pose
* Display decision on screen

---

### 3. 🎥 Run on Live Video

To start real-time camera-based detection:

```bash
python final.py
```

Features:

* Uses webcam or external camera feed
* Displays stick figure (pose skeleton)
* Speaks “Person Not OK” if detected

---

## 🗣️ Voice Alert

The final model announces “Person Not OK!” using text-to-speech when a dangerous pose is detected.

---

## ✅ 3D Privacy Feature

Only pose skeletons are used for final decision-making, ensuring that:

* No raw RGB images are stored
* The system respects privacy requirements

---

## 📌 Notes

* This project was developed as part of the Smart Scene Detect initiative
* Ideal for smart homes, elderly care, and fall detection systems

---

## 🤝 Acknowledgements

Special thanks to:

* MediaPipe for pose estimation
* pyttsx3 for offline speech synthesis
* MiDaS for experimental depth map trials

---

## 🧠 Future Work

* Integrate OpenPose for better 3D pose accuracy
* Support for night-time detection (IR or low-light models)
* Integrate with IoT-based alert systems

```