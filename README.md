# 🛡️ Real-Time Drone Intrusion Detection System

> A real-time aerial object detection project using custom-trained YOLOv11 models. Developed for surveillance and defense use cases, with both stock video and live webcam detection modules.

[![Ultralytics](https://img.shields.io/badge/YOLOv11m-Ultralytics-blue?logo=github)](https://github.com/ultralytics/ultralytics)
[![Colab GPU](https://img.shields.io/badge/Colab-Tesla%20T4-yellow?logo=googlecolab)](https://colab.research.google.com/)
[![Streamlit](https://img.shields.io/badge/Deployed%20on-Streamlit%20Cloud-red)](https://realtime-aerial-objects-detection-system.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📸 Project Demo

> 🔗 **Live App**: [Streamlit Web App](https://realtime-aerial-objects-detection-system.streamlit.app/)

Two Modules:

* 🎥 Stock Footage Detection (YOLOv11m)
* 🔍 Real-Time Live Detection (YOLOv11n)

---

## 🧠 Overview

This project was developed as part of an internship to create a real-time aerial surveillance system for **drone intrusion detection**. It features:

* Four custom YOLOv11 models: `1_best.pt`, `2_best.pt`, `1_nano.pt`, `2_nano.pt`
* Streamlit-based frontend with live and recorded video detection
* Metrics logging and model benchmarking

---

## 📆 Datasets Used

### Custom Dataset 1

* Train: 8378 images
* Validation: 1505 images
* Test: 65 images

### Custom Dataset 2

* Train: 10359 images
* Validation: 2922 images
* Test: 1470 images

---
<p align="center">
  <img src="demo_sample.jpg" alt="Sample Output" width="600">
</p>
## 🔹 Model Training

### ✅ 1\_best.pt (YOLOv11m)

* Trained on Colab T4 GPU (Custom Dataset 1)
* Params: 20M | Layers: 125 | GFLOPs: 67.6
* Training Time: ∼4.15 hours (40 epochs)
* mAP\@0.5: **0.943** | mAP\@0.5:0.95: **0.661**

### ✅ 2\_best.pt (YOLOv11m)

* Fine-tuned 1\_best.pt on Laptop GPU (Custom Dataset 2)
* Training Time: ∼8+ hours (80 epochs)
* mAP\@0.5: **0.9573** | mAP\@0.5:0.95: **0.6438**

### ✅ 1\_nano.pt (YOLOv11n)

* Trained on Laptop GPU (Custom Dataset 1)
* Layers: 100 | Params: 2.58M | GFLOPs: 6.3
* mAP\@0.5: **0.948** | mAP\@0.5:0.95: **0.678**

### ✅ 2\_nano.pt (YOLOv11n)

* Fine-tuned 1\_nano.pt on Laptop GPU (Custom Dataset 2)
* mAP\@0.5: **0.959** | mAP\@0.5:0.95: **0.661**

---

## 📊 Benchmark Results

### Test Set - Custom Dataset 1

| Model      | mAP\@0.5   | mAP\@0.5:0.95 |
| ---------- | ---------- | ------------- |
| 1\_best.pt | 0.8956     | 0.6850        |
| 2\_best.pt | **0.9824** | **0.7389**    |
| 1\_nano.pt | **0.9863** | 0.775         |
| 2\_nano.pt | 0.9699     | **0.8021**    |

### Test Set - Custom Dataset 2

| Model      | mAP\@0.5   | mAP\@0.5:0.95 |
| ---------- | ---------- | ------------- |
| 1\_best.pt | 0.8171     | 0.3999        |
| 2\_best.pt | **0.9573** | **0.6438**    |
| 1\_nano.pt | 0.7983     | 0.3903        |
| 2\_nano.pt | **0.9626** | **0.6635**    |

---

## 🏆 Final Recommendation

>
> * `2_best.pt` for **stock video detection**
> * `2_nano.pt` for **real-time webcam detection**

---

## 🚨 Deployment Platform

* Streamlit Cloud
* Live app: [https://realtime-aerial-objects-detection-system.streamlit.app/](https://realtime-aerial-objects-detection-system.streamlit.app/)

<p align="center">
  <img src="ui.png" alt="App UI Screenshot" width="600">
</p>

---

## 🚀 Inference Speed

| Device         | Inference Time          | FPS      |
| -------------- | ----------------------- | -------- |
| T4 GPU (Colab) | ∼11.2 ms/img (YOLOv11m) | ∼75 FPS  |
| T4 GPU (Colab) | ∼2.6 ms/img (YOLOv11n)  | ∼75 FPS  |
| CPU (Colab)    | ∼300 ms/img             | ∼1-3 FPS |



---

## ✅ Project Features

* Custom YOLOv11m and YOLOv11n training & validation
* Multi-stage fine-tuning
* Real-time object detection via webcam
* Stock footage detection pipeline
* Streamlit-based UI with image previews
* Metrics logging for evaluation

---

## 🎓 Skills & Tools Used

* Python, PyTorch, YOLOv11
* Roboflow for annotation/export
* Google Colab + Local GPU (RTX 3050)
* Streamlit for frontend
* OpenCV for frame extraction

---

## ✅ TODOs Before Final Briefing

* [ ] Minor debugging of Streamlit app
* [ ] Clean up and optimize live module
* [ ] Prepare documentation PDF/report

<p align="center">
  <img src="info.png" alt="Additional Info" width="600">
</p>



## 📁 Directory Structure

```my-app-local/
├── .git/                 # Git directory for version control history.
├── core/                 # Core application logic, kept separate from the UI.
│   ├── __init__.py       # Makes 'core' a Python package.
│   ├── extract.py        # Logic to extract frames from video files.
│   └── predict.py        # Logic for loading models and running drone detection.
├── models/               # Contains the trained YOLOv11 models.
│   ├── 2_best.pt         # The primary, more accurate model for video file detection.
│   └── nano/
│       └── 2_nano.pt     # A lightweight "nano" model for fast, real-time webcam detection.
├── notebook/             # Jupyter notebooks for model training, experimentation, and analysis.
├── app.py                # The main Streamlit application. This file runs the web UI.
├── README.md             # Project documentation, setup instructions, and usage guide.
├── requirements.txt      # Lists all Python dependencies needed to run the project.
├── demo_sample.jpg       # A sample image for testing or demonstration.
└── sample.jpg            # Another sample image.
```


---


## 🙋‍♂️ Author

**Abinash Pradhan**
🚀 Aspiring Machine Learning Engineer | CV & Defense AI Projects
📝 Reach me: [LinkedIn](#https://www.linkedin.com/in/abinash-pradhan-a42157297/) | [Twitter](#https://x.com/abinashp01) | [Website](#https://abinashpradhan01.github.io/)

---

## 📄 License

This project is licensed under the APACHE 2.0 License - see the [LICENSE](LICENSE) file for details.

---
