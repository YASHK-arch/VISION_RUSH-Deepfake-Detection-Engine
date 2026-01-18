# 🎭 Deepfake Video Detection MVP (Self-Trained ViT)

This repository contains a **self-trained deepfake video detection system** built using a **Vision Transformer (ViT-B/14 with DINOv2 backbone)**.  
The system analyzes videos temporally, identifies manipulated segments, and outputs **timestamp-localized deepfake regions**, with an emphasis on **localization quality over raw classification accuracy**.

---

## 🚀 Key Features

- ✅ **Self-trained deepfake model** (not a prebuilt classifier)
- 🎯 **Vision Transformer (ViT-B/14, DINOv2)**
- ⏱️ **Timestamp localization of manipulated segments**
- 📊 **Median smoothing + temporal segment merging**
- 🎞️ **Video-level and segment-level confidence scores**
- 🖥️ **Interactive Streamlit web interface**
- ⚡ Efficient inference via **2 FPS frame sampling**

---

## 🧠 Model Overview

| Component | Description |
|---------|-------------|
| Backbone | `vit_base_patch14_dinov2` |
| Framework | PyTorch + TIMM |
| Input Resolution | 518 × 518 |
| Output | Binary classification (Real / Fake) |
| Weights | Self-trained (`df_detector_mvp.pth`) |

> Each frame produces a **single logit**, converted to a probability using a sigmoid function.

---

## 📂 Project Structure

.
├── app.py # Streamlit application
├── df_detector_mvp.pth # Self-trained model weights
├── README.md # Documentation
└── requirements.txt # Python dependencies

yaml
Copy code

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone <your-repo-url>
cd deepfake-video-detector
2️⃣ Create a Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
Required Packages
torch

timm

torchvision

opencv-python

streamlit

Pillow

numpy

▶️ Running the Application
bash
Copy code
streamlit run app.py
Then open your browser at:

arduino
Copy code
http://localhost:8501
🎥 Video Detection Pipeline
🔹 Frame Sampling
Videos are sampled at 2 frames per second

Sampling step is automatically adjusted based on original FPS

🔹 Frame Processing
Extract frame

(MVP) Use full frame as face crop

Resize to 518 × 518

Normalize using ImageNet statistics

Run ViT inference → fake probability

🧪 Temporal Post-Processing
1️⃣ Probability Smoothing
Median filtering across neighboring frames:

text
Copy code
smooth_window = 3
2️⃣ Thresholding
text
Copy code
Frame is fake if probability ≥ 0.65
3️⃣ Segment Formation
Consecutive fake frames are grouped

Minimum segment duration: 0.6 seconds

4️⃣ Segment Merging
Segments closer than 0.5 seconds are merged

📤 Output Format (JSON)
json
Copy code
{
  "input_type": "video",
  "video_is_fake": true,
  "overall_confidence": 0.93,
  "manipulated_segments": [
    {
      "start_time": "00:00:42",
      "end_time": "00:00:50",
      "confidence": 0.88
    }
  ]
}
Output Field Description
Field	Meaning
video_is_fake	Whether any manipulated segment is detected
overall_confidence	Maximum smoothed frame probability
manipulated_segments	Localized fake time intervals

🖥️ Streamlit UI Features
📤 Video upload

🎞️ Video preview

🔍 One-click analysis

🚨 Fake / Real verdict display

🕒 Timestamped manipulation table

🧾 Raw JSON output

⚠️ Current Limitations (MVP)
❌ No explicit face detector (full frame used)

❌ No audio-based deepfake detection

❌ No frame-level spatial heatmaps

❌ Binary classification only

🔮 Planned Improvements
✅ Face detection (RetinaFace / YOLOv8-Face)

✅ Multiple Instance Learning (MIL)

✅ Frame-level manipulation heatmaps

✅ Audio-visual fusion

✅ Temporal transformer modeling

✅ Explainability via attention visualization

🏗️ Why Vision Transformer + Temporal Logic?
ViT + Temporal Processing	CNN / CNN+LSTM
Global context awareness	Texture-biased
Fewer sampled frames	Dense sampling needed
Better generalization	Overfits artifacts
Cleaner segment localization	Noisy predictions

🧑‍💻 Author
Yash Kumar
Deepfake Detection · Computer Vision · AI Systems

Built as an engineering MVP with a strong focus on timestamp localization, not leaderboard accuracy.