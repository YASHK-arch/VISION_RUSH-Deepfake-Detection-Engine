
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
