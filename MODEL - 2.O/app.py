import streamlit as st
import tempfile
import os
import json
import torch
import timm
from PIL import Image
import cv2
import numpy as np
from statistics import median
import torchvision.transforms as transforms

# ----------------------------
# CONFIG
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "df_detector_mvp.pth"
IMG_SIZE = 518  # since you're using 518 model
FPS_SAMPLE = 2

# ----------------------------
# Load model (cached)
# ----------------------------
@st.cache_resource
def load_model():
    model = timm.create_model(
        "vit_base_patch14_dinov2",
        pretrained=False,
        num_classes=1
    )
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    return model

model = load_model()

# ----------------------------
# Preprocess
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ----------------------------
# Simple face crop (fallback = full frame)
# ----------------------------
def detect_face(frame):
    # For MVP: just return full frame (you can plug your face detector later)
    h, w = frame.shape[:2]
    return frame, (0,0,w,h)

# ----------------------------
# Video inference
# ----------------------------
def compute_step(orig_fps, target_fps):
    return max(1, int(round((orig_fps if orig_fps > 0 else 30) / target_fps)))

def run_video_detection(video_path, threshold=0.65, smooth_window=3, min_seg_dur=0.6, merge_gap=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = compute_step(orig_fps, FPS_SAMPLE)

    times = []
    probs = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            crop, _ = detect_face(frame)
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            inp = transform(pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logit = model(inp)
                if logit.dim() == 2:
                    logit = logit[:,0]
                prob = torch.sigmoid(logit)[0].item()

            t = frame_idx / orig_fps
            times.append(t)
            probs.append(prob)

        frame_idx += 1

    cap.release()

    if len(probs) == 0:
        return {"input_type":"video","video_is_fake":False,"overall_confidence":0.0,"manipulated_segments":[]}

    # ----------------------------
    # Smoothing (median)
    # ----------------------------
    smoothed = []
    n = len(probs)
    w = smooth_window

    for i in range(n):
        l = max(0, i - w//2)
        r = min(n, i + w//2 + 1)
        smoothed.append(float(median(probs[l:r])))

    # ----------------------------
    # Threshold & segments
    # ----------------------------
    mask = [p >= threshold for p in smoothed]

    segments = []
    start = None

    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            s = times[start]
            e = times[i-1]
            if e - s >= min_seg_dur:
                segments.append((s, e, max(smoothed[start:i])))
            start = None

    if start is not None:
        s = times[start]
        e = times[-1]
        if e - s >= min_seg_dur:
            segments.append((s, e, max(smoothed[start:])))

    # Merge close segments
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            prev = merged[-1]
            if seg[0] - prev[1] <= merge_gap:
                merged[-1] = (prev[0], seg[1], max(prev[2], seg[2]))
            else:
                merged.append(seg)

    # Format output
    def sec_to_hms(s):
        s = int(round(s))
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"

    out_segments = []
    for s,e,c in merged:
        out_segments.append({
            "start_time": sec_to_hms(s),
            "end_time": sec_to_hms(e),
            "confidence": round(c,3)
        })

    overall = float(max(smoothed))
    return {
        "input_type": "video",
        "video_is_fake": len(out_segments) > 0,
        "overall_confidence": round(overall,3),
        "manipulated_segments": out_segments
    }

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Deepfake Detector MVP", layout="centered")
st.title("🎭 Deepfake Video Detector (MVP)")

st.write("Upload a video. The system will analyze it at 2 FPS and return timestamps of suspected fake segments.")

uploaded = st.file_uploader("Upload a video", type=["mp4","mov","avi"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing video..."):
            result = run_video_detection(video_path)

        st.subheader("📊 Result")

        if result["video_is_fake"]:
            st.error(f"🚨 FAKE detected! Confidence: {result['overall_confidence']}")
        else:
            st.success(f"✅ Video looks REAL. Confidence: {result['overall_confidence']}")

        st.subheader("🕒 Manipulated Segments")

        if len(result["manipulated_segments"]) == 0:
            st.write("No manipulated segments detected.")
        else:
            st.table(result["manipulated_segments"])

        st.subheader("🧾 Raw JSON")
        st.json(result)

    # cleanup optional
    # os.remove(video_path)