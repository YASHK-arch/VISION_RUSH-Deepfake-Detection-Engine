import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNeXt50_32X4D_Weights
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import os
import numpy as np

class DeepfakeDetector(nn.Module):
    """
    ResNext50 + LSTM Architecture for Deepfake Detection.
    ResNext handles spatial feature extraction per frame.
    LSTM captures temporal inconsistencies (e.g., flickering, unnatural movement).
    """
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        # Use modern weights parameter to ensure the latest ImageNet features
        backbone = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        # LSTM hidden state and input size match the 2048-d ResNext output
        self.lstm = nn.LSTM(input_size=2048, hidden_size=2048, num_layers=1, batch_first=True)
        self.dp = nn.Dropout(0.4)
        self.linear = nn.Linear(2048, 2)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        # Collapse batch and sequence to process frames through CNN
        x = x.view(batch_size * seq_length, c, h, w)
        f = self.feature_extractor(x).view(batch_size, seq_length, -1)
        
        # Sequence processing via LSTM
        out, _ = self.lstm(f)
        
        # Return logits from the final temporal state
        return self.linear(self.dp(out[:, -1, :]))

def get_inference(video_path, model_path, num_frames=40):
    """
    Extracts frames, performs face detection with denoising, and runs inference.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepfakeDetector().to(device)
    
    # Load custom weights if available
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    
    # Increase margin to include more facial context and reduce edge artifacts
    mtcnn = MTCNN(margin=20, keep_all=False, device=device).eval()
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return None
        
    # UNIFORM SAMPLING: Pick frames spread across the entire video
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # --- DENOISING FILTER ---
        # Bilateral filter removes compression noise while preserving edges
        # This helps reduce false positives in compressed datasets like FaceForensics++
        filtered_frame = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # Prepare for face extraction
        frame_rgb = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(Image.fromarray(frame_rgb))
        
        if face is not None:
            frames.append(face)
            
    cap.release()

    # Validation: Ensure the sequence is complete for the LSTM
    if len(frames) < num_frames:
        return None
    
    # Stack into tensor: (Batch=1, Sequence=40, Channels=3, H=224, W=224)
    input_tensor = torch.stack(frames).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        res = torch.argmax(prob, dim=1).item()
        
        return {
            "prediction": "FAKE" if res == 1 else "REAL", 
            "confidence": round(float(prob[0][res] * 100), 2)
        }