import torch
from torch import nn
from torchvision import models
import cv2
from facenet_pytorch import MTCNN
from PIL import Image

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        backbone = models.resnext50_32x4d(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.lstm = nn.LSTM(2048, 2048, 1, batch_first=True)
        self.dp = nn.Dropout(0.4)
        self.linear = nn.Linear(2048, 2)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        f = self.feature_extractor(x).view(batch_size, seq_length, -1)
        out, _ = self.lstm(f)
        return self.linear(self.dp(out[:, -1, :]))

def get_inference(video_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepfakeDetector().to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    mtcnn = MTCNN(margin=14, keep_all=False, device=device).eval()
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < 20:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(Image.fromarray(frame))
        if face is not None:
            frames.append(face)
    cap.release()

    if len(frames) < 20: return None
    
    input_tensor = torch.stack(frames).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        res = torch.argmax(prob, dim=1).item()
        return {"prediction": "FAKE" if res == 1 else "REAL", "confidence": float(prob[0][res]*100)}
