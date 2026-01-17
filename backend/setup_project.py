import os

# Define the folder structure
folders = [
    'backend/core',
    'backend/api',
    'backend/media',
    'backend/weights'
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Define file contents
files = {
    'backend/requirements.txt': """django
django-rest-framework
django-cors-headers
torch
torchvision
opencv-python
facenet-pytorch
Pillow""",

    'backend/api/model_utils.py': """import torch
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
""",

    'backend/api/views.py': """from rest_framework.views import APIView
from rest_framework.response import Response
from .model_utils import get_inference
import os
from django.conf import settings

class UploadView(APIView):
    def post(self, request):
        file = request.FILES.get('video')
        path = os.path.join(settings.MEDIA_ROOT, file.name)
        with open(path, 'wb+') as f:
            for chunk in file.chunks(): f.write(chunk)
        return Response({"video_id": file.name})

class PredictView(APIView):
    def post(self, request):
        video_id = request.data.get('video_id')
        video_path = os.path.join(settings.MEDIA_ROOT, video_id)
        result = get_inference(video_path, 'weights/model.pt')
        if not result: return Response({"error": "No faces detected"}, status=400)
        result['details'] = {"framesAnalyzed": 20, "modelUsed": "ResNext+LSTM", "device": "cpu"}
        return Response(result)
""",
    'backend/api/urls.py': """from django.urls import path
from .views import UploadView, PredictView

urlpatterns = [
    path('upload/', UploadView.as_view()),
    path('predict/', PredictView.as_view()),
]"""
}

# Write files
for path, content in files.items():
    with open(path, 'w') as f:
        f.write(content)

print("Project Structure Created Successfully!")