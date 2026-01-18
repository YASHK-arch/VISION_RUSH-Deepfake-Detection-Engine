from rest_framework.views import APIView
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
