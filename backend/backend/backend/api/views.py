from rest_framework.views import APIView
from rest_framework.response import Response
from .model_utils import get_inference
import os
from django.conf import settings

class UploadView(APIView):
    """
    Handles the initial video upload from the frontend.
    Saves the file to the media directory for later processing.
    """
    def post(self, request):
        file = request.FILES.get('video')
        
        if not file:
            return Response({"error": "No video file provided"}, status=400)
            
        # Ensure the media directory exists
        if not os.path.exists(settings.MEDIA_ROOT):
            os.makedirs(settings.MEDIA_ROOT)
            
        path = os.path.join(settings.MEDIA_ROOT, file.name)
        
        # Save the uploaded file in chunks to handle larger videos efficiently
        with open(path, 'wb+') as f:
            for chunk in file.chunks():
                f.write(chunk)
                
        return Response({"video_id": file.name})

class PredictView(APIView):
    """
    Triggers the AI detection logic.
    Passes the video through MTCNN face extraction and the ResNext+LSTM model.
    """
    def post(self, request):
        video_id = request.data.get('video_id')
        
        if not video_id:
            return Response({"error": "No video_id provided"}, status=400)
            
        video_path = os.path.join(settings.MEDIA_ROOT, video_id)
        
        # This matches the increased sampling rate in your model_utils.py
        frames_to_scan = 40 
        
        # Run the updated inference engine
        # result contains: {"prediction": "FAKE/REAL", "confidence": float}
        result = get_inference(video_path, 'weights/model.pt', num_frames=frames_to_scan)
        
        if not result:
            return Response({
                "error": f"Face detection failed across the video sequence. Try a video with better lighting."
            }, status=400)
            
        # Return the structured JSON exactly as frontend.html expects
        return Response({
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "details": {
                "framesAnalyzed": frames_to_scan, # This updates the UI value
                "modelUsed": "ResNext+LSTM",
                "device": "cpu"
            }
        })