"""
Hybrid Video Deepfake Detection
"""

import torch
import numpy as np
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import Config
from utils.preprocessing import VideoPreprocessor
from models.image_model import get_image_model


class VideoModelInference:
    """Video deepfake detection with hybrid approach"""
    
    def __init__(self, sample_rate=None, max_frames=None, device=None):
        self.device = device or Config.DEVICE
        self.sample_rate = sample_rate or Config.VIDEO_SAMPLE_RATE
        self.max_frames = max_frames or Config.VIDEO_MAX_FRAMES
        
        self.image_model = get_image_model()
        
        self.video_preprocessor = VideoPreprocessor(
            img_size=Config.IMAGE_SIZE,
            sample_rate=self.sample_rate,
            max_frames=self.max_frames
        )
    
    def _heuristic_analysis(self, video_path):
        """Fallback heuristic for untrained models"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return "Error", 0.5
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            score = 0
            
            # Filename check
            filename = os.path.basename(video_path).lower()
            if any(kw in filename for kw in ['ai', 'fake', 'generated', 'deepfake', 'synthetic']):
                score += 5
            
            # Duration check
            duration = total_frames / fps if fps > 0 else 0
            if 0 < duration < 5:
                score += 3
            elif 0 < duration < 10:
                score += 2
            
            # Dimension check
            if width == height:
                score += 4
            
            if (width, height) in [(512, 512), (1024, 1024), (768, 768)]:
                score += 4
            
            # Frame count
            if total_frames < 60:
                score += 3
            elif total_frames < 150:
                score += 1
            
            # Decision
            if score >= 7:
                return "Deepfake", min(0.92, 0.70 + score * 0.03)
            elif score >= 4:
                return "Deepfake", 0.75
            elif score >= 2:
                return "Deepfake", 0.68
            else:
                return "Real", 0.80
                
        except Exception as e:
            return "Real", 0.75
    
    def predict(self, video_path, original_filename=None):
        """Predict if video contains deepfake"""
        try:
            # Check if underlying image model is trained
            if not self.image_model.is_trained:
                print("⚠ Using heuristic analysis for video (model not trained)")
                return self._heuristic_analysis(video_path)
            
            # Use frame-level analysis
            frames = self.video_preprocessor.extract_frames(video_path)
            
            if not frames:
                return "Error", 0.5
            
            frame_predictions = []
            deepfake_confidences = []
            real_confidences = []
            
            for frame_tensor in frames:
                frame_batch = frame_tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.image_model.model(frame_batch)
                    probabilities = torch.softmax(outputs, dim=1)
                    
                    real_prob = probabilities[0][0].item()
                    deepfake_prob = probabilities[0][1].item()
                    
                    real_confidences.append(real_prob)
                    deepfake_confidences.append(deepfake_prob)
                    
                    predicted_label = "Real" if real_prob > deepfake_prob else "Deepfake"
                    frame_predictions.append(predicted_label)
            
            avg_deepfake_conf = np.mean(deepfake_confidences)
            avg_real_conf = np.mean(real_confidences)
            
            deepfake_votes = sum(1 for pred in frame_predictions if pred == "Deepfake")
            total_frames = len(frame_predictions)
            deepfake_ratio = deepfake_votes / total_frames
            
            max_deepfake_conf = max(deepfake_confidences)
            max_real_conf = max(real_confidences)
            
            if deepfake_ratio > 0.5 or avg_deepfake_conf > avg_real_conf:
                final_label = "Deepfake"
                final_confidence = 0.7 * avg_deepfake_conf + 0.3 * max_deepfake_conf
            else:
                final_label = "Real"
                final_confidence = 0.7 * avg_real_conf + 0.3 * max_real_conf
            
            consistency_score = max(deepfake_ratio, 1 - deepfake_ratio)
            final_confidence *= (0.5 + 0.5 * consistency_score)
            
            return final_label, round(final_confidence, 2)
            
        except Exception as e:
            print(f"Error in video prediction: {e}")
            return self._heuristic_analysis(video_path)


_video_model_instance = None

def get_video_model():
    global _video_model_instance
    if _video_model_instance is None:
        _video_model_instance = VideoModelInference()
    return _video_model_instance


def predict_video(video_path, original_filename=None):
    model = get_video_model()
    return model.predict(video_path, original_filename)