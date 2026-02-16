"""
Configuration file for deepfake detection models.
Contains hyperparameters, paths, and model settings.
"""

import torch
import os

class Config:
    """Central configuration for all models"""
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'checkpoints')
    IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, 'image_model.pth')
    AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, 'audio_model.pth')
    
    # Image model configuration
    IMAGE_SIZE = 224
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]
    IMAGE_BACKBONE = 'efficientnet-b4'  # or 'xception'
    
    # Video model configuration
    VIDEO_SAMPLE_RATE = 10  # Sample every Nth frame
    VIDEO_MAX_FRAMES = 30   # Maximum frames to analyze
    VIDEO_CONFIDENCE_THRESHOLD = 0.5
    
    # Audio model configuration
    AUDIO_SAMPLE_RATE = 22050
    AUDIO_DURATION = 5.0  # seconds
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    AUDIO_IMG_HEIGHT = 128
    AUDIO_IMG_WIDTH = 128
    
    # Fusion configuration
    FUSION_WEIGHTS = {
        'image': 0.4,
        'video': 0.35,
        'audio': 0.25
    }
    
    # Training configuration (for reference)
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    NUM_CLASSES = 2  # Real vs Deepfake