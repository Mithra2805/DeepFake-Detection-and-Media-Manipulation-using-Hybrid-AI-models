"""
Preprocessing utilities for images, videos, and audio.
Handles data transformation and feature extraction.
"""

import torch
import torchvision.transforms as transforms
import numpy as np
import librosa
import cv2
from PIL import Image

class ImagePreprocessor:
    """Preprocessing pipeline for images"""
    
    def __init__(self, img_size=224, mean=None, std=None):
        """
        Args:
            img_size: Target image size
            mean: Normalization mean
            std: Normalization std
        """
        self.img_size = img_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def preprocess(self, image_path):
        """
        Preprocess image for model input
        
        Args:
            image_path: Path to image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor (1, 3, H, W)
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension


class VideoPreprocessor:
    """Preprocessing pipeline for videos"""
    
    def __init__(self, img_size=224, sample_rate=10, max_frames=30):
        """
        Args:
            img_size: Target frame size
            sample_rate: Sample every Nth frame
            max_frames: Maximum number of frames to extract
        """
        self.img_size = img_size
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.image_preprocessor = ImagePreprocessor(img_size=img_size)
    
    def extract_frames(self, video_path):
        """
        Extract and preprocess frames from video
        
        Args:
            video_path: Path to video file
            
        Returns:
            list: List of preprocessed frame tensors
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at specified rate
            if frame_count % self.sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame_resized = cv2.resize(frame_rgb, (self.img_size, self.img_size))
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_resized)
                # Apply transformations
                frame_tensor = self.image_preprocessor.transform(pil_image)
                frames.append(frame_tensor)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        return frames


class AudioPreprocessor:
    """Preprocessing pipeline for audio"""
    
    def __init__(self, sr=22050, duration=5.0, n_mels=128, n_fft=2048, hop_length=512):
        """
        Args:
            sr: Sample rate
            duration: Audio duration to process
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def audio_to_melspectrogram(self, audio_path):
        """
        Convert audio to mel-spectrogram
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            torch.Tensor: Mel-spectrogram tensor (1, 1, H, W)
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
        
        # Pad if necessary
        target_length = int(self.sr * self.duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        # Resize to fixed dimensions
        mel_spec_resized = cv2.resize(mel_spec_norm, (128, 128))
        
        # Convert to tensor
        mel_tensor = torch.FloatTensor(mel_spec_resized).unsqueeze(0).unsqueeze(0)
        
        return mel_tensor