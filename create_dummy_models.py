"""
Script to create dummy model weights for testing purposes
Run this ONCE before using the application
"""

import torch
import os
from models.image_model import ImageDeepfakeDetector
from models.audio_model import AudioDeepfakeDetector

# Create checkpoints directory
os.makedirs('models/checkpoints', exist_ok=True)

print("Creating dummy model weights...")

# Create image model with pretrained weights
print("1/2 Creating image model...")
img_model = ImageDeepfakeDetector(pretrained=True)
torch.save(img_model.state_dict(), 'models/checkpoints/image_model.pth')
print("✓ Image model saved to models/checkpoints/image_model.pth")

# Create audio model
print("2/2 Creating audio model...")
audio_model = AudioDeepfakeDetector()
torch.save(audio_model.state_dict(), 'models/checkpoints/audio_model.pth')
print("✓ Audio model saved to models/checkpoints/audio_model.pth")

print("\n✅ All dummy models created successfully!")
print("You can now run: streamlit run app.py")
#venv\Scripts\activate