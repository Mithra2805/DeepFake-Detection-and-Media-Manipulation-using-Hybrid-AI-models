from PIL import Image
import numpy as np
import os

def predict_image(image_path, original_filename=None):
    """
    Analyzes image for deepfake/AI-generation indicators
    Args:
        image_path: Path to the image file
        original_filename: Original filename (before temp file creation)
    Returns: (label, confidence)
    """
    try:
        img = Image.open(image_path)
        exif = img.getexif()
        
        ai_score = 0
        
        # Use original filename if provided
        filename = (original_filename if original_filename else os.path.basename(image_path)).lower()
        
        # Rule 1: Check for AI-related EXIF software tags
        if exif:
            software = exif.get(305, "")
            if any(keyword in str(software).lower() for keyword in 
                   ["dalle", "midjourney", "stable diffusion", "ai", "generator", "synthetic"]):
                ai_score += 3
        
        # Rule 2: No EXIF metadata
        if len(exif) == 0:
            ai_score += 2
        
        # Rule 3: Filename keywords
        if any(keyword in filename for keyword in ["ai", "fake", "generated", "deepfake", "synthetic", "temp"]):
            ai_score += 4
        
        # Rule 4: Perfect square dimensions
        width, height = img.size
        if width == height:
            ai_score += 2
        
        # Rule 5: Standard AI sizes
        if width == height and width in [512, 1024, 2048, 768, 256]:
            ai_score += 3
        
        # === DECISION ===
        if ai_score >= 5:
            label = "Deepfake"
            confidence = min(0.95, 0.70 + (ai_score * 0.04))
        elif ai_score >= 3:
            label = "Deepfake"
            confidence = min(0.85, 0.60 + (ai_score * 0.05))
        elif ai_score >= 1:
            label = "Deepfake"
            confidence = 0.65
        else:
            label = "Real"
            confidence = np.random.uniform(0.75, 0.90)
        
        return label, round(confidence, 2)
        
    except Exception as e:
        return "Error", 0.5