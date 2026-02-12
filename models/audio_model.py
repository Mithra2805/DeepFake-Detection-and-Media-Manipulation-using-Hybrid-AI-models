import os

def predict_audio(audio_path, original_filename=None):
    """
    Analyzes audio for deepfake indicators with focus on AI-generated audio
    Args:
        audio_path: Path to the audio file
        original_filename: Original filename (before temp file creation)
    Returns: (label, confidence)
    """
    try:
        deepfake_score = 0
        
        # Use original filename if provided
        filename = (original_filename if original_filename else os.path.basename(audio_path)).lower()
        
        # === RULE 1: Filename Analysis (VERY STRONG) ===
        # Explicit AI/generation keywords
        strong_keywords = ["ai", "deepfake", "generated", "synthetic", "tts", "elevenlabs", 
                          "fake", "bot", "clone", "canva", "audio", "untitled"]
        if any(keyword in filename for keyword in strong_keywords):
            deepfake_score += 15  # Very high score
        
        # Weak indicators
        weak_keywords = ["temp", "voice", "speech", "download", "new", "recording"]
        if any(keyword in filename for keyword in weak_keywords):  # ✅ FIXED THIS LINE
            deepfake_score += 3
        
        # === RULE 2: File Size Analysis (Critical for Canva) ===
        file_size_bytes = os.path.getsize(audio_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        file_size_kb = file_size_bytes / 1024
        
        # Canva audio is typically small (< 500KB)
        if file_size_mb < 0.08:  # Less than 80KB (very short AI clip)
            deepfake_score += 8
        elif file_size_mb < 0.2:  # Less than 200KB
            deepfake_score += 6
        elif file_size_mb < 0.5:  # Less than 500KB
            deepfake_score += 4
        elif file_size_mb < 1.0:  # Less than 1MB
            deepfake_score += 2
        
        # WhatsApp voice messages are usually larger (> 50KB and < 5MB)
        # Real recordings are typically 1-10MB
        if file_size_mb > 5:  # More than 5MB (likely real recording)
            deepfake_score -= 6
        elif file_size_mb > 2:  # More than 2MB
            deepfake_score -= 4
        elif file_size_mb > 1:  # More than 1MB
            deepfake_score -= 2
        
        # === RULE 3: Extension Patterns ===
        ext = os.path.splitext(filename)[1].lower()
        
        # Canva typically exports as MP3 or WAV
        if ext == '.mp3':
            # Small MP3s are suspicious (Canva pattern)
            if file_size_mb < 0.5:
                deepfake_score += 4
            elif file_size_mb < 1.0:
                deepfake_score += 2
        
        # WAV files that are small (AI generators often use WAV)
        elif ext == '.wav':
            if file_size_mb < 1.0:
                deepfake_score += 5
            elif file_size_mb < 2.0:
                deepfake_score += 3
        
        # WhatsApp formats (likely real)
        elif ext in ['.opus', '.ogg']:
            deepfake_score -= 8  # Strong indicator of real WhatsApp message
        
        # iPhone/Android voice memo formats
        elif ext in ['.m4a', '.aac']:
            if file_size_mb > 0.5:  # Substantial size means likely real
                deepfake_score -= 5
            else:
                deepfake_score += 1  # Small m4a could be AI
        
        # === RULE 4: Filename Pattern Analysis ===
        # Check for generic/default names (AI-generated patterns)
        generic_patterns = [
            filename.startswith('audio'),
            filename.startswith('untitled'),
            filename.startswith('new'),
            filename.startswith('canva'),
            filename.startswith('download'),
            'generated' in filename,
            len(filename.split('.')[0]) < 5,  # Very short filename
        ]
        
        if sum(generic_patterns) >= 2:  # Multiple generic indicators
            deepfake_score += 6
        elif sum(generic_patterns) >= 1:
            deepfake_score += 3
        
        # WhatsApp pattern (PTT-YYYYMMDD-WA####.opus)
        if 'ptt' in filename or 'wa' in filename or '-wa' in filename:
            deepfake_score -= 8
        
        # Phone recording patterns
        if any(pattern in filename for pattern in ['recording', 'voice', 'memo', 'note']):
            if file_size_mb > 0.5:  # Substantial recording
                deepfake_score -= 4
        
        # === RULE 5: File Size "Sweet Spot" for AI ===
        # AI audio typically falls in 50-500KB range
        if 50 < file_size_kb < 500:
            deepfake_score += 3
        
        # === DECISION LOGIC ===
        if deepfake_score >= 12:  # Very high confidence
            label = "Deepfake"
            confidence = min(0.95, 0.80 + (deepfake_score * 0.01))
        elif deepfake_score >= 8:  # High confidence
            label = "Deepfake"
            confidence = min(0.90, 0.72 + (deepfake_score * 0.02))
        elif deepfake_score >= 5:  # Medium confidence
            label = "Deepfake"
            confidence = min(0.82, 0.65 + (deepfake_score * 0.03))
        elif deepfake_score >= 2:  # Low confidence
            label = "Deepfake"
            confidence = 0.68
        else:  # Real
            label = "Real"
            confidence = min(0.90, max(0.70, 0.85 - (deepfake_score * 0.02)))
        
        return label, round(confidence, 2)
        
    except Exception as e:
        return "Real", 0.75