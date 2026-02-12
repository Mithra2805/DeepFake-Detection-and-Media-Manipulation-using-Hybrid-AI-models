import numpy as np

def hybrid_decision(results):
    """
    Fuses decisions from multiple modalities
    results: list of tuples [(label, confidence), ...]
    """
    if not results:
        return "Unknown", 0.0
    
    deepfake_votes = 0
    real_votes = 0
    total_confidence = 0
    
    for label, conf in results:
        total_confidence += conf
        if label == "Deepfake":
            deepfake_votes += conf  # Weight by confidence
        else:
            real_votes += conf
    
    # Majority voting weighted by confidence
    if deepfake_votes > real_votes:
        final_label = "Deepfake"
        final_conf = deepfake_votes / (deepfake_votes + real_votes)
    else:
        final_label = "Real"
        final_conf = real_votes / (deepfake_votes + real_votes)
    
    return final_label, round(final_conf, 2)