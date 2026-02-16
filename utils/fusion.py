"""
Multi-modal fusion module for combining predictions from different modalities.
Implements weighted confidence fusion with dynamic weight adjustment.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional

class ModalityFusion:
    """
    Fusion module for combining predictions from multiple modalities
    using weighted confidence aggregation.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize fusion module with modality weights
        
        Args:
            weights: Dictionary of modality weights {'image': w1, 'video': w2, 'audio': w3}
                    Weights should sum to 1.0
        """
        self.weights = weights or {
            'image': 0.4,
            'video': 0.35,
            'audio': 0.25
        }
        
        # Validate weights
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0):
            # Normalize weights
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update fusion weights
        
        Args:
            new_weights: New weight dictionary
        """
        self.weights = new_weights
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0):
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def weighted_fusion(self, predictions: List[Tuple[str, float, str]]) -> Tuple[str, float]:
        """
        Perform weighted fusion of multi-modal predictions
        
        Args:
            predictions: List of tuples (label, confidence, modality)
                        Example: [("Deepfake", 0.85, "image"), ("Real", 0.60, "video")]
        
        Returns:
            Tuple of (final_label, final_confidence)
        """
        if not predictions:
            return "Unknown", 0.0
        
        # Aggregate weighted probabilities
        deepfake_score = 0.0
        real_score = 0.0
        total_weight = 0.0
        
        for label, confidence, modality in predictions:
            weight = self.weights.get(modality, 0.0)
            
            if label == "Deepfake":
                deepfake_score += confidence * weight
                real_score += (1.0 - confidence) * weight
            elif label == "Real":
                real_score += confidence * weight
                deepfake_score += (1.0 - confidence) * weight
            
            total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            deepfake_score /= total_weight
            real_score /= total_weight
        
        # Determine final prediction
        if deepfake_score > real_score:
            final_label = "Deepfake"
            final_confidence = deepfake_score
        else:
            final_label = "Real"
            final_confidence = real_score
        
        return final_label, round(final_confidence, 2)
    
    def confidence_based_fusion(self, predictions: List[Tuple[str, float, str]]) -> Tuple[str, float]:
        """
        Confidence-based fusion that dynamically adjusts weights based on prediction confidence
        
        Args:
            predictions: List of tuples (label, confidence, modality)
        
        Returns:
            Tuple of (final_label, final_confidence)
        """
        if not predictions:
            return "Unknown", 0.0
        
        # Calculate dynamic weights based on confidence
        dynamic_weights = {}
        total_confidence = sum(conf for _, conf, _ in predictions)
        
        for label, confidence, modality in predictions:
            base_weight = self.weights.get(modality, 0.0)
            # Boost weight for high-confidence predictions
            dynamic_weights[modality] = base_weight * (confidence / total_confidence) * len(predictions)
        
        # Normalize dynamic weights
        total_dynamic_weight = sum(dynamic_weights.values())
        if total_dynamic_weight > 0:
            dynamic_weights = {k: v/total_dynamic_weight for k, v in dynamic_weights.items()}
        
        # Aggregate with dynamic weights
        deepfake_score = 0.0
        real_score = 0.0
        
        for label, confidence, modality in predictions:
            weight = dynamic_weights.get(modality, 0.0)
            
            if label == "Deepfake":
                deepfake_score += confidence * weight
            else:
                real_score += confidence * weight
        
        # Determine final prediction
        if deepfake_score > real_score:
            final_label = "Deepfake"
            final_confidence = deepfake_score
        else:
            final_label = "Real"
            final_confidence = real_score
        
        return final_label, round(final_confidence, 2)


def hybrid_decision(results: List[Tuple[str, float]], modalities: Optional[List[str]] = None) -> Tuple[str, float]:
    """
    Legacy function for backward compatibility with existing app.py
    
    Args:
        results: List of (label, confidence) tuples
        modalities: List of modality names corresponding to results
    
    Returns:
        Tuple of (final_label, final_confidence)
    """
    if not results:
        return "Unknown", 0.0
    
    # If modalities not provided, infer from order
    if modalities is None:
        modalities = ['image', 'video', 'audio'][:len(results)]
    
    # Convert to format expected by fusion module
    predictions = [(label, conf, mod) for (label, conf), mod in zip(results, modalities)]
    
    fusion = ModalityFusion()
    return fusion.weighted_fusion(predictions)