# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG, ops
import torch
import numpy as np


class CariesSegmentationPredictor(SegmentationPredictor):
    """
    Enhanced predictor for caries detection in X-ray images.
    
    This predictor extends the standard segmentation predictor with:
    1. X-ray specific preprocessing
    2. Enhanced visualization for dental images
    3. Caries severity assessment
    4. Confidence calibration for medical applications
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize CariesSegmentationPredictor.
        
        Args:
            cfg: Configuration dictionary.
            overrides: Configuration overrides.
            _callbacks: Callback functions.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"
        
        # Caries-specific parameters
        self.confidence_threshold = 0.3  # Lower threshold for medical applications
        self.severity_thresholds = {
            "mild": 0.3,
            "moderate": 0.6,
            "severe": 0.8
        }
    
    def preprocess(self, im):
        """
        Preprocess images with X-ray specific enhancements.
        
        Args:
            im: Input images.
            
        Returns:
            Preprocessed images.
        """
        # Apply standard preprocessing
        im = super().preprocess(im)
        
        # Apply X-ray specific enhancement
        if isinstance(im, torch.Tensor):
            im = self._enhance_xray_contrast(im)
        
        return im
    
    def _enhance_xray_contrast(self, images: torch.Tensor) -> torch.Tensor:
        """
        Enhance contrast for X-ray images during inference.
        
        Args:
            images: Input images.
            
        Returns:
            Contrast-enhanced images.
        """
        enhanced_images = []
        for img in images:
            img_np = img.cpu().numpy()
            img_enhanced = self._clahe_enhancement(img_np)
            enhanced_images.append(torch.from_numpy(img_enhanced).to(images.device))
        
        return torch.stack(enhanced_images)
    
    def _clahe_enhancement(self, img: np.ndarray) -> np.ndarray:
        """
        Apply simplified contrast enhancement for inference.
        
        Args:
            img: Input image.
            
        Returns:
            Enhanced image.
        """
        # Simplified contrast enhancement (gamma correction)
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
        gamma = 0.8
        img_enhanced = np.power(img_norm, gamma)
        
        return img_enhanced
    
    def postprocess(self, preds, img, orig_imgs):
        """
        Postprocess predictions with caries-specific enhancements.
        
        Args:
            preds: Model predictions.
            img: Preprocessed images.
            orig_imgs: Original images.
            
        Returns:
            Postprocessed results.
        """
        # Get standard postprocessing
        results = super().postprocess(preds, img, orig_imgs)
        
        # Add caries-specific postprocessing
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                # Apply confidence threshold if available
                if hasattr(result, 'conf') and result.conf is not None:
                    confident_indices = result.conf > self.confidence_threshold
                    if confident_indices.sum() > 0:
                        result.masks = result.masks[confident_indices]
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            result.boxes = result.boxes[confident_indices]
                        result.conf = result.conf[confident_indices]
        
        return results
    
    def _assess_caries_severity(self, masks, confidences):
        """
        Assess the severity of detected caries lesions.
        
        Args:
            masks: Predicted masks.
            confidences: Confidence scores.
            
        Returns:
            List of severity assessments.
        """
        severity_assessments = []
        
        for mask, conf in zip(masks, confidences):
            # Calculate lesion area
            lesion_area = mask.sum().item()
            
            # Calculate average intensity (darker = more severe)
            avg_intensity = mask.mean().item()
            
            # Combine factors for severity assessment
            severity_score = (1 - avg_intensity) * conf.item() * (lesion_area / 1000)  # Normalize area
            
            # Classify severity
            if severity_score < self.severity_thresholds["mild"]:
                severity = "mild"
            elif severity_score < self.severity_thresholds["moderate"]:
                severity = "moderate"
            else:
                severity = "severe"
            
            severity_assessments.append({
                "severity": severity,
                "score": severity_score,
                "area": lesion_area,
                "confidence": conf.item()
            })
        
        return severity_assessments 