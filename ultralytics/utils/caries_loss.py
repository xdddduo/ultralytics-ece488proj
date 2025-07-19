# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Specialized loss functions for caries detection in X-ray images."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from .loss import v8SegmentationLoss


class CariesSegmentationLoss(v8SegmentationLoss):
    """
    Enhanced segmentation loss specifically designed for caries detection.
    
    This loss function addresses the unique challenges of caries detection:
    1. Class imbalance (caries vs. healthy tissue)
    2. Small lesion detection
    3. Fine boundary accuracy
    4. X-ray specific intensity variations
    """
    
    def __init__(self, model, overlap=True):
        """
        Initialize caries segmentation loss.
        
        Args:
            model: The model to compute loss for.
            overlap (bool): Whether to use overlap mask processing.
        """
        super().__init__(model, overlap)
        
        # Caries-specific loss weights
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.boundary_weight = 1.5
        self.small_lesion_weight = 2.0
        
        # Dice loss for better segmentation
        self.dice_weight = 1.0
        
    def __call__(self, preds, targets):
        """
        Compute caries-specific segmentation loss.
        
        Args:
            preds: Model predictions.
            targets: Ground truth targets.
            
        Returns:
            Dict: Loss dictionary with caries-specific components.
        """
        # Get base segmentation loss
        loss_dict = super().__call__(preds, targets)
        
        # Extract predictions and targets
        if isinstance(preds, tuple):
            pred_boxes, pred_masks, pred_protos = preds
        else:
            pred_boxes, pred_masks, pred_protos = preds[0], preds[1], preds[2]
            
        # Extract mask predictions and targets
        pred_masks = pred_masks.view(-1, pred_masks.shape[-2], pred_masks.shape[-1])
        target_masks = targets["masks"].view(-1, targets["masks"].shape[-2], targets["masks"].shape[-1])
        
        # Caries-specific loss components
        focal_loss = self._focal_loss(pred_masks, target_masks)
        dice_loss = self._dice_loss(pred_masks, target_masks)
        boundary_loss = self._boundary_loss(pred_masks, target_masks)
        small_lesion_loss = self._small_lesion_loss(pred_masks, target_masks)
        
        # Combine losses
        total_loss = (
            loss_dict["loss"] +  # Base loss
            focal_loss +
            self.dice_weight * dice_loss +
            self.boundary_weight * boundary_loss +
            self.small_lesion_weight * small_lesion_loss
        )
        
        # Update loss dictionary
        loss_dict.update({
            "loss": total_loss,
            "focal_loss": focal_loss,
            "dice_loss": dice_loss,
            "boundary_loss": boundary_loss,
            "small_lesion_loss": small_lesion_loss
        })
        
        return loss_dict
    
    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for handling class imbalance.
        
        Args:
            pred: Predicted masks.
            target: Target masks.
            
        Returns:
            Focal loss value.
        """
        # Apply sigmoid to predictions
        pred_sigmoid = torch.sigmoid(pred)
        
        # Focal loss calculation
        ce_loss = F.binary_cross_entropy(pred_sigmoid, target, reduction='none')
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        
        return focal_loss.mean()
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss for better segmentation accuracy.
        
        Args:
            pred: Predicted masks.
            target: Target masks.
            
        Returns:
            Dice loss value.
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # Dice coefficient
        intersection = (pred_sigmoid * target).sum(dim=(1, 2))
        union = pred_sigmoid.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice.mean()
        
        return dice_loss
    
    def _boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary loss for fine edge detection.
        
        Args:
            pred: Predicted masks.
            target: Target masks.
            
        Returns:
            Boundary loss value.
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # Compute gradients for boundary detection
        pred_grad_x = torch.abs(pred_sigmoid[:, :, 1:] - pred_sigmoid[:, :, :-1])
        pred_grad_y = torch.abs(pred_sigmoid[:, 1:, :] - pred_sigmoid[:, :-1, :])
        
        target_grad_x = torch.abs(target[:, :, 1:] - target[:, :, :-1])
        target_grad_y = torch.abs(target[:, 1:, :] - target[:, :-1, :])
        
        # Boundary loss
        boundary_loss = (
            F.mse_loss(pred_grad_x, target_grad_x) +
            F.mse_loss(pred_grad_y, target_grad_y)
        )
        
        return boundary_loss
    
    def _small_lesion_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute specialized loss for small lesions.
        
        Args:
            pred: Predicted masks.
            target: Target masks.
            
        Returns:
            Small lesion loss value.
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # Calculate lesion sizes
        lesion_sizes = target.sum(dim=(1, 2))
        
        # Weight small lesions more heavily
        small_lesion_mask = lesion_sizes < lesion_sizes.mean()
        
        if small_lesion_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Enhanced loss for small lesions
        small_lesion_loss = F.binary_cross_entropy(
            pred_sigmoid[small_lesion_mask],
            target[small_lesion_mask],
            reduction='mean'
        )
        
        return small_lesion_loss


class XRayPreprocessingLoss(nn.Module):
    """
    Loss function for X-ray specific preprocessing.
    
    This loss encourages the model to learn X-ray specific features
    and handle intensity variations common in dental X-rays.
    """
    
    def __init__(self, intensity_weight: float = 0.1):
        """
        Initialize X-ray preprocessing loss.
        
        Args:
            intensity_weight (float): Weight for intensity consistency loss.
        """
        super().__init__()
        self.intensity_weight = intensity_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, original_image: torch.Tensor) -> torch.Tensor:
        """
        Compute X-ray specific preprocessing loss.
        
        Args:
            pred: Predicted masks.
            target: Target masks.
            original_image: Original X-ray image.
            
        Returns:
            X-ray preprocessing loss value.
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # Intensity consistency loss
        # Encourage predictions to be consistent with X-ray intensity patterns
        intensity_loss = self._intensity_consistency_loss(pred_sigmoid, original_image)
        
        # Contrast enhancement loss
        contrast_loss = self._contrast_enhancement_loss(pred_sigmoid, target)
        
        total_loss = self.intensity_weight * intensity_loss + contrast_loss
        
        return total_loss
    
    def _intensity_consistency_loss(self, pred: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Compute intensity consistency loss.
        
        Args:
            pred: Predicted masks.
            image: Original X-ray image.
            
        Returns:
            Intensity consistency loss value.
        """
        # Normalize image to [0, 1]
        image_norm = (image - image.min()) / (image.max() - image.min() + 1e-6)
        
        # Compute local intensity statistics
        pred_intensity = pred * image_norm
        target_intensity = image_norm
        
        # Loss based on intensity consistency
        intensity_loss = F.mse_loss(pred_intensity.mean(), target_intensity.mean())
        
        return intensity_loss
    
    def _contrast_enhancement_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute contrast enhancement loss.
        
        Args:
            pred: Predicted masks.
            target: Target masks.
            
        Returns:
            Contrast enhancement loss value.
        """
        # Encourage higher contrast in predicted regions
        pred_contrast = pred.std(dim=(1, 2))
        target_contrast = target.std(dim=(1, 2))
        
        contrast_loss = F.mse_loss(pred_contrast, target_contrast)
        
        return contrast_loss 