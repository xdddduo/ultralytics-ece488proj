# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, mask_iou


class CariesSegmentationValidator(SegmentationValidator):
    """
    Enhanced validator for caries detection in X-ray images.
    
    This validator extends the standard segmentation validator with:
    1. Caries-specific metrics (precision, recall for small lesions)
    2. Enhanced visualization for dental X-rays
    3. Specialized evaluation criteria
    4. Caries severity assessment
    """
    
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize CariesSegmentationValidator.
        
        Args:
            dataloader: DataLoader for validation.
            save_dir: Directory to save results.
            args: Arguments for validation.
            _callbacks: Callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "segment"
        
        # Caries-specific metrics
        self.caries_metrics = {
            "small_lesion_precision": [],
            "small_lesion_recall": [],
            "boundary_accuracy": [],
            "severity_accuracy": []
        }
        
    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess batch with X-ray specific enhancements.
        
        Args:
            batch: Input batch.
            
        Returns:
            Preprocessed batch.
        """
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        
        # Apply X-ray specific preprocessing
        if "img" in batch:
            batch["img"] = self._enhance_xray_contrast(batch["img"])
        
        return batch
    
    def _enhance_xray_contrast(self, images: torch.Tensor) -> torch.Tensor:
        """
        Enhance contrast for X-ray images during validation.
        
        Args:
            images: Input images.
            
        Returns:
            Contrast-enhanced images.
        """
        # Apply the same contrast enhancement as in training
        enhanced_images = []
        for img in images:
            img_np = img.cpu().numpy()
            img_enhanced = self._clahe_enhancement(img_np)
            enhanced_images.append(torch.from_numpy(img_enhanced).to(images.device))
        
        return torch.stack(enhanced_images)
    
    def _clahe_enhancement(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply CLAHE enhancement for validation.
        
        Args:
            img: Input image.
            
        Returns:
            Enhanced image.
        """
        # Same implementation as in trainer
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
        gamma = 0.8
        img_enhanced = torch.pow(img_norm, gamma)
        return img_enhanced
    
    def _process_batch(self, preds: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Process batch with caries-specific evaluation.
        
        Args:
            preds: Model predictions.
            batch: Ground truth batch.
            
        Returns:
            Dictionary with evaluation results.
        """
        # Get standard processing results
        tp = super()._process_batch(preds, batch)
        
        # Add caries-specific metrics
        gt_cls, gt_masks = batch["cls"], batch["masks"]
        if len(gt_cls) == 0 or len(preds["cls"]) == 0:
            return tp
        
        pred_masks = preds["masks"]
        
        # Small lesion detection metrics
        small_lesion_metrics = self._compute_small_lesion_metrics(pred_masks, gt_masks)
        
        # Boundary accuracy metrics
        boundary_accuracy = self._compute_boundary_accuracy(pred_masks, gt_masks)
        
        # Update results
        tp.update({
            "small_lesion_tp": small_lesion_metrics["tp"],
            "small_lesion_fp": small_lesion_metrics["fp"],
            "small_lesion_fn": small_lesion_metrics["fn"],
            "boundary_accuracy": boundary_accuracy
        })
        
        return tp
    
    def _compute_small_lesion_metrics(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Compute metrics specifically for small lesions.
        
        Args:
            pred_masks: Predicted masks.
            gt_masks: Ground truth masks.
            
        Returns:
            Dictionary with small lesion metrics.
        """
        # Define small lesion threshold (in pixels)
        small_lesion_threshold = 100  # Adjust based on your dataset
        
        # Calculate lesion sizes
        gt_sizes = gt_masks.sum(dim=(1, 2))
        pred_sizes = pred_masks.sum(dim=(1, 2))
        
        # Identify small lesions
        small_gt_mask = gt_sizes < small_lesion_threshold
        small_pred_mask = pred_sizes < small_lesion_threshold
        
        # Compute IoU for small lesions
        if small_gt_mask.sum() > 0 and small_pred_mask.sum() > 0:
            small_gt = gt_masks[small_gt_mask]
            small_pred = pred_masks[small_pred_mask]
            
            # Compute IoU
            iou = mask_iou(
                small_gt.view(small_gt.shape[0], -1),
                small_pred.view(small_pred.shape[0], -1)
            )
            
            # Threshold for true positives
            tp_threshold = 0.5
            tp = (iou > tp_threshold).sum()
            fp = small_pred_mask.sum() - tp
            fn = small_gt_mask.sum() - tp
        else:
            tp = fp = fn = 0
        
        return {
            "tp": np.array([tp]),
            "fp": np.array([fp]),
            "fn": np.array([fn])
        }
    
    def _compute_boundary_accuracy(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> np.ndarray:
        """
        Compute boundary accuracy for fine detail detection.
        
        Args:
            pred_masks: Predicted masks.
            gt_masks: Ground truth masks.
            
        Returns:
            Boundary accuracy scores.
        """
        # Apply sigmoid to predictions
        pred_sigmoid = torch.sigmoid(pred_masks)
        
        # Compute gradients for boundary detection
        pred_grad_x = torch.abs(pred_sigmoid[:, :, 1:] - pred_sigmoid[:, :, :-1])
        pred_grad_y = torch.abs(pred_sigmoid[:, 1:, :] - pred_sigmoid[:, :-1, :])
        
        gt_grad_x = torch.abs(gt_masks[:, :, 1:] - gt_masks[:, :, :-1])
        gt_grad_y = torch.abs(gt_masks[:, 1:, :] - gt_masks[:, :-1, :])
        
        # Compute boundary accuracy
        boundary_accuracy = (
            F.mse_loss(pred_grad_x, gt_grad_x) +
            F.mse_loss(pred_grad_y, gt_grad_y)
        )
        
        return np.array([boundary_accuracy.item()])
    
    def get_desc(self) -> str:
        """Return a formatted description of evaluation metrics including caries-specific ones."""
        return ("%22s" + "%11s" * 14) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Small(P",
            "R",
            "Boundary",
            "Severity)",
        )
    
    def finalize_metrics(self, *args, **kwargs):
        """Finalize metrics with caries-specific calculations."""
        # Call parent method
        super().finalize_metrics(*args, **kwargs)
        
        # Add caries-specific metrics
        self._compute_caries_metrics()
    
    def _compute_caries_metrics(self):
        """Compute caries-specific metrics."""
        # Small lesion precision and recall
        if hasattr(self, 'stats') and 'small_lesion_tp' in self.stats:
            tp = self.stats['small_lesion_tp'].sum()
            fp = self.stats['small_lesion_fp'].sum()
            fn = self.stats['small_lesion_fn'].sum()
            
            small_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            small_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            self.caries_metrics["small_lesion_precision"].append(small_precision)
            self.caries_metrics["small_lesion_recall"].append(small_recall)
        
        # Boundary accuracy
        if hasattr(self, 'stats') and 'boundary_accuracy' in self.stats:
            boundary_acc = self.stats['boundary_accuracy'].mean()
            self.caries_metrics["boundary_accuracy"].append(boundary_acc)
        
        # Log caries-specific metrics
        LOGGER.info(f"Small Lesion Precision: {small_precision:.3f}")
        LOGGER.info(f"Small Lesion Recall: {small_recall:.3f}")
        LOGGER.info(f"Boundary Accuracy: {boundary_acc:.3f}")
    
    def plot_predictions(self, batch: Dict[str, Any], preds: List[Dict[str, torch.Tensor]], ni: int) -> None:
        """
        Plot predictions with caries-specific enhancements.
        
        Args:
            batch: Input batch.
            preds: Model predictions.
            ni: Batch index.
        """
        # Apply standard plotting
        super().plot_predictions(batch, preds, ni)
        
        # Add caries-specific visualizations
        self._plot_caries_analysis(batch, preds, ni)
    
    def _plot_caries_analysis(self, batch: Dict[str, Any], preds: List[Dict[str, torch.Tensor]], ni: int):
        """
        Plot caries-specific analysis.
        
        Args:
            batch: Input batch.
            preds: Model predictions.
            ni: Batch index.
        """
        import matplotlib.pyplot as plt
        
        # Create caries analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original X-ray
        axes[0, 0].imshow(batch["img"][ni].cpu().numpy().transpose(1, 2, 0), cmap='gray')
        axes[0, 0].set_title("Original X-ray")
        axes[0, 0].axis('off')
        
        # Enhanced X-ray
        enhanced_img = self._enhance_xray_contrast(batch["img"][ni:ni+1])[0]
        axes[0, 1].imshow(enhanced_img.cpu().numpy().transpose(1, 2, 0), cmap='gray')
        axes[0, 1].set_title("Enhanced X-ray")
        axes[0, 1].axis('off')
        
        # Ground truth masks
        if "masks" in batch:
            gt_mask = batch["masks"][ni].cpu().numpy()
            axes[1, 0].imshow(gt_mask, cmap='Reds', alpha=0.7)
            axes[1, 0].set_title("Ground Truth")
            axes[1, 0].axis('off')
        
        # Predicted masks
        if preds and "masks" in preds[0]:
            pred_mask = preds[0]["masks"][0].cpu().numpy()
            axes[1, 1].imshow(pred_mask, cmap='Blues', alpha=0.7)
            axes[1, 1].set_title("Predicted Caries")
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'caries_analysis_{ni}.png')
        plt.close() 