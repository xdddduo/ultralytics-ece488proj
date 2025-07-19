# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from ultralytics.models import yolo
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.caries_loss import CariesSegmentationLoss


class CariesSegmentationTrainer(yolo.segment.SegmentationTrainer):
    """
    Specialized trainer for caries detection in X-ray images.
    
    This trainer extends the standard segmentation trainer with:
    1. X-ray specific preprocessing
    2. Enhanced data augmentation for dental images
    3. Specialized loss functions
    4. Caries-specific validation metrics
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides: Optional[Dict] = None, _callbacks=None):
        """
        Initialize CariesSegmentationTrainer.
        
        Args:
            cfg (dict): Configuration dictionary with default training settings.
            overrides (dict, optional): Dictionary of parameter overrides.
            _callbacks (list, optional): List of callback functions.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)
        
        # Caries-specific configuration
        self.caries_config = {
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "boundary_weight": 1.5,
            "small_lesion_weight": 2.0,
            "dice_weight": 1.0
        }
        
    def get_model(
        self, cfg: Optional[Union[Dict, str]] = None, weights: Optional[Union[str, Path]] = None, verbose: bool = True
    ):
        """
        Initialize and return a CariesSegmentationModel.
        
        Args:
            cfg (dict | str, optional): Model configuration.
            weights (str | Path, optional): Path to pretrained weights file.
            verbose (bool): Whether to display model information.
            
        Returns:
            CariesSegmentationModel: Initialized caries segmentation model.
        """
        model = CariesSegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        
        return model
    
    def get_validator(self):
        """Return an instance of CariesSegmentationValidator for validation."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "focal_loss", "dice_loss", "boundary_loss", "small_lesion_loss"
        return yolo.caries.CariesSegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
    
    def init_criterion(self):
        """Initialize the caries-specific loss criterion."""
        return CariesSegmentationLoss(self.model, overlap=self.args.overlap_mask)
    
    def preprocess_batch(self, batch):
        """
        Preprocess batch with X-ray specific enhancements.
        
        Args:
            batch: Input batch.
            
        Returns:
            Preprocessed batch.
        """
        # Apply standard preprocessing
        batch = super().preprocess_batch(batch)
        
        # X-ray specific preprocessing
        if "img" in batch:
            batch["img"] = self._enhance_xray_contrast(batch["img"])
        
        return batch
    
    def _enhance_xray_contrast(self, images: torch.Tensor) -> torch.Tensor:
        """
        Enhance contrast for X-ray images.
        
        Args:
            images: Input images.
            
        Returns:
            Contrast-enhanced images.
        """
        # Histogram equalization for better contrast
        enhanced_images = []
        for img in images:
            # Convert to numpy for processing
            img_np = img.cpu().numpy()
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # This is a simplified version - in practice, you might want to use cv2.createCLAHE
            img_enhanced = self._clahe_enhancement(img_np)
            
            # Convert back to tensor
            enhanced_images.append(torch.from_numpy(img_enhanced).to(images.device))
        
        return torch.stack(enhanced_images)
    
    def _clahe_enhancement(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply CLAHE enhancement to improve X-ray contrast.
        
        Args:
            img: Input image.
            
        Returns:
            Enhanced image.
        """
        # Simplified CLAHE implementation
        # In practice, you would use cv2.createCLAHE for better results
        
        # Normalize to [0, 1]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
        
        # Apply gamma correction for contrast enhancement
        gamma = 0.8
        img_enhanced = torch.pow(img_norm, gamma)
        
        return img_enhanced
    
    def plot_metrics(self):
        """Plot training/validation metrics with caries-specific components."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)
        
        # Add caries-specific metric plots
        if hasattr(self, 'caries_metrics'):
            self._plot_caries_metrics()
    
    def _plot_caries_metrics(self):
        """Plot caries-specific metrics."""
        import matplotlib.pyplot as plt
        
        # Plot focal loss, dice loss, etc.
        metrics = ['focal_loss', 'dice_loss', 'boundary_loss', 'small_lesion_loss']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            if metric in self.caries_metrics:
                axes[i].plot(self.caries_metrics[metric])
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'caries_metrics.png')
        plt.close()


class CariesSegmentationModel(SegmentationModel):
    """
    Enhanced segmentation model specifically for caries detection.
    
    This model includes:
    1. X-ray specific preprocessing layers
    2. Enhanced feature extraction
    3. Specialized loss computation
    """
    
    def __init__(self, cfg="yolov8-caries-seg.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize CariesSegmentationModel.
        
        Args:
            cfg (str | dict): Model configuration.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        
        # Add X-ray preprocessing layer
        self.xray_preprocessing = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, *args, **kwargs):
        """
        Forward pass with X-ray preprocessing.
        
        Args:
            x: Input tensor.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Model output.
        """
        # Apply X-ray preprocessing
        if isinstance(x, dict):
            x["img"] = self.xray_preprocessing(x["img"])
        else:
            x = self.xray_preprocessing(x)
        
        return super().forward(x, *args, **kwargs)
    
    def init_criterion(self):
        """Initialize caries-specific loss criterion."""
        return CariesSegmentationLoss(self) 