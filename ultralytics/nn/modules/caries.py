# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Caries detection specialized modules for X-ray image analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

from .block import Proto
from .conv import Conv
from .head import Segment


class XRayAttention(nn.Module):
    """
    X-Ray specific attention module for detecting fine details in dental X-ray images.
    
    This module is designed to focus on subtle intensity variations and edge patterns
    that are characteristic of caries lesions in X-ray images.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize X-Ray attention module.
        
        Args:
            channels (int): Number of input channels.
            reduction (int): Reduction factor for channel attention.
        """
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # Spatial attention for edge detection
        self.spatial_conv = nn.Sequential(
            Conv(channels, channels // 4, 7, 1, 3, g=1),  # Regular conv instead of depthwise
            Conv(channels // 4, 1, 1, 1)  # Pointwise conv
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through X-Ray attention module.
        
        Args:
            x (torch.Tensor): Input feature map.
            
        Returns:
            torch.Tensor: Attention-weighted feature map.
        """
        # Channel attention
        avg_out = self.channel_fc(self.avg_pool(x))
        max_out = self.channel_fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        
        # Spatial attention
        spatial_att = self.sigmoid(self.spatial_conv(x))
        
        # Apply attention
        x = x * channel_att * spatial_att
        return x


class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-scale feature fusion module for detecting caries at different scales.
    
    This module combines features from different resolution levels to better
    detect caries lesions that can vary significantly in size.
    """
    
    def __init__(self, channels: Union[List[int], Tuple], out_channels: int):
        """
        Initialize multi-scale feature fusion module.
        
        Args:
            channels (Union[List[int], Tuple]): List or tuple of channel counts for each scale.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        # Convert to list if it's a tuple
        if isinstance(channels, tuple):
            channels = list(channels)
        self.num_scales = len(channels)
        self.out_channels = out_channels
        
        # Scale-specific processing
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                Conv(ch, out_channels // self.num_scales, 3, 1, 1),
                XRayAttention(out_channels // self.num_scales)
            ) for ch in channels
        ])
        
        # Ensure the total output channels match
        total_channels = (out_channels // self.num_scales) * self.num_scales
        if total_channels != out_channels:
            # Adjust the last conv to make up the difference
            last_channels = out_channels - (out_channels // self.num_scales) * (self.num_scales - 1)
            self.scale_convs[-1] = nn.Sequential(
                Conv(channels[-1], last_channels, 3, 1, 1),
                XRayAttention(last_channels)
            )
        
        # Final fusion
        self.fusion_conv = Conv(out_channels, out_channels, 3, 1, 1)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through multi-scale feature fusion.
        
        Args:
            features (List[torch.Tensor]): List of feature maps at different scales.
            
        Returns:
            torch.Tensor: Fused feature map.
        """
        # Process each scale
        processed_features = []
        target_size = features[0].shape[-2:]  # Use the first feature's spatial size as target
        
        for i, (feat, conv) in enumerate(zip(features, self.scale_convs)):
            processed_feat = conv(feat)
            # Resize to target size if different
            if processed_feat.shape[-2:] != target_size:
                processed_feat = F.interpolate(processed_feat, size=target_size, mode='bilinear', align_corners=False)
            processed_features.append(processed_feat)
        
        # Concatenate and fuse
        fused = torch.cat(processed_features, dim=1)
        fused = self.fusion_conv(fused)
        
        return fused


class CariesSegment(Segment):
    """
    Enhanced segmentation head specifically designed for caries detection in X-ray images.
    
    This module extends the standard Segment head with:
    1. X-Ray specific attention mechanisms
    2. Multi-scale feature fusion
    3. Enhanced mask prediction for fine details
    4. Specialized loss weighting for small lesions
    """
    
    def __init__(self, nc: int = 1, nm: int = 32, npr: int = 256, ch: Union[Tuple, List] = ()):
        """
        Initialize CariesSegment head.
        
        Args:
            nc (int): Number of classes (1 for caries).
            nm (int): Number of masks.
            npr (int): Number of protos.
            ch (Union[Tuple, List]): Tuple or list of channel sizes from backbone feature maps.
        """
        # Convert ch to tuple if it's a list
        if isinstance(ch, list):
            ch = tuple(ch)
        super().__init__(nc, nm, npr, ch)
        
        # Enhanced prototype generation with attention
        self.enhanced_proto = nn.Sequential(
            Proto(ch[0], self.npr, self.nm),
            XRayAttention(self.nm)
        )
        
        # Multi-scale feature fusion
        self.feature_fusion = MultiScaleFeatureFusion(ch, ch[0])
        
        # Enhanced mask coefficient prediction
        c4 = max(ch[0] // 4, self.nm)
        self.enhanced_cv4 = nn.ModuleList([
            nn.Sequential(
                Conv(x, c4, 3),
                XRayAttention(c4),
                Conv(c4, c4, 3),
                nn.Conv2d(c4, self.nm, 1)
            ) for x in ch
        ])
        
        # Edge enhancement for fine detail detection
        self.edge_enhancement = nn.ModuleList([
            nn.Sequential(
                Conv(x, x // 4, 3, 1, 1),
                Conv(x // 4, x // 4, 3, 1, 1),  # Regular conv instead of dilated
                Conv(x // 4, x, 1, 1)
            ) for x in ch
        ])
        
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        # Apply edge enhancement
        enhanced_features = []
        for feat, edge_conv in zip(x, self.edge_enhancement):
            enhanced_feat = feat + edge_conv(feat)  # Residual connection
            enhanced_features.append(enhanced_feat)
        
        # Multi-scale feature fusion
        fused_features = self.feature_fusion(enhanced_features)
        
        # Generate enhanced prototypes
        p = self.enhanced_proto(fused_features)  # mask protos
        bs = p.shape[0]  # batch size
        
        # Enhanced mask coefficients
        mc = torch.cat([
            self.enhanced_cv4[i](enhanced_features[i]).view(bs, self.nm, -1) 
            for i in range(self.nl)
        ], 2)  # mask coefficients
        
        # Standard detection forward pass
        detect_out = super(Segment, self).forward(enhanced_features)
        
        if self.training:
            return detect_out, mc, p
        # For non-training, mimic Segment: return only the tensor
        return torch.cat([detect_out[0], mc], 1)


class CariesLossWeighting(nn.Module):
    """
    Specialized loss weighting for caries detection.
    
    This module provides adaptive loss weighting to handle the class imbalance
    and small lesion detection challenges in caries detection.
    """
    
    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        """
        Initialize caries loss weighting.
        
        Args:
            focal_alpha (float): Alpha parameter for focal loss.
            focal_gamma (float): Gamma parameter for focal loss.
        """
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss for caries detection.
        
        Args:
            pred (torch.Tensor): Predicted masks.
            target (torch.Tensor): Target masks.
            
        Returns:
            torch.Tensor: Weighted loss.
        """
        # Focal loss for handling class imbalance
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        
        # Additional weighting for small lesions
        lesion_size = target.sum(dim=(2, 3))  # Sum over spatial dimensions
        small_lesion_weight = torch.clamp(1.0 / (lesion_size + 1e-6), max=10.0)
        small_lesion_weight = small_lesion_weight.unsqueeze(-1).unsqueeze(-1)
        
        weighted_loss = focal_loss * small_lesion_weight
        return weighted_loss.mean() 