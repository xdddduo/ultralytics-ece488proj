#!/usr/bin/env python3
"""
Example script for using the modified YOLOv8-segmentation model for caries detection.

This script demonstrates:
1. Loading the custom model configuration
2. Training with caries-specific components
3. Validation with enhanced metrics
4. Inference with medical-grade confidence calibration
"""

import os
import sys
from pathlib import Path
import torch

# Add the ultralytics directory to the path
sys.path.append(str(Path(__file__).parent))

import ultralytics.nn.modules.caries  # Ensure CariesSegment is registered
from ultralytics import YOLO
from ultralytics.models.yolo.caries import (
    CariesSegmentationTrainer,
    CariesSegmentationValidator,
    CariesSegmentationPredictor
)


def get_device():
    """Automatically detect and return the best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🚀 CUDA detected! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        print("💻 CUDA not available. Using CPU.")
    return device


def train_caries_model():
    """Train the custom caries detection model."""
    
    print("\n🎯 Starting Caries Detection Model Training")
    print("=" * 50)
    
    # Model configuration
    model_config = "ultralytics/cfg/models/v8/yolov8-caries-seg.yaml"
    
    if not os.path.exists(model_config):
        print(f"❌ Model configuration not found: {model_config}")
        print("Please ensure the YOLO configuration file exists.")
        return
    
    # Initialize the model
    model = YOLO(model_config)
    
    # Get the best available device
    device = get_device()
    
    # Training configuration
    training_config = {
        'data': 'dentex-2-clahe-cropped/data.yaml',  # Updated to use your dataset
        'epochs': 100,
        'imgsz': 640,
        'batch': 16 if device == 'cuda' else 8,  # Larger batch for GPU, smaller for CPU
        'device': device,
        'workers': 8 if device == 'cuda' else 4,  # More workers for GPU
        'project': 'runs/caries_detection',
        'name': 'train',
        'save': True,
        'save_period': 10,
        'patience': 50,
        'optimizer': 'AdamW',
        'lr0': 1e-4,
        'weight_decay': 0.01,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'dfl': 1.5,  # DFL loss gain
        'pose': 12.0,  # Pose loss gain
        'kobj': 2.0,  # Keypoint obj loss gain
        'label_smoothing': 0.0,  # Label smoothing epsilon
        'nbs': 64,  # Nominal batch size
        'overlap_mask': True,  # Masks should overlap during training
        'mask_ratio': 4,  # Mask downsample ratio
        'dropout': 0.0,  # Use dropout regularization
        'val': True,  # Validate during training
    }
    
    try:
        # Start training
        print("📊 Training Configuration:")
        for key, value in training_config.items():
            print(f"  {key}: {value}")
        
        print("\n🎯 Starting training...")
        results = model.train(**training_config)
        
        print("✅ Training completed successfully!")
        print("📁 Results saved to: runs/caries_detection/train")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("Please check your dataset configuration and hardware requirements.")


def validate_caries_model():
    """Validate the trained caries detection model."""
    
    print("\n🔍 Starting Caries Detection Model Validation")
    print("=" * 50)
    
    # Path to trained model
    model_path = "runs/caries_detection/train/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        print("Please train the model first or update the model path.")
        return
    
    try:
        # Load the trained model
        model = YOLO(model_path)
        
        # Get the best available device
        device = get_device()
        
        # Validation configuration
        validation_config = {
            'data': 'dentex-2-clahe-cropped/data.yaml',  # Updated to use your dataset
            'imgsz': 640,
            'batch': 16 if device == 'cuda' else 8,  # Larger batch for GPU, smaller for CPU
            'device': device,
            'workers': 8 if device == 'cuda' else 4,  # More workers for GPU
            'project': 'runs/caries_detection',
            'name': 'validate',
            'save_txt': True,
            'save_conf': True,
            'save_json': True,
            'conf': 0.3,  # Lower confidence threshold for medical applications
            'iou': 0.5,
            'max_det': 300,
            'half': True if device == 'cuda' else False,  # Use half precision only on GPU
            'dnn': False,
            'plots': True,
        }
        
        print("📊 Validation Configuration:")
        for key, value in validation_config.items():
            print(f"  {key}: {value}")
        
        print("\n🔍 Starting validation...")
        results = model.val(**validation_config)
        
        print("✅ Validation completed successfully!")
        print(f"📁 Results saved to: {results.save_dir}")
        
        # Print caries-specific metrics
        if hasattr(results, 'caries_metrics'):
            print("\n📈 Caries-Specific Metrics:")
            for metric, value in results.caries_metrics.items():
                print(f"  {metric}: {value:.3f}")
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")


def predict_caries():
    """Run inference on new X-ray images."""
    
    print("\n🔮 Starting Caries Detection Inference")
    print("=" * 50)
    
    # Path to trained model
    model_path = "runs/caries_detection/train/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        print("Please train the model first or update the model path.")
        return
    
    # Path to test images
    test_images = "dentex-2-clahe-cropped/test/images"  # Updated to use your test images
    
    if not os.path.exists(test_images):
        print(f"❌ Test images not found: {test_images}")
        print("Please update the test image path.")
        return
    
    try:
        # Load the trained model
        model = YOLO(model_path)
        
        # Get the best available device
        device = get_device()
        
        # Create custom predictor
        predictor = CariesSegmentationPredictor(
            overrides={
                'conf': 0.3,  # Medical-grade confidence threshold
                'iou': 0.5,
                'max_det': 300,
                'save_txt': True,
                'save_conf': True,
                'save_crop': True,
                'show_labels': True,
                'show_conf': True,
                'show': False,
                'save': True,
            }
        )
        
        # Inference configuration
        inference_config = {
            'source': test_images,
            'conf': 0.3,
            'iou': 0.5,
            'max_det': 300,
            'device': device,  # Use detected device
            'show': False,
            'save': True,
            'save_txt': True,
            'save_conf': True,
            'save_crop': True,
            'project': 'runs/caries_detection',
            'name': 'predict',
            'exist_ok': True,
            'line_thickness': 3,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'classes': None,
            'retina_masks': False,
            'boxes': True,
            'half': True if device == 'cuda' else False,  # Use half precision only on GPU
        }
        
        print("📊 Inference Configuration:")
        for key, value in inference_config.items():
            print(f"  {key}: {value}")
        
        print("\n🔮 Starting inference...")
        results = model.predict(**inference_config)
        
        print("✅ Inference completed successfully!")
        print(f"📁 Results saved to: runs/caries_detection/predict")
        
        # Note: Caries report generation would be implemented here
        print("📋 Caries detection results saved!")
        print("📁 Check the output directory for detailed results.")
        
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")


def create_dataset_yaml():
    """Create a sample dataset configuration file."""
    
    print("\n📋 Creating Sample Dataset Configuration")
    print("=" * 50)
    
    dataset_config = """
# Caries Detection Dataset Configuration
# Update paths according to your dataset structure

path: ../datasets/caries  # Dataset root directory
train: images/train  # Train images (relative to 'path')
val: images/val      # Val images (relative to 'path')
test: images/test    # Test images (optional, relative to 'path')

# Classes
nc: 1  # Number of classes
names: ['caries']  # Class names

# Additional dataset information
# - Ensure images are in common formats (jpg, png, etc.)
# - Annotations should be in YOLO format with segmentation masks
# - Recommended image size: 640x640 or larger
# - Ensure proper contrast and quality for X-ray images
"""
    
    # Create the dataset configuration file
    config_path = "caries_dataset.yaml"
    with open(config_path, 'w') as f:
        f.write(dataset_config.strip())
    
    print(f"✅ Dataset configuration created: {config_path}")
    print("📝 Please update the paths according to your dataset structure.")


def main():
    """Main function to run the caries detection example."""
    
    print("🦷 YOLOv8 Caries Detection Example")
    print("=" * 50)
    print("This example demonstrates the modified YOLOv8-segmentation")
    print("architecture for dental caries detection in X-ray images.")
    print()
    
    # Check if required files exist
    model_config = "ultralytics/cfg/models/v8/yolov8-caries-seg.yaml"
    if not os.path.exists(model_config):
        print("❌ Required files not found!")
        print("Please ensure you have:")
        print("1. The modified YOLOv8 source code")
        print("2. The yolov8-caries-seg.yaml configuration file")
        print("3. The caries-specific modules")
        return
    
    print("✅ Required files found!")
    print()
    
    # Show available options
    print("Available operations:")
    print("1. Create dataset configuration")
    print("2. Train caries detection model")
    print("3. Validate trained model")
    print("4. Run inference on test images")
    print("5. Run complete pipeline (train + validate + predict)")
    print()
    
    try:
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            create_dataset_yaml()
        elif choice == "2":
            train_caries_model()
        elif choice == "3":
            validate_caries_model()
        elif choice == "4":
            predict_caries()
        elif choice == "5":
            print("\n🔄 Running complete pipeline...")
            train_caries_model()
            validate_caries_model()
            predict_caries()
        else:
            print("❌ Invalid choice. Please enter a number between 1-5.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Operation cancelled by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
    
    print("\n🎉 Example completed!")
    print("📖 For more information, see README_CARIES_DETECTION.md")


if __name__ == "__main__":
    main() 