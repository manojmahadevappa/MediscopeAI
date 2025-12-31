"""
Comprehensive Model & Grad-CAM Testing Script
Tests model predictions and visualization on sample images from the dataset.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch.nn.functional as F

# Define the model architecture (same as in app.py - MultiModalNet)
class MultiModalNet(nn.Module):
    """Simple multimodal dual-encoder model"""

    def __init__(self, num_classes=3, share_weights=False, pretrained=False):
        super().__init__()
        # CT encoder
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.ct_encoder = models.resnet18(weights=weights)
        self.ct_encoder.fc = nn.Identity()

        # MRI encoder
        if share_weights:
            self.mri_encoder = self.ct_encoder
        else:
            self.mri_encoder = models.resnet18(weights=weights)
            self.mri_encoder.fc = nn.Identity()

        # feature dim from resnet18 final (512)
        feat_dim = 512

        # fusion dim = sum of available encoders
        self.fusion_dim = feat_dim * 2

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # stage regression head
        self.stage_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # survival risk head
        self.surv_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, ct=None, mri=None):
        # ct, mri: tensors or None. Expect shape [B, C, H, W]
        device = next(self.parameters()).device

        features = []
        if ct is not None:
            ct = ct.to(device)
            ct_f = self.ct_encoder(ct)  # [B, feat_dim]
            features.append(ct_f)
        if mri is not None:
            mri = mri.to(device)
            mri_f = self.mri_encoder(mri)
            features.append(mri_f)

        if not features:
            raise ValueError("At least one modality (CT or MRI) must be provided")
        elif len(features) == 1:
            # Only one modality present - use it directly, pad to fusion_dim
            fused = features[0]
            if fused.size(1) != self.fusion_dim:
                pad = self.fusion_dim - fused.size(1)
                batch_size = fused.size(0)
                zeros = torch.zeros((batch_size, pad), device=device, dtype=fused.dtype)
                fused = torch.cat([fused, zeros], dim=1)
        else:
            # Both modalities present - concatenate
            fused = torch.cat(features, dim=1)

        # normalize fused features for stability
        fused = F.normalize(fused, dim=1)

        cls_logits = self.classifier(fused)
        stage_out = self.stage_head(fused).squeeze(1)
        surv_out = self.surv_head(fused).squeeze(1)

        return {
            'logits': cls_logits,
            'stage': stage_out,
            'surv': surv_out
        }

# Image preprocessing (same as in app.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_image, class_idx=None, modality='ct'):
        # Forward pass
        if modality == 'ct':
            output = self.model(ct=input_image, mri=None)
        else:
            output = self.model(ct=None, mri=input_image)
        
        logits = output['logits']
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = logits[0, class_idx]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = torch.clamp(cam, min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cam.cpu().numpy()
        
        return cam, class_idx

def load_model(model_path):
    """Load the trained model"""
    print(f"Loading model from: {model_path}")
    
    model = MultiModalNet(num_classes=3, share_weights=False, pretrained=False)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("✅ Model loaded successfully!")
        return model
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_image(model, image_path, grad_cam, class_names, modality='ct'):
    """Test a single image and generate Grad-CAM"""
    print(f"\nTesting: {os.path.basename(image_path)}")
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        if modality == 'ct':
            output = model(ct=input_tensor, mri=None)
        else:
            output = model(ct=None, mri=input_tensor)
        
        logits = output['logits']
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class = logits.argmax(dim=1).item()
        confidence = probabilities[predicted_class].item()
    
    print(f"  Predicted: {class_names[predicted_class]} ({confidence*100:.2f}%)")
    print(f"  Probabilities: {[f'{p*100:.1f}%' for p in probabilities.tolist()]}")
    
    # Generate Grad-CAM
    try:
        cam, class_idx = grad_cam.generate(input_tensor, class_idx=predicted_class, modality=modality)
        
        # Resize CAM to match image size
        cam_resized = np.array(Image.fromarray(cam).resize(image.size, Image.BILINEAR))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Grad-CAM heatmap
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(cam_resized, cmap='jet', alpha=0.5)
        axes[2].set_title(f'Overlay - {class_names[predicted_class]} ({confidence*100:.1f}%)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_path = f"test_result_{os.path.basename(image_path)}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Visualization saved: {output_path}")
        plt.close()
        
        return True, predicted_class, confidence
    
    except Exception as e:
        print(f"  ❌ Grad-CAM failed: {e}")
        import traceback
        traceback.print_exc()
        return False, predicted_class, confidence

def main():
    """Main testing function"""
    print("=" * 70)
    print("COMPREHENSIVE MODEL & GRAD-CAM TEST")
    print("=" * 70)
    
    # Paths
    model_path = "model_multiclass.pth"
    dataset_path = "Dataset"
    
    # Class names
    class_names = ['Healthy', 'Benign', 'Malignant']
    
    # Test cases: (folder_path, expected_class, label, modality)
    test_cases = [
        # CT Healthy
        (os.path.join(dataset_path, "Brain Tumor CT scan Images", "Healthy", "ct_healthy (1).jpg"), 0, "CT Healthy", "ct"),
        (os.path.join(dataset_path, "Brain Tumor CT scan Images", "Healthy", "ct_healthy (100).jpg"), 0, "CT Healthy", "ct"),
        
        # CT Tumor
        (os.path.join(dataset_path, "Brain Tumor CT scan Images", "Tumor", "ct_tumor (1).jpg"), 2, "CT Tumor (Malignant)", "ct"),
        (os.path.join(dataset_path, "Brain Tumor CT scan Images", "Tumor", "ct_tumor (50).jpg"), 2, "CT Tumor (Malignant)", "ct"),
        
        # MRI Healthy
        (os.path.join(dataset_path, "Brain Tumor MRI images", "Healthy", "mri_healthy (1).jpg"), 0, "MRI Healthy", "mri"),
        (os.path.join(dataset_path, "Brain Tumor MRI images", "Healthy", "mri_healthy (500).jpg"), 0, "MRI Healthy", "mri"),
        
        # MRI Tumor - Glioma (Malignant)
        (os.path.join(dataset_path, "Brain Tumor MRI images", "Tumor", "glioma (1).jpg"), 2, "MRI Glioma (Malignant)", "mri"),
        (os.path.join(dataset_path, "Brain Tumor MRI images", "Tumor", "glioma (100).jpg"), 2, "MRI Glioma (Malignant)", "mri"),
        
        # MRI Tumor - Meningioma (Benign)
        (os.path.join(dataset_path, "Brain Tumor MRI images", "Tumor", "meningioma (1).jpg"), 1, "MRI Meningioma (Benign)", "mri"),
        (os.path.join(dataset_path, "Brain Tumor MRI images", "Tumor", "meningioma (500).jpg"), 1, "MRI Meningioma (Benign)", "mri"),
        
        # MRI Tumor - Pituitary (Benign)
        (os.path.join(dataset_path, "Brain Tumor MRI images", "Tumor", "pituitary (1).jpg"), 1, "MRI Pituitary (Benign)", "mri"),
        (os.path.join(dataset_path, "Brain Tumor MRI images", "Tumor", "pituitary (300).jpg"), 1, "MRI Pituitary (Benign)", "mri"),
    ]
    
    # Check model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Load model
    model = load_model(model_path)
    if model is None:
        return
    
    # Initialize Grad-CAM for both CT and MRI encoders
    ct_target_layer = model.ct_encoder.layer4[-1]
    mri_target_layer = model.mri_encoder.layer4[-1]
    
    # Use ct_encoder for CT images, mri_encoder for MRI images
    ct_grad_cam = GradCAM(model, ct_target_layer)
    mri_grad_cam = GradCAM(model, mri_target_layer)
    print("✅ Grad-CAM initialized for CT and MRI encoders")
    
    # Run tests
    print("\n" + "=" * 70)
    print("TESTING MODEL PREDICTIONS")
    print("=" * 70)
    
    results = []
    gradcam_success = 0
    correct_predictions = 0
    
    for image_path, expected_class, label, modality in test_cases:
        if not os.path.exists(image_path):
            print(f"\n⚠️  Image not found: {image_path}")
            continue
        
        print(f"\n{label}:")
        grad_cam = ct_grad_cam if modality == 'ct' else mri_grad_cam
        success, predicted_class, confidence = test_image(model, image_path, grad_cam, class_names, modality)
        
        if success:
            gradcam_success += 1
        
        if predicted_class == expected_class:
            correct_predictions += 1
            results.append(("✅", label, class_names[predicted_class], f"{confidence*100:.1f}%"))
        else:
            results.append(("❌", label, class_names[predicted_class], f"{confidence*100:.1f}%"))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal Tests: {len(test_cases)}")
    print(f"Correct Predictions: {correct_predictions}/{len(test_cases)} ({correct_predictions/len(test_cases)*100:.1f}%)")
    print(f"Grad-CAM Success: {gradcam_success}/{len(test_cases)} ({gradcam_success/len(test_cases)*100:.1f}%)")
    
    print("\n" + "-" * 70)
    print("DETAILED RESULTS:")
    print("-" * 70)
    print(f"{'Status':<8} {'Expected':<30} {'Predicted':<15} {'Confidence':<12}")
    print("-" * 70)
    for status, label, predicted, conf in results:
        print(f"{status:<8} {label:<30} {predicted:<15} {conf:<12}")
    
    print("\n" + "=" * 70)
    print("MODEL HEALTH CHECK:")
    print("=" * 70)
    accuracy = correct_predictions / len(test_cases) * 100
    
    if accuracy >= 90:
        print(f"✅ EXCELLENT: Model accuracy is {accuracy:.1f}% - Production ready!")
    elif accuracy >= 75:
        print(f"✅ GOOD: Model accuracy is {accuracy:.1f}% - Acceptable performance")
    elif accuracy >= 60:
        print(f"⚠️  FAIR: Model accuracy is {accuracy:.1f}% - Needs improvement")
    else:
        print(f"❌ POOR: Model accuracy is {accuracy:.1f}% - Requires retraining")
    
    if gradcam_success == len(test_cases):
        print("✅ Grad-CAM: All visualizations generated successfully")
    elif gradcam_success > 0:
        print(f"⚠️  Grad-CAM: {gradcam_success}/{len(test_cases)} visualizations succeeded")
    else:
        print("❌ Grad-CAM: Failed to generate visualizations")
    
    print("\n" + "=" * 70)
    print("Test images with Grad-CAM overlays saved in current directory")
    print("=" * 70)

if __name__ == "__main__":
    main()
