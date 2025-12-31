"""
Improved Multimodal Training Script
Addresses issues found in model testing:
- Class imbalance (use class weights)
- Better data augmentation
- Proper stratified splits
- Per-class metrics tracking
- Early stopping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Dataset class
class BrainTumorDataset(Dataset):
    """Dataset for brain tumor images with proper class labels"""
    
    def __init__(self, image_paths, labels, modalities, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.modalities = modalities  # 'ct' or 'mri'
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        modality = self.modalities[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            return {
                'image': image,
                'label': label,
                'modality': modality,
                'path': str(img_path)
            }
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return {
                'image': image,
                'label': label,
                'modality': modality,
                'path': str(img_path)
            }

# Model definition (same as in app.py)
class MultiModalNet(nn.Module):
    """Multimodal dual-encoder model"""

    def __init__(self, num_classes=3, share_weights=False, pretrained=True):
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
        device = next(self.parameters()).device

        features = []
        if ct is not None:
            ct = ct.to(device)
            ct_f = self.ct_encoder(ct)
            features.append(ct_f)
        if mri is not None:
            mri = mri.to(device)
            mri_f = self.mri_encoder(mri)
            features.append(mri_f)

        if not features:
            raise ValueError("At least one modality must be provided")
        elif len(features) == 1:
            fused = features[0]
            if fused.size(1) != self.fusion_dim:
                pad = self.fusion_dim - fused.size(1)
                batch_size = fused.size(0)
                zeros = torch.zeros((batch_size, pad), device=device, dtype=fused.dtype)
                fused = torch.cat([fused, zeros], dim=1)
        else:
            fused = torch.cat(features, dim=1)

        fused = F.normalize(fused, dim=1)

        cls_logits = self.classifier(fused)
        stage_out = self.stage_head(fused).squeeze(1)
        surv_out = self.surv_head(fused).squeeze(1)

        return {
            'logits': cls_logits,
            'stage': stage_out,
            'surv': surv_out
        }

def collect_dataset():
    """Collect all images from dataset with proper labels"""
    
    dataset_root = Path("Dataset")
    
    image_paths = []
    labels = []
    modalities = []
    
    class_names = ['Healthy', 'Benign', 'Malignant']
    print("\n" + "="*70)
    print("COLLECTING DATASET")
    print("="*70)
    
    # CT Healthy (label 0)
    ct_healthy = dataset_root / "Brain Tumor CT scan Images" / "Healthy"
    if ct_healthy.exists():
        ct_healthy_files = list(ct_healthy.glob("*.jpg"))
        image_paths.extend(ct_healthy_files)
        labels.extend([0] * len(ct_healthy_files))
        modalities.extend(['ct'] * len(ct_healthy_files))
        print(f"✓ CT Healthy: {len(ct_healthy_files)} images")
    
    # CT Tumor (label 2 - Malignant, based on test expectations)
    ct_tumor = dataset_root / "Brain Tumor CT scan Images" / "Tumor"
    if ct_tumor.exists():
        ct_tumor_files = list(ct_tumor.glob("*.jpg"))
        image_paths.extend(ct_tumor_files)
        labels.extend([2] * len(ct_tumor_files))  # Malignant
        modalities.extend(['ct'] * len(ct_tumor_files))
        print(f"✓ CT Tumor (Malignant): {len(ct_tumor_files)} images")
    
    # MRI Healthy (label 0)
    mri_healthy = dataset_root / "Brain Tumor MRI images" / "Healthy"
    if mri_healthy.exists():
        mri_healthy_files = list(mri_healthy.glob("*.jpg"))
        image_paths.extend(mri_healthy_files)
        labels.extend([0] * len(mri_healthy_files))
        modalities.extend(['mri'] * len(mri_healthy_files))
        print(f"✓ MRI Healthy: {len(mri_healthy_files)} images")
    
    # MRI Tumors - all in same flat directory, distinguished by filename prefix
    mri_tumor_dir = dataset_root / "Brain Tumor MRI images" / "Tumor"
    if mri_tumor_dir.exists():
        # MRI Glioma (label 2 - Malignant)
        mri_glioma_files = list(mri_tumor_dir.glob("glioma*.jpg"))
        image_paths.extend(mri_glioma_files)
        labels.extend([2] * len(mri_glioma_files))  # Malignant
        modalities.extend(['mri'] * len(mri_glioma_files))
        print(f"✓ MRI Glioma (Malignant): {len(mri_glioma_files)} images")
        
        # MRI Meningioma (label 1 - Benign)
        mri_meningioma_files = list(mri_tumor_dir.glob("meningioma*.jpg"))
        image_paths.extend(mri_meningioma_files)
        labels.extend([1] * len(mri_meningioma_files))  # Benign
        modalities.extend(['mri'] * len(mri_meningioma_files))
        print(f"✓ MRI Meningioma (Benign): {len(mri_meningioma_files)} images")
        
        # MRI Pituitary (label 1 - Benign)
        mri_pituitary_files = list(mri_tumor_dir.glob("pituitary*.jpg"))
        image_paths.extend(mri_pituitary_files)
        labels.extend([1] * len(mri_pituitary_files))  # Benign
        modalities.extend(['mri'] * len(mri_pituitary_files))
        print(f"✓ MRI Pituitary (Benign): {len(mri_pituitary_files)} images")
    
    print(f"\nTotal images: {len(image_paths)}")
    
    # Class distribution
    label_counts = Counter(labels)
    print("\nClass distribution:")
    for cls_idx, count in sorted(label_counts.items()):
        print(f"  {class_names[cls_idx]}: {count} ({count/len(labels)*100:.1f}%)")
    
    return image_paths, labels, modalities

def create_splits(image_paths, labels, modalities, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create stratified train/val/test splits"""
    
    from sklearn.model_selection import train_test_split
    
    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    modalities = np.array(modalities)
    
    # First split: train vs (val + test)
    train_paths, temp_paths, train_labels, temp_labels, train_mods, temp_mods = train_test_split(
        image_paths, labels, modalities, 
        test_size=(val_ratio + test_ratio), 
        stratify=labels, 
        random_state=42
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels, val_mods, test_mods = train_test_split(
        temp_paths, temp_labels, temp_mods,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=42
    )
    
    print("\n" + "="*70)
    print("DATASET SPLITS")
    print("="*70)
    print(f"Train: {len(train_paths)} images ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"Val:   {len(val_paths)} images ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"Test:  {len(test_paths)} images ({len(test_paths)/len(image_paths)*100:.1f}%)")
    
    return (train_paths, train_labels, train_mods), \
           (val_paths, val_labels, val_mods), \
           (test_paths, test_labels, test_mods)

def calculate_class_weights(labels):
    """Calculate class weights for imbalanced dataset"""
    label_counts = Counter(labels)
    total = len(labels)
    num_classes = len(label_counts)
    
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weight = total / (num_classes * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)

def create_data_loaders(train_data, val_data, test_data, batch_size=16):
    """Create data loaders with augmentation"""
    
    # Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Val/test transform without augmentation
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_paths, train_labels, train_mods = train_data
    val_paths, val_labels, val_mods = val_data
    test_paths, test_labels, test_mods = test_data
    
    train_dataset = BrainTumorDataset(train_paths, train_labels, train_mods, train_transform)
    val_dataset = BrainTumorDataset(val_paths, val_labels, val_mods, eval_transform)
    test_dataset = BrainTumorDataset(test_paths, test_labels, test_mods, eval_transform)
    
    # Create weighted sampler for training to handle class imbalance
    class_weights = calculate_class_weights(train_labels)
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"\nClass weights (to handle imbalance): {class_weights.tolist()}")
    
    return train_loader, val_loader, test_loader, class_weights

def train_epoch(model, loader, criterion, optimizer, device, class_names):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        modalities = batch['modality']
        
        # Split batch by modality
        ct_mask = np.array([m == 'ct' for m in modalities])
        mri_mask = np.array([m == 'mri' for m in modalities])
        
        optimizer.zero_grad()
        
        # Process CT images
        if ct_mask.any():
            ct_images = images[ct_mask]
            ct_labels = labels[ct_mask]
            
            outputs = model(ct=ct_images, mri=None)
            loss = criterion(outputs['logits'], ct_labels)
            loss.backward()
            
            _, preds = torch.max(outputs['logits'], 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(ct_labels.cpu().numpy())
            
            correct += (preds == ct_labels).sum().item()
            total += ct_labels.size(0)
            running_loss += loss.item() * ct_labels.size(0)
        
        # Process MRI images
        if mri_mask.any():
            mri_images = images[mri_mask]
            mri_labels = labels[mri_mask]
            
            outputs = model(ct=None, mri=mri_images)
            loss = criterion(outputs['logits'], mri_labels)
            loss.backward()
            
            _, preds = torch.max(outputs['logits'], 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(mri_labels.cpu().numpy())
            
            correct += (preds == mri_labels).sum().item()
            total += mri_labels.size(0)
            running_loss += loss.item() * mri_labels.size(0)
        
        optimizer.step()
        
        pbar.set_postfix({'loss': running_loss / total, 'acc': correct / total})
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def validate(model, loader, criterion, device, class_names):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            modalities = batch['modality']
            
            # Split batch by modality
            ct_mask = np.array([m == 'ct' for m in modalities])
            mri_mask = np.array([m == 'mri' for m in modalities])
            
            # Process CT images
            if ct_mask.any():
                ct_images = images[ct_mask]
                ct_labels = labels[ct_mask]
                
                outputs = model(ct=ct_images, mri=None)
                loss = criterion(outputs['logits'], ct_labels)
                
                _, preds = torch.max(outputs['logits'], 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(ct_labels.cpu().numpy())
                
                correct += (preds == ct_labels).sum().item()
                total += ct_labels.size(0)
                running_loss += loss.item() * ct_labels.size(0)
            
            # Process MRI images
            if mri_mask.any():
                mri_images = images[mri_mask]
                mri_labels = labels[mri_mask]
                
                outputs = model(ct=None, mri=mri_images)
                loss = criterion(outputs['logits'], mri_labels)
                
                _, preds = torch.max(outputs['logits'], 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(mri_labels.cpu().numpy())
                
                correct += (preds == mri_labels).sum().item()
                total += mri_labels.size(0)
                running_loss += loss.item() * mri_labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_training_history(history, save_path='training_curves.png'):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()

def main():
    print("\n" + "="*70)
    print("IMPROVED MULTIMODAL MODEL TRAINING")
    print("="*70)
    
    # Hyperparameters
    EPOCHS = 30
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    PATIENCE = 5  # Early stopping patience
    
    class_names = ['Healthy', 'Benign', 'Malignant']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Collect dataset
    image_paths, labels, modalities = collect_dataset()
    
    # Create splits
    train_data, val_data, test_data = create_splits(image_paths, labels, modalities)
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        train_data, val_data, test_data, batch_size=BATCH_SIZE
    )
    
    # Create model
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    model = MultiModalNet(num_classes=3, share_weights=False, pretrained=True).to(device)
    print("✓ Model created with pretrained ResNet18 encoders")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device, class_names
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device, class_names
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model_multiclass_best.pth')
            print(f"✓ Best model saved! (Val Acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    model.load_state_dict(torch.load('model_multiclass_best.pth'))
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, device, class_names
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Plot training curves
    plot_training_history(history, 'training_curves_improved.png')
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_preds, class_names, 'confusion_matrix_improved.png')
    
    # Save training history
    with open('training_history_improved.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("✓ Training history saved to training_history_improved.json")
    
    # Rename best model to production name
    import shutil
    shutil.copy('model_multiclass_best.pth', 'model_multiclass.pth')
    print("\n✓ Model copied to model_multiclass.pth for production use")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nGenerated files:")
    print("  - model_multiclass.pth (production model)")
    print("  - model_multiclass_best.pth (backup)")
    print("  - training_curves_improved.png")
    print("  - confusion_matrix_improved.png")
    print("  - training_history_improved.json")

if __name__ == '__main__':
    main()
