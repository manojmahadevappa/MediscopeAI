"""
Convenient runner script to train both binary and multiclass models
Automatically finds dataset paths and trains with optimal hyperparameters
"""

import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"

# Dataset paths
CT_DATA = DATASET_ROOT / "Brain Tumor CT scan Images"
MRI_DATA = DATASET_ROOT / "Brain Tumor MRI images"

# Training script
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train" / "train_resnet50.py"

# Use virtual environment python if available
VENV_PYTHON = PROJECT_ROOT / "env" / "Scripts" / "python.exe"
PYTHON_EXE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

def train_binary_model():
    """Train binary classification model on CT images"""
    print("\n" + "="*70)
    print("TRAINING BINARY CT MODEL (Healthy vs Tumor)")
    print("="*70 + "\n")
    
    cmd = [
        PYTHON_EXE,
        str(TRAIN_SCRIPT),
        "--data_dir", str(CT_DATA),
        "--model_type", "binary",
        "--epochs", "50",
        "--batch_size", "32",
        "--lr", "0.0001",
        "--output_dir", str(PROJECT_ROOT)
    ]
    
    subprocess.run(cmd, check=True)
    print("\n✅ Binary model training complete!")
    print(f"Model saved as: model_binary_best.pth\n")


def train_multiclass_model():
    """Train multiclass classification model on MRI images"""
    print("\n" + "="*70)
    print("TRAINING MULTICLASS MRI MODEL (Healthy vs Tumor subtypes)")
    print("="*70 + "\n")
    
    cmd = [
        PYTHON_EXE,
        str(TRAIN_SCRIPT),
        "--data_dir", str(MRI_DATA),
        "--model_type", "multiclass",
        "--epochs", "75",
        "--batch_size", "16",
        "--lr", "0.00005",
        "--output_dir", str(PROJECT_ROOT)
    ]
    
    subprocess.run(cmd, check=True)
    print("\n✅ Multiclass model training complete!")
    print(f"Model saved as: model_multiclass_best.pth\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train brain tumor models')
    parser.add_argument('--model', choices=['binary', 'multiclass', 'both'], 
                       default='both', help='Which model(s) to train')
    
    args = parser.parse_args()
    
    if args.model in ['binary', 'both']:
        train_binary_model()
    
    if args.model in ['multiclass', 'both']:
        train_multiclass_model()
    
    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - model_binary_best.pth (CT binary classifier)")
    print("  - model_multiclass_best.pth (MRI multiclass classifier)")
    print("  - training_history_*.json (metrics)")
    print("  - training_curves_*.png (visualization)")
    print("  - confusion_matrix_*.png (performance)")
    print("\nTo use in webapp, rename:")
    print("  model_binary_best.pth → model_binary.pth")
    print("  model_multiclass_best.pth → model_multiclass.pth")
