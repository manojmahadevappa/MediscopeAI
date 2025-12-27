"""Basic training script using local `Dataset/` images.

Usage (PowerShell):
  env\Scripts\Activate.ps1; python -m src.train.train_basic --modality mri --epochs 2 --batch-size 16

This script attempts to import torch; if torch is not installed it prints instructions.
"""
import argparse
from pathlib import Path
from src.data.loaders import scan_image_folder
import json
import numpy as np


def build_datasets(root_dir: Path, modality: str):
    if modality.lower() == 'mri':
        base = root_dir / 'Brain Tumor MRI images'
    elif modality.lower() == 'ct':
        base = root_dir / 'Brain Tumor CT scan Images'
    else:
        raise ValueError('Unknown modality')
    records = scan_image_folder(str(base))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='mri', help='mri or ct')
    parser.add_argument('--data-root', type=str, default='Dataset', help='path to Dataset folder')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    root = Path(args.data_root)
    records = build_datasets(root, args.modality)
    n = len(records)
    print(f'Found {n} images for modality={args.modality}')

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import transforms, models
        from torch.utils.data import DataLoader
        from src.data.loaders import ImageFolderDataset

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Build stratified 80/20 split
        labels = np.array([r['label'] for r in records])
        idx0 = np.where(labels == 0)[0]
        idx1 = np.where(labels == 1)[0]
        rng = np.random.RandomState(42)
        rng.shuffle(idx0)
        rng.shuffle(idx1)

        def split_idx(idxs, test_frac=0.2):
            ntest = max(1, int(len(idxs) * test_frac))
            return idxs[ntest:], idxs[:ntest]

        train_idx0, test_idx0 = split_idx(idx0)
        train_idx1, test_idx1 = split_idx(idx1)
        train_idx = np.concatenate([train_idx0, train_idx1])
        test_idx = np.concatenate([test_idx0, test_idx1])

        train_records = [records[i] for i in train_idx]
        test_records = [records[i] for i in test_idx]

        print(f'Train size: {len(train_records)}, Test size: {len(test_records)}')

        train_ds = ImageFolderDataset(train_records, transform=transform)
        test_ds = ImageFolderDataset(test_records, transform=transform)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.to(args.device)

        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(args.epochs):
            running = 0.0
            total = 0
            model.train()
            for xb, yb in train_loader:
                if not isinstance(xb, torch.Tensor):
                    xb = torch.tensor(xb, dtype=torch.float32)
                if xb.ndim == 3:
                    xb = xb.unsqueeze(1)
                yb = torch.tensor(yb, dtype=torch.long)
                xb = xb.to(args.device)
                yb = yb.to(args.device)
                opt.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
                total += xb.size(0)
            print(f'Epoch {epoch+1}/{args.epochs} loss={running/total:.4f}')

        # Evaluate on held-out test set
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in test_loader:
                if not isinstance(xb, torch.Tensor):
                    xb = torch.tensor(xb, dtype=torch.float32)
                if xb.ndim == 3:
                    xb = xb.unsqueeze(1)
                xb = xb.to(args.device)
                out = model(xb)
                pred = out.argmax(dim=1).cpu().numpy()
                all_preds.append(pred)
                all_labels.append(np.array(yb))

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        acc = (preds == labels).mean()
        metrics = {'accuracy': float(acc), 'n_test': int(len(labels))}
        for cls in [0, 1]:
            idx = np.where(labels == cls)[0]
            if len(idx) > 0:
                metrics[f'class_{cls}_accuracy'] = float((preds[idx] == labels[idx]).mean())
                metrics[f'class_{cls}_n'] = int(len(idx))

        print('Test metrics:', metrics)

        # Save model and metrics
        torch.save(model.state_dict(), 'model_basic.pth')
        outdir = Path('experiments') / 'exp_001'
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print('Saved model to model_basic.pth and metrics to', outdir / 'metrics.json')

    except Exception as exc:
        print('Torch or torchvision not available or error during training:')
        print(exc)
        print('\nIf you want to run training, install dependencies:')
        print('pip install torch torchvision')


if __name__ == '__main__':
    main()
