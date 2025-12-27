"""Evaluate the `model_basic.pth` ResNet18 model on an 80/20 test split.
Saves metrics to `experiments/exp_001/metrics.json`.
"""
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from src.data.loaders import scan_image_folder, ImageFolderDataset


def build_records(root: Path):
    base = root / 'Brain Tumor MRI images'
    records = scan_image_folder(str(base))
    return records


def stratified_split(y, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    def split_idx(idxs):
        ntest = max(1, int(len(idxs) * test_frac))
        return idxs[ntest:], idxs[:ntest]
    train0, test0 = split_idx(idx0)
    train1, test1 = split_idx(idx1)
    train_idx = np.concatenate([train0, train1])
    test_idx = np.concatenate([test0, test1])
    return train_idx, test_idx


def main():
    root = Path('Dataset')
    records = build_records(root)
    n = len(records)
    print('Total records:', n)

    # build labels array
    y = np.array([r['label'] for r in records])
    train_idx, test_idx = stratified_split(y, test_frac=0.2)

    test_records = [records[i] for i in test_idx]
    print('Test size:', len(test_records))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_ds = ImageFolderDataset(test_records, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    device = torch.device('cpu')
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    # load weights
    ckpt = Path('model_basic.pth')
    if not ckpt.exists():
        print('Model checkpoint model_basic.pth not found in project root')
        return
    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            if not isinstance(xb, torch.Tensor):
                xb = torch.tensor(xb, dtype=torch.float32)
            # ensure shape [B,C,H,W]
            if xb.ndim == 3:
                xb = xb.unsqueeze(1)
            xb = xb.to(device)
            out = model(xb)
            pred = out.argmax(dim=1).cpu().numpy()
            all_preds.append(pred)
            all_labels.append(np.array(yb))

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    acc = (preds == labels).mean()
    # per-class
    metrics = {'accuracy': float(acc), 'n_test': int(len(labels))}
    for cls in [0, 1]:
        idx = np.where(labels == cls)[0]
        if len(idx) > 0:
            metrics[f'class_{cls}_accuracy'] = float((preds[idx] == labels[idx]).mean())
            metrics[f'class_{cls}_n'] = int(len(idx))

    print('Evaluation metrics:', metrics)

    outdir = Path('experiments') / 'exp_001'
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Saved metrics to', outdir / 'metrics.json')


if __name__ == '__main__':
    main()
