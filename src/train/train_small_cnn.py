"""Small CNN trainer for quick CPU runs.

Usage:
  python -m src.train.train_small_cnn --modality mri --epochs 2 --batch-size 64
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.loaders import scan_image_folder, ImageFolderDataset


class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def build_dataset(root: Path, modality: str):
    if modality.lower() == 'mri':
        base = root / 'Brain Tumor MRI images'
    else:
        base = root / 'Brain Tumor CT scan Images'
    records = scan_image_folder(str(base))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='mri')
    parser.add_argument('--data-root', type=str, default='Dataset')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    records = build_dataset(Path(args.data_root), args.modality)
    n = len(records)
    print(f'Found {n} images')

    # simple split
    train_recs = records[: int(0.8 * n)]
    test_recs = records[int(0.8 * n):]

    train_ds = ImageFolderDataset(train_recs, transform=transform)
    test_ds = ImageFolderDataset(test_recs, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = SmallCNN(num_classes=2).to(args.device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        total = 0
        for xb, yb in train_loader:
            # xb: numpy array or tensor depending on dataset transform
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
        print(f'Epoch {epoch+1} train loss {running/total:.4f}')

    # Eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            if not isinstance(xb, torch.Tensor):
                xb = torch.tensor(xb, dtype=torch.float32)
            if xb.ndim == 3:
                xb = xb.unsqueeze(1)
            xb = xb.to(args.device)
            yb = torch.tensor(yb, dtype=torch.long).to(args.device)
            out = model(xb)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    print(f'Small CNN test accuracy: {correct/total:.4f} (n={total})')


if __name__ == '__main__':
    main()
