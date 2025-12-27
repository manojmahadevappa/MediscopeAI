import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.optim import Adam

from src.data.multimodal_dataset import MultiModalDataset
from src.models.multimodal import MultiModalNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=str, default='manifest.csv')
    p.add_argument('--root', type=str, default='.')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--output', type=str, default='model_multimodal.pth')
    return p.parse_args()


def collate_fn(batch):
    # batch: list of samples
    ct_list = []
    mri_list = []
    labels = []
    meta = []
    for s in batch:
        ct_list.append(s['ct'])
        mri_list.append(s['mri'])
        labels.append(s['class_label'])
        meta.append({'age': s.get('age'), 'sex': s.get('sex')})

    # convert lists to tensors, allowing None
    def stack_or_none(lst):
        if all(x is None for x in lst):
            return None
        # for missing entries replace with zeros
        tensors = []
        for x in lst:
            if x is None:
                # create zeros tensor of expected shape
                tensors.append(torch.zeros(3, 224, 224))
            else:
                tensors.append(x)
        return torch.stack(tensors, dim=0)

    ct_batch = stack_or_none(ct_list)
    mri_batch = stack_or_none(mri_list)

    labels_tensor = torch.tensor([l if l >= 0 else -1 for l in labels], dtype=torch.long)

    return {'ct': ct_batch, 'mri': mri_batch, 'labels': labels_tensor, 'meta': meta}


def train():
    args = parse_args()
    ds = MultiModalDataset(args.manifest, root_dir=args.root)
    # split
    n = len(ds)
    if n < 2:
        raise RuntimeError('Not enough samples in dataset')
    val_size = max(1, int(0.2 * n))
    train_size = n - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalNet(num_classes=3, share_weights=False, pretrained=False).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    mse_loss = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            ct = batch['ct'].to(device) if batch['ct'] is not None else None
            mri = batch['mri'].to(device) if batch['mri'] is not None else None
            labels = batch['labels'].to(device)

            # Debug: print shapes on first batch
            if batch_idx == 0:
                print(f"Debug - CT shape: {ct.shape if ct is not None else 'None'}")
                print(f"Debug - MRI shape: {mri.shape if mri is not None else 'None'}")
                print(f"Debug - Labels shape: {labels.shape}")

            out = model(ct=ct, mri=mri)
            logits = out['logits']
            stage_pred = out['stage']

            # classification loss (only where label != -1)
            loss = ce_loss(logits, labels)

            # TODO: incorporate stage and survival losses when labels available

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{args.epochs} train_loss={avg_loss:.4f}')

        # quick validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                ct = batch['ct'].to(device) if batch['ct'] is not None else None
                mri = batch['mri'].to(device) if batch['mri'] is not None else None
                labels = batch['labels'].to(device)
                out = model(ct=ct, mri=mri)
                logits = out['logits']
                preds = torch.argmax(logits, dim=1)
                mask = labels != -1
                if mask.any():
                    correct += (preds[mask] == labels[mask]).sum().item()
                    total += int(mask.sum().item())
        if total > 0:
            print(f'Validation accuracy: {correct}/{total} = {correct/total:.4f}')
        else:
            print('Validation: no labeled samples to compute accuracy')

        # save checkpoint each epoch
        torch.save(model.state_dict(), args.output)


if __name__ == '__main__':
    train()
