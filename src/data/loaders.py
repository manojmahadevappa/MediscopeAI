"""Dataset and dataloader helpers (stubs).
"""
from typing import Optional, List, Dict
from pathlib import Path
from PIL import Image
import numpy as np


def scan_image_folder(base_dir: str) -> List[Dict]:
    """Scan a folder with subfolders `Healthy` and `Tumor` and return records.

    Expected layout:
      base_dir/
        Healthy/
          img1.jpg
        Tumor/
          img2.jpg

    Returns list of dicts: {"path": str, "label": 0|1}
    """
    p = Path(base_dir)
    records: List[Dict] = []
    mapping = {"Healthy": 0, "Tumor": 1}
    for cls_name, label in mapping.items():
        folder = p / cls_name
        if not folder.exists():
            continue
        for img_path in folder.iterdir():
            if img_path.is_file():
                records.append({"path": str(img_path), "label": label})
    return records


class ImageFolderDataset:
    """Lightweight image dataset that returns numpy arrays and labels.

    This avoids importing torch at scan time so a quick smoke-test can be run.
    If you want a PyTorch Dataset, wrap this or modify to return tensors.
    """
    def __init__(self, records: List[Dict], transform: Optional[callable] = None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        arr = np.array(img)
        if self.transform:
            arr = self.transform(arr)
        return arr, rec["label"]

