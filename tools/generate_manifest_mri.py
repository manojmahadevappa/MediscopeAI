"""Generate a `manifest.csv` for the MRI dataset.

This script scans `Dataset/Brain Tumor MRI images` and writes a CSV with columns
expected by `src.data.multimodal_dataset.MultiModalDataset`.

Usage:
  python tools\generate_manifest_mri.py --root Dataset --out manifest.csv

You can provide an optional JSON mapping file to map filename substrings
to one of the class labels: `Benign`, `Low-grade`, or `High-grade`.
If a tumor filename doesn't match any mapping key, the `class_label` will
be left empty (treated as unknown -> -1) so it won't be used for
supervised multiclass training until you supply labels.
"""
import argparse
from pathlib import Path
import csv
import json


DEFAULT_MAPPING = {
    # common substrings in filenames -> grade label (user should verify)
    "meningioma": "Benign",
    "pituitary": "Benign",
    "glioma": "High-grade"
}


def build_manifest(root: Path, out_path: Path, mapping: dict):
    base = root / 'Brain Tumor MRI images'
    if not base.exists():
        raise FileNotFoundError(f"Dataset base not found: {base}")

    healthy_dir = base / 'Healthy'
    tumor_dir = base / 'Tumor'

    rows = []

    # Healthy images: map to 'Healthy' (MultiModalDataset maps Healthy -> Benign)
    if healthy_dir.exists():
        for p in sorted(healthy_dir.iterdir()):
            if p.is_file():
                rows.append({
                    'patient_id': p.stem,
                    'ct_path': '',
                    'mri_path': str(p.relative_to(root)),
                    'class_label': 'Healthy',
                    'stage': '',
                    'survival_time': '',
                    'censor': '',
                    'age': '',
                    'sex': '',
                    'notes': ''
                })

    # Tumor images: try to infer label from filename using mapping
    if tumor_dir.exists():
        for p in sorted(tumor_dir.iterdir()):
            if not p.is_file():
                continue
            fname = p.name.lower()
            cls = ''
            for key, label in mapping.items():
                if key.lower() in fname:
                    cls = label
                    break
            rows.append({
                'patient_id': p.stem,
                'ct_path': '',
                'mri_path': str(p.relative_to(root)),
                'class_label': cls,
                'stage': '',
                'survival_time': '',
                'censor': '',
                'age': '',
                'sex': '',
                'notes': ''
            })

    # write CSV
    fieldnames = ['patient_id','ct_path','mri_path','class_label','stage','survival_time','censor','age','sex','notes']
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, default='Dataset', help='Path to Dataset folder')
    p.add_argument('--out', type=str, default='manifest.csv', help='Output manifest CSV path')
    p.add_argument('--map', type=str, default=None, help='Optional JSON mapping file (substring -> grade label)')
    args = p.parse_args()

    mapping = DEFAULT_MAPPING.copy()
    if args.map:
        mpath = Path(args.map)
        if not mpath.exists():
            raise FileNotFoundError(f'Mapping file not found: {mpath}')
        with mpath.open('r', encoding='utf-8') as f:
            user_map = json.load(f)
            mapping.update(user_map)

    root = Path(args.root)
    out = Path(args.out)
    n = build_manifest(root, out, mapping)
    print(f'Wrote manifest with {n} rows to {out} (mapping used: {mapping})')


if __name__ == '__main__':
    main()
