from pathlib import Path
from typing import Optional, List, Dict, Any
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MultiModalDataset(Dataset):
    """Dataset that loads paired CT and MRI images from a manifest CSV.

    CSV columns expected: patient_id, ct_path, mri_path, class_label, stage, survival_time, censor, age, sex, notes
    Missing modality paths should be empty strings.
    Unknown/empty class_label values will be treated as -1 (unlabeled) for supervised classification.
    """

    def __init__(self, manifest_path: str, root_dir: Optional[str] = None, transforms_img=None):
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir) if root_dir else None
        self.rows = []
        self.transforms = transforms_img or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self._load_manifest()

    def _load_manifest(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        with open(self.manifest_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                # normalize paths relative to repo root if needed
                ct = r.get('ct_path','') or ''
                mri = r.get('mri_path','') or ''
                if self.root_dir:
                    if ct:
                        ct = str((self.root_dir / ct).resolve())
                    if mri:
                        mri = str((self.root_dir / mri).resolve())
                # map class_label to integer: Healthy=0, Benign=1, Malignant/High-grade=2 ; unknown/empty -> -1
                lbl = (r.get('class_label') or '').strip().lower()
                if lbl in ('healthy','normal'):
                    cls = 0
                elif lbl in ('benign','benign tumor','low-grade','low_grade','low'):
                    cls = 1
                elif lbl in ('high-grade','high_grade','high','malignant'):
                    cls = 2
                else:
                    cls = -1

                # optional numeric fields
                stage = r.get('stage') or ''
                try:
                    stage_val = float(stage) if stage != '' else None
                except Exception:
                    stage_val = None

                survival_time = r.get('survival_time') or ''
                try:
                    survival_val = float(survival_time) if survival_time != '' else None
                except Exception:
                    survival_val = None

                censor = r.get('censor') or ''
                try:
                    censor_val = int(censor) if censor != '' else None
                except Exception:
                    censor_val = None

                age = r.get('age') or ''
                try:
                    age_val = float(age) if age != '' else None
                except Exception:
                    age_val = None

                sex = (r.get('sex') or '').strip() or None

                self.rows.append({
                    'patient_id': r.get('patient_id'),
                    'ct_path': ct,
                    'mri_path': mri,
                    'class_label': cls,
                    'stage': stage_val,
                    'survival_time': survival_val,
                    'censor': censor_val,
                    'age': age_val,
                    'sex': sex,
                    'notes': r.get('notes','')
                })

    def __len__(self):
        return len(self.rows)

    def _load_image(self, path: str):
        if not path:
            return None
        try:
            img = Image.open(path).convert('RGB')
            return img
        except Exception:
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        ct_img = None
        mri_img = None
        if r['ct_path']:
            try:
                ct_img = Image.open(r['ct_path']).convert('RGB')
            except Exception:
                ct_img = None
        if r['mri_path']:
            try:
                mri_img = Image.open(r['mri_path']).convert('RGB')
            except Exception:
                mri_img = None

        ct_tensor = self.transforms(ct_img) if ct_img is not None else None
        mri_tensor = self.transforms(mri_img) if mri_img is not None else None

        sample = {
            'patient_id': r['patient_id'],
            'ct': ct_tensor,
            'mri': mri_tensor,
            'class_label': r['class_label'],
            'stage': r['stage'],
            'survival_time': r['survival_time'],
            'censor': r['censor'],
            'age': r['age'],
            'sex': r['sex']
        }
        return sample
