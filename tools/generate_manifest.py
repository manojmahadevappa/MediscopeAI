#!/usr/bin/env python3
"""
Best-effort manifest generator for the Brain Tumer dataset.

What it does:
- Walks `Dataset/` for CT and MRI image files (png/jpg/jpeg)
- Attempts to pair CT and MRI images by a shared patient token extracted from filenames
- Applies simple heuristics to infer `class_label` from folder names or filenames when present
- Writes `manifest.csv` at project root with columns:
  patient_id, ct_path, mri_path, class_label, stage, survival_time, censor, age, sex, notes

Notes:
- This script produces a best-effort CSV with placeholders for missing labels.
- Manual review or clinical annotation is required for accurate class/stage/survival labels.
"""
import csv
import os
from pathlib import Path
import re
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'Dataset'
OUT_CSV = ROOT / 'manifest.csv'

IMAGE_EXTS = {'.png', '.jpg', '.jpeg'}


def find_images():
    ct_files = []
    mri_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in IMAGE_EXTS:
                # classify by parent folder names
                parent = str(p.parent).lower()
                if 'ct' in parent or 'ct scan' in parent or 'ct_' in f.lower() or 'ct' in f.lower():
                    ct_files.append(p)
                elif 'mri' in parent or 'mr' in parent or 't1' in f.lower() or 't2' in f.lower() or 'flair' in f.lower():
                    mri_files.append(p)
                else:
                    # ambiguous: put into MRI list if folder contains 'tumor' but not 'healthy'
                    if 'tumor' in parent and 'ct' not in parent:
                        mri_files.append(p)
                    elif 'ct' in parent:
                        ct_files.append(p)
                    else:
                        # fallback: if folder path contains 'CT' earlier in tree
                        if any('ct' in part.lower() for part in Path(root).parts):
                            ct_files.append(p)
                        else:
                            mri_files.append(p)
    return ct_files, mri_files


def patient_token(filename: str):
    """Extract a patient token from filename by removing common separators and trailing indices.
    Examples:
      - 'patient123_ct.png' -> 'patient123'
      - '12345_CT_1.png' -> '12345'
    """
    name = Path(filename).stem
    # split on non-alphanumeric and take first token that has digits or letters
    parts = re.split(r'[^A-Za-z0-9]+', name)
    # prefer tokens containing digits
    for p in parts:
        if any(ch.isdigit() for ch in p):
            return p.lower()
    # otherwise return first non-empty token
    for p in parts:
        if p:
            return p.lower()
    return name.lower()


def infer_label_from_path(p: Path):
    s = str(p).lower()
    # look for explicit keywords
    if 'benign' in s:
        return 'benign'
    if 'low' in s and 'grade' in s:
        return 'low-grade'
    if 'high' in s and 'grade' in s:
        return 'high-grade'
    # some MRI folders might be named with tumor type
    if 'tumor' in s and 'healthy' not in s:
        # unknown tumor type -> mark as unknown (needs review)
        return 'unknown'
    if 'healthy' in s or 'normal' in s:
        return 'healthy'
    return ''


def build_manifest():
    ct_files, mri_files = find_images()
    by_token = defaultdict(lambda: {'ct': [], 'mri': []})

    for p in ct_files:
        tok = patient_token(p.name)
        by_token[tok]['ct'].append(p)
    for p in mri_files:
        tok = patient_token(p.name)
        by_token[tok]['mri'].append(p)

    rows = []
    used = set()
    # First, create rows for tokens that have at least one modality
    for tok, d in by_token.items():
        ct = d['ct'][0] if d['ct'] else ''
        mri = d['mri'][0] if d['mri'] else ''
        # try to infer class label from MRI first, else CT, else blank
        lbl = ''
        if mri:
            lbl = infer_label_from_path(Path(mri))
        if not lbl and ct:
            lbl = infer_label_from_path(Path(ct))
        # canonicalize unknown/empty
        if lbl == '':
            lbl = 'unknown'
        # generate patient id
        patient_id = tok
        rows.append({
            'patient_id': patient_id,
            'ct_path': str(ct) if ct else '',
            'mri_path': str(mri) if mri else '',
            'class_label': lbl,
            'stage': '',
            'survival_time': '',
            'censor': '',
            'age': '',
            'sex': '',
            'notes': ''
        })
        if ct:
            used.add(str(ct))
        if mri:
            used.add(str(mri))

    # For any CT/MRI files not assigned (rare), add them as singleton rows
    all_files = set(str(p) for p in (ct_files + mri_files))
    leftover = all_files - used
    for pstr in leftover:
        p = Path(pstr)
        tok = patient_token(p.name)
        lbl = infer_label_from_path(p)
        rows.append({
            'patient_id': tok + '_unpaired',
            'ct_path': str(p) if 'ct' in str(p).lower() else '',
            'mri_path': str(p) if 'mri' in str(p).lower() else '',
            'class_label': lbl or 'unknown',
            'stage': '',
            'survival_time': '',
            'censor': '',
            'age': '',
            'sex': '',
            'notes': 'unpaired-file'
        })

    # write CSV
    fieldnames = ['patient_id','ct_path','mri_path','class_label','stage','survival_time','censor','age','sex','notes']
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f'Wrote manifest with {len(rows)} rows to: {OUT_CSV}')


if __name__ == '__main__':
    print('Scanning dataset under:', DATA_DIR)
    build_manifest()
