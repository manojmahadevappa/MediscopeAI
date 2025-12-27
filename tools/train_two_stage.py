"""Convenience script to train the two-stage pipeline.

Steps performed:
  1. Train binary classifier using `src.train.train_basic` (Healthy vs Tumor)
  2. Copy `model_basic.pth` -> `model_binary.pth` so the web UI loads it
  3. Generate `manifest.csv` for multiclass training (if not present)
  4. Train multiclass classifier using `src.train.train_multimodal` and save
     the checkpoint to `model_multiclass.pth` (web UI loads this filename)

This script shells out to the repository training scripts and therefore
expects you to run it in the repo root with your Python environment active.

Usage (PowerShell):
  env\Scripts\Activate.ps1; python tools\train_two_stage.py
"""
import argparse
import subprocess
from pathlib import Path
import shutil
import sys


def run_cmd(cmd, check=True):
    print('\n>> Running:', ' '.join(cmd))
    res = subprocess.run(cmd, shell=False)
    if check and res.returncode != 0:
        print('Command failed with exit code', res.returncode)
        sys.exit(res.returncode)
    return res.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, default='Dataset', help='Dataset root')
    p.add_argument('--manifest', type=str, default='manifest.csv', help='Manifest path for multiclass training')
    p.add_argument('--mapping', type=str, default=None, help='Optional JSON mapping for manifest generator')
    p.add_argument('--epochs-bin', type=int, default=5)
    p.add_argument('--epochs-multi', type=int, default=5)
    p.add_argument('--batch-size-bin', type=int, default=32)
    p.add_argument('--batch-size-multi', type=int, default=8)
    args = p.parse_args()

    root = Path(args.data_root)
    manifest = Path(args.manifest)

    # 1) Train binary (CT-only)
    cmd_bin = [sys.executable, '-m', 'src.train.train_basic', '--modality', 'ct', '--data-root', str(root), '--epochs', str(args.epochs_bin), '--batch-size', str(args.batch_size_bin)]
    run_cmd(cmd_bin)

    # 2) copy model_basic.pth -> model_binary.pth if present
    basic = Path('model_basic.pth')
    bin_dst = Path('model_binary.pth')
    if basic.exists():
        shutil.copy(str(basic), str(bin_dst))
        print(f'Copied {basic} -> {bin_dst}')
    else:
        print('Warning: model_basic.pth not found after binary training; webapp may not find binary model')

    # 3) generate manifest if not exists
    if not manifest.exists():
        gen_cmd = [sys.executable, 'tools\generate_manifest_mri.py', '--root', str(root), '--out', str(manifest)]
        if args.mapping:
            gen_cmd += ['--map', args.mapping]
        run_cmd(gen_cmd)
    else:
        print(f'Manifest {manifest} already exists; skipping generation')

    # 4) Train multiclass
    cmd_multi = [sys.executable, '-m', 'src.train.train_multimodal', '--manifest', str(manifest), '--root', str(root), '--epochs', str(args.epochs_multi), '--batch-size', str(args.batch_size_multi), '--output', 'model_multiclass.pth']
    run_cmd(cmd_multi)

    print('\nTwo-stage training complete. Models saved as:')
    print(' - model_binary.pth (binary detector)')
    print(' - model_multiclass.pth (stage2 multiclass grader)')
    print('\nYou can run the web UI: python webapp\app.py')


if __name__ == '__main__':
    main()
