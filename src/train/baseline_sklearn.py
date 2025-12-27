"""Scikit-learn LogisticRegression baseline for the MRI dataset.

Usage:
  python -m src.train.baseline_sklearn --data-root Dataset
"""
from pathlib import Path
from PIL import Image
import numpy as np
import argparse


def load_flat_images(root: Path, size=(64, 64)):
    root = root / 'Brain Tumor MRI images'
    paths = []
    labels = []
    for label_name, label in [('Healthy', 0), ('Tumor', 1)]:
        folder = root / label_name
        for p in folder.iterdir():
            if p.is_file():
                paths.append(p)
                labels.append(label)
    X = []
    y = []
    for p, lab in zip(paths, labels):
        img = Image.open(p).convert('L').resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        X.append(arr.ravel())
        y.append(lab)
    return np.stack(X), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='Dataset')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    X, y = load_flat_images(Path(args.data_root))
    print('Loaded', X.shape, y.shape)

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)
    print('Train/test sizes:', X_train.shape[0], X_test.shape[0])

    clf = LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f'LogisticRegression test accuracy: {acc:.4f}')
    print('\nClassification report:\n')
    print(classification_report(y_test, pred, digits=4))


if __name__ == '__main__':
    main()
