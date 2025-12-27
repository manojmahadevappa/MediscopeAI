"""Inference helpers and CLI for model prediction.

This module provides `run_inference` which routes inputs to the correct
model (CT-only, MRI-only, or fusion) and returns a structured result.

Assumes the following globals exist in the runtime environment:
  - `ct_model`, `mri_model`, `fusion_model`
  - `CT_CLASS_LABELS` and `MRI_CLASS_LABELS` (dicts mapping int->str)

The tensors passed to `run_inference` are assumed to be preprocessed and
already on the correct device.
"""
from typing import Optional, Dict, Any
import argparse
import torch


def run_inference(ct_tensor: Optional[torch.Tensor] = None,
                  mri_tensor: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """Route inputs to the correct model and return prediction details.

    Returns a dict with keys:
      - "modality_used": "ct", "mri", or "fusion"
      - "num_classes": int
      - "predicted_class_id": int
      - "predicted_class_label": str
      - "probabilities": list[float]

    Raises ValueError if both inputs are None. Raises RuntimeError if the
    required model or label mapping is not available.
    """
    # Validate inputs
    if ct_tensor is None and mri_tensor is None:
        raise ValueError('At least one of ct_tensor or mri_tensor must be provided')

    # Determine modality
    if ct_tensor is not None and mri_tensor is None:
        modality = 'ct'
    elif ct_tensor is None and mri_tensor is not None:
        modality = 'mri'
    else:
        modality = 'fusion'

    # Access globals (expected to be defined by the application)
    global ct_model, mri_model, fusion_model
    global CT_CLASS_LABELS, MRI_CLASS_LABELS

    # Sanity checks for models and label maps
    if modality == 'ct':
        if 'ct_model' not in globals() or ct_model is None:
            raise RuntimeError('CT model is not loaded (ct_model is None)')
        model = ct_model
        label_map = globals().get('CT_CLASS_LABELS')
        if not isinstance(label_map, dict):
            raise RuntimeError('CT_CLASS_LABELS mapping is not available')
        num_classes = 2
    elif modality == 'mri':
        if 'mri_model' not in globals() or mri_model is None:
            raise RuntimeError('MRI model is not loaded (mri_model is None)')
        model = mri_model
        label_map = globals().get('MRI_CLASS_LABELS')
        if not isinstance(label_map, dict):
            raise RuntimeError('MRI_CLASS_LABELS mapping is not available')
        num_classes = len(label_map)
    else:  # fusion
        if 'fusion_model' not in globals() or fusion_model is None:
            raise RuntimeError('Fusion model is not loaded (fusion_model is None)')
        model = fusion_model
        label_map = globals().get('MRI_CLASS_LABELS')
        if not isinstance(label_map, dict):
            raise RuntimeError('MRI_CLASS_LABELS mapping is not available for fusion')
        num_classes = len(label_map)

    # Run inference
    model.eval()
    with torch.no_grad():
        if modality == 'ct':
            logits = model(ct_tensor)
        elif modality == 'mri':
            logits = model(mri_tensor)
        else:
            # fusion expects both tensors
            logits = model(ct_tensor, mri_tensor)

        if not isinstance(logits, torch.Tensor):
            raise RuntimeError('Model did not return a torch.Tensor for logits')

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # Validate logits shape
        if logits.size(1) != num_classes:
            # Allow CT logits to be size 2 even if num_classes differs only for MRI
            if modality == 'ct' and logits.size(1) == 2:
                pass
            else:
                raise RuntimeError(f'Logits have shape {list(logits.shape)} but expected second dim={num_classes}')

        probs = torch.softmax(logits, dim=1)
        top_idx = int(torch.argmax(probs, dim=1).item())
        probs_list = [float(x) for x in probs[0].tolist()]

        # Map predicted index to label
        predicted_label = label_map.get(top_idx, str(top_idx))

    return {
        'modality_used': modality,
        'num_classes': num_classes,
        'predicted_class_id': top_idx,
        'predicted_class_label': predicted_label,
        'probabilities': probs_list,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to model')
    parser.add_argument('--input', type=str, help='Input image path')
    args = parser.parse_args()
    print('This file exposes `run_inference(ct_tensor, mri_tensor)` for programmatic use')


if __name__ == '__main__':
    main()
