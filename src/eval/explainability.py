"""Explainability helpers: Grad-CAM implementation and SHAP placeholder.

The `grad_cam` function returns a 2D heatmap (numpy float32, 0..1)
corresponding to the given input tensor (shape [1,C,H,W]).
"""
from typing import Optional
import numpy as np
import torch


def grad_cam(model: torch.nn.Module, input_tensor: torch.Tensor, target_class: Optional[int] = None, target_layer: str = 'layer4', input_key: Optional[str] = None) -> np.ndarray:
    """Compute Grad-CAM for `model` and `input_tensor`.

    Args:
      model: torch model (e.g., resnet18)
      input_tensor: tensor with shape [1,C,H,W]
      target_class: class index to compute gradients for. If None, uses predicted class.
      target_layer: name of module to use for activations (default 'layer4' for ResNet)

    Returns:
      heatmap: numpy array shape (H,W) with values in [0,1]
    """
    model.eval()

    activations = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    # find target module (support dotted names like 'mri_encoder.layer4')
    target_module = dict(model.named_modules()).get(target_layer, None)
    if target_module is None:
        raise ValueError(f'Target layer "{target_layer}" not found in model')

    fh = target_module.register_forward_hook(forward_hook)
    bh = target_module.register_backward_hook(backward_hook)

    device = next(model.parameters()).device
    inp = input_tensor.to(device)

    # Forward: support models that expect named inputs (e.g., MultiModalNet(ct=..., mri=...)).
    if input_key is None:
        out = model(inp)
    else:
        if input_key == 'mri':
            out = model(ct=None, mri=inp)
        elif input_key == 'ct':
            out = model(ct=inp, mri=None)
        else:
            out = model(inp)

    # Models may return a dict with a 'logits' entry (multimodal model). Accept both.
    logits = out['logits'] if isinstance(out, dict) and 'logits' in out else out
    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    score = logits[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=False)

    fh.remove()
    bh.remove()

    if activations is None or gradients is None:
        raise RuntimeError('Failed to obtain activations or gradients for Grad-CAM')

    # activations: [1, C, H, W], gradients: [1, C, H, W]
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1,C,1,1]
    cam = (weights * activations).sum(dim=1, keepdim=True)  # [1,1,H,W]
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    # normalize
    cam -= cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    else:
        cam = np.zeros_like(cam)

    return cam.astype(np.float32)


def shap_stub(model, background, inputs):
    """Return SHAP values (placeholder). Use `shap` package in real implementation."""
    return None
