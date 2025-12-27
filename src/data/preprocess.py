"""Preprocessing helpers: registration, resampling, normalization (stubs).
"""
import numpy as np


def resample_volume(volume, new_spacing=(1.0, 1.0, 1.0)):
    # TODO: implement with SimpleITK or nibabel
    return volume


def normalize_volume(volume):
    v = volume.astype('float32')
    v -= v.mean()
    v /= (v.std() + 1e-8)
    return v
