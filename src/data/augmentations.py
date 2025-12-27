"""Augmentation helpers (stubs).
"""
import numpy as np


def random_flip(image, p=0.5):
    if np.random.rand() < p:
        return np.flip(image, axis=1).copy()
    return image


def random_noise(image, sigma=0.01):
    return image + np.random.normal(scale=sigma, size=image.shape)
