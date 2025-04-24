import numpy as np
import torch
from torch.utils.data import TensorDataset


def normalize_features(arr):
    if isinstance(arr, np.ndarray) and arr.size > 0:
        std = arr.std() if arr.std() > 1e-6 else 1.0
        return (arr - arr.mean()) / std
    return np.empty((0, arr.shape[1]))


def to_tensor(arr):
    return torch.tensor(arr.copy()).float() if isinstance(arr, np.ndarray) else torch.empty(0)


def build_dataset(features, labels):
    if features.shape[0] > 0 and labels.shape[0] > 0:
        return TensorDataset(features, labels)
    return None
