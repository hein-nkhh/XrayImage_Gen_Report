import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from config import MAX_SEQ_LEN, BATCH_SIZE, BIOBART_MODEL_NAME


def normalize_features(arr):
    mean, std = arr.mean(), arr.std()
    std = std if std > 1e-6 else 1.0
    return (arr - mean) / std

def to_tensor(arr, input_dim):
    return torch.tensor(arr.copy()).float() if isinstance(arr, np.ndarray) else torch.empty((0, input_dim))

def tokenize_reports(texts, tokenizer):
    return tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding='max_length',
        return_tensors='pt'
    )["input_ids"]

def get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, input_dim, tokenizer):
    X_train, X_val, X_test = map(normalize_features, [X_train, X_val, X_test])
    X_train_tensor = to_tensor(X_train, input_dim)
    X_val_tensor = to_tensor(X_val, input_dim)
    X_test_tensor = to_tensor(X_test, input_dim)

    y_train_tensor = tokenize_reports(y_train, tokenizer)
    y_val_tensor = tokenize_reports(y_val, tokenizer)
    y_test_tensor = tokenize_reports(y_test, tokenizer)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    val_ds = TensorDataset(X_val_tensor, y_val_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE),
        DataLoader(test_ds, batch_size=BATCH_SIZE)
    )
