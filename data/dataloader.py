import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from config import MLP_INPUT_DIM, MAX_SEQ_LEN, BATCH_SIZE, BIOBART_MODEL_NAME

biobart_tokenizer = AutoTokenizer.from_pretrained(BIOBART_MODEL_NAME)

def normalize_features(arr):
    mean, std = arr.mean(), arr.std()
    std = std if std > 1e-6 else 1.0
    return (arr - mean) / std

def to_tensor(arr):
    return torch.tensor(arr.copy()).float() if isinstance(arr, np.ndarray) else torch.empty((0, MLP_INPUT_DIM))

def tokenize_reports(texts):
    return biobart_tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding='max_length',
        return_tensors='pt'
    )["input_ids"]

def get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test):
    X_train, X_val, X_test = map(normalize_features, [X_train, X_val, X_test])
    X_train_tensor, X_val_tensor, X_test_tensor = map(to_tensor, [X_train, X_val, X_test])

    y_train_tensor = tokenize_reports(y_train)
    y_val_tensor = tokenize_reports(y_val)
    y_test_tensor = tokenize_reports(y_test)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    val_ds = TensorDataset(X_val_tensor, y_val_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE),
        DataLoader(test_ds, batch_size=BATCH_SIZE)
    )
