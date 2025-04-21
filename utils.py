import torch

def collate_fn(batch):
    """
    Custom collate function that handles both training and inference batches
    """
    if isinstance(batch[0], tuple):
        # Training mode: (image, report) pairs
        images, reports = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, list(reports)
    else:
        # Inference mode: only images
        images = torch.stack(batch, dim=0)
        return images