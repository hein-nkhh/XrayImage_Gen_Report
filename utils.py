import torch

def collate_fn(batch):
    """
    Enhanced collate function for dual-view processing
    Handles both training and inference batches with dictionary items
    """
    # Kiểm tra kiểu dữ liệu đầu vào
    if isinstance(batch[0], dict):
        # Xử lý batch dạng dictionary
        front_images = [item['front'] for item in batch]
        lateral_images = [item['lateral'] for item in batch]
        
        # Stack riêng từng view
        front_batch = torch.stack(front_images, dim=0)
        lateral_batch = torch.stack(lateral_images, dim=0)
        
        # Trả về dictionary hoàn chỉnh
        return {
            'front': front_batch,
            'lateral': lateral_batch,
            'report': [item.get('report', "") for item in batch]  # Xử lý cả trường hợp không có report
        }
    
    elif isinstance(batch[0], tuple):
        # Fallback cho dữ liệu cũ (nếu cần)
        images, reports = zip(*batch)
        return torch.stack(images, dim=0), list(reports)
    
    else:
        # Xử lý inference mode đơn giản
        return torch.stack(batch, dim=0)