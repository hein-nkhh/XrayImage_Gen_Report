# inference.py

import torch

def run_inference(model, dataloader, device):
    model.eval()
    predictions, references = [], []

    with torch.no_grad():
        for batch in dataloader:
            front = batch['front'].to(device)
            lateral = batch['lateral'].to(device)
            report = batch['report']  # list of strings

            # Generate reports
            generated_ids = model.generate(
                front,
                lateral,
                max_length=150,              # Kích thước tối đa của văn bản sinh ra
                num_beams=5,                 # Sử dụng beam search để tạo ra câu đa dạng hơn
                repetition_penalty=2.0,      # Phạt các từ lặp lại
                length_penalty=1.0,          # Điều chỉnh độ dài của câu
                no_repeat_ngram_size=2,      # Ngăn không cho n-gram lặp lại
                early_stopping=True,         # Dừng khi đạt được câu tốt nhất
                do_sample=True,              # Bật sampling để mô hình sinh ra các câu ngẫu nhiên
                top_k=50,                    # Lựa chọn ngẫu nhiên từ top-k token có xác suất cao
                top_p=0.95                   # Lựa chọn từ trong top-p (cumulative probability)
            )
            generated_texts = model.decode(generated_ids)

            # Save
            predictions.extend(generated_texts)
            references.extend(report)

    return predictions, references
