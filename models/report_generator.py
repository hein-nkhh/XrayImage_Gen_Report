import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from config import BIOBART_MODEL_NAME, DEVICE

class ReportGenerator:
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained(BIOBART_MODEL_NAME)
        self.model = BartForConditionalGeneration.from_pretrained(BIOBART_MODEL_NAME).to(DEVICE)

    def generate(self, embeddings, max_length=150):
        # embeddings shape: (batch, hidden) → expand to (batch, 1, hidden)
        inputs = embeddings.unsqueeze(1)
        output_ids = self.model.generate(
            inputs_embeds=inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def get_loss(self, embeddings, labels, attention_mask):
        batch_size, seq_len = labels.shape
        inputs_embeds = embeddings.unsqueeze(1).expand(-1, seq_len, -1)
        output = self.model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            decoder_attention_mask=attention_mask,
        )
        return output.loss
    
