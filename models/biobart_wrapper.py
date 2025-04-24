from transformers import BartForConditionalGeneration, BartTokenizer

class BioBARTWrapper:
    def __init__(self, model_name: str, device):
        self.device = device
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

    def tokenize_reports(self, texts, max_length=150):
        return self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )["input_ids"]

    def decode(self, token_ids):
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    def generate(self, image_embeddings, max_length=100):
        inputs = image_embeddings.unsqueeze(1)  # (batch_size, 1, hidden_size)
        return self.model.generate(
            inputs_embeds=inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

    def get_loss(self, embeddings, labels):
        batch_size, seq_len = labels.size()
        decoder_inputs_embeds = embeddings.unsqueeze(1).expand(-1, seq_len, -1)
        attention_mask = (labels != self.tokenizer.pad_token_id).long().to(self.device)
        outputs = self.model(
            inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            decoder_attention_mask=attention_mask
        )
        return outputs.loss
