import torch
import torch.nn as nn
from timm import create_model
from transformers import AutoModelForCausalLM, AutoTokenizer

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.projection(x)

class VisionEncoder(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window7_224', output_dim=1024):
        super().__init__()
        self.encoder = create_model(model_name, pretrained=True, num_classes=0, features_only=False)
        self.projection = ProjectionHead(self.encoder.num_features, output_dim)

    def forward(self, images):
        feats = self.encoder.forward_features(images)  # (B, H, W, C)
        B, H, W, C = feats.shape
        feats = feats.view(B, H*W, C)
        return self.projection(feats)

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_embeds, vision_embeds):
        attended, _ = self.attn(query=text_embeds, key=vision_embeds, value=vision_embeds)
        return self.dropout(attended + text_embeds)

class TextDecoder(nn.Module):
    def __init__(self, model_name='microsoft/biogpt'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.embedding_dim = self.model.get_input_embeddings().weight.shape[1]
        
        # Add special tokens if needed
        special_tokens = {'bos_token': '<BOS>', 'eos_token': '<EOS>'}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def encode_text(self, texts, max_length=153):
        """Encode text to input_ids and attention_mask"""
        if not isinstance(texts, list):
            texts = [texts]
            
        encoding = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
        
    def decode(self, token_ids):
        """Decode token IDs to text"""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None):
        """Forward pass of text decoder"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels
        )

    def generate(self, input_ids=None, attention_mask=None, inputs_embeds=None,
             max_length=150, max_new_tokens=None, **kwargs):
        """Generate text from inputs"""
        if inputs_embeds is None and input_ids is None:
            raise ValueError("Either inputs_embeds or input_ids must be provided")

        with torch.no_grad():
            # When using inputs_embeds, we need to make sure max_new_tokens is set properly
            if inputs_embeds is not None and input_ids is None:
                # Use max_new_tokens instead of max_length when using inputs_embeds
                if max_new_tokens is None:
                    max_new_tokens = max_length

                # Create a dummy input_ids tensor to help with generation
                batch_size = inputs_embeds.size(0)
                bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
                input_ids = torch.tensor([[bos_token_id]] * batch_size, device=inputs_embeds.device)
                
                # For inputs_embeds case, we'll use max_new_tokens instead of max_length
                output_ids = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    **kwargs
                )
            else:
                # For input_ids case, we can use max_length as originally intended
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    max_length=max_length,
                    **kwargs
                )
                
            return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

class XrayReportModel(nn.Module):
    def __init__(self, vision_encoder, text_decoder, cross_attention):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.cross_attention = cross_attention
    
    def forward(self, images, text=None, labels=None):
        """
        Forward pass for training
        
        Args:
            images: Image tensor of shape (B, C, H, W)
            text: List of report texts (for training)
            labels: Optional labels for computing loss
            
        Returns:
            Output from text decoder
        """
        device = images.device
        
        # Get vision embeddings
        vision_embeds = self.vision_encoder(images)
        
        if text is not None:
            # Training mode
            input_ids, attention_mask = self.text_decoder.encode_text(text)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Get text embeddings
            text_embeds = self.text_decoder.get_input_embeddings()(input_ids)
            
            # Combine with vision features via cross-attention
            fused = self.cross_attention(text_embeds, vision_embeds)
            
            # Forward pass through text decoder
            return self.text_decoder(
                inputs_embeds=fused,
                attention_mask=attention_mask,
                labels=labels if labels is not None else input_ids
            )
        
        else:
            # This branch should not be used in training
            # For inference, use generate() instead
            pass
        
    def generate_report(self, images, prompt_text=None, max_length=150, max_new_tokens=None, **kwargs):
        """
        Generate report from X-ray images
        
        Args:
            images: Image tensor of shape (B, C, H, W)
            prompt_text: Optional prompt text to start generation
            max_length: Maximum length of generated text
            max_new_tokens: Maximum number of new tokens to generate (alternative to max_length)
            
        Returns:
            List of generated reports
        """
        device = images.device
        batch_size = images.shape[0]
        
        # Get vision embeddings
        vision_embeds = self.vision_encoder(images)
        
        # Create initial prompt if not provided
        if prompt_text is None:
            prompt_text = [""] * batch_size
        elif isinstance(prompt_text, str):
            prompt_text = [prompt_text] * batch_size
            
        # Encode prompt text
        input_ids, attention_mask = self.text_decoder.encode_text(prompt_text)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Get text embeddings from the prompt
        text_embeds = self.text_decoder.get_input_embeddings()(input_ids)
        
        # Combine with vision features via cross-attention
        fused = self.cross_attention(text_embeds, vision_embeds)
        
        # Generate text with fused embeddings
        return self.text_decoder.generate(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            **kwargs
        )