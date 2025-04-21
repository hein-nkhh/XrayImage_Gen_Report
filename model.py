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

class DualViewEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Base encoder cho từng view
        self.front_encoder = create_model(config.vision_encoder_name, 
                                        pretrained=True, 
                                        num_classes=0)
        self.lateral_encoder = create_model(config.vision_encoder_name, 
                                        pretrained=True, 
                                        num_classes=0)
        
        # Projection layers
        self.front_proj = ProjectionHead(self.front_encoder.num_features, 
                                        config.vision_output_dim)
        self.lateral_proj = ProjectionHead(self.lateral_encoder.num_features,
                                        config.vision_output_dim)
        
        # Cross-view attention
        self.cross_attn = EnhancedCrossAttention(hidden_dim=config.cross_attn_dim,
                                    num_heads=config.cross_attn_heads)
    
    def forward(self, front, lateral):
        # Encode features
        front_feat = self.front_encoder.forward_features(front)
        lateral_feat = self.lateral_encoder.forward_features(lateral)
        
        # Projection
        front_proj = self.front_proj(front_feat)
        lateral_proj = self.lateral_proj(lateral_feat)
        
        # Cross-view attention
        fused_feat = self.cross_attn(front_proj, lateral_proj)
        return fused_feat

class EnhancedCrossAttention(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=8):
        super().__init__()
        self.front_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.lateral_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, front, lateral):
        # Attention giữa các view
        attn_front, _ = self.front_attn(front, lateral, lateral)
        attn_lateral, _ = self.lateral_attn(lateral, front, front)
        
        # Kết hợp features
        combined = torch.cat([attn_front, attn_lateral], dim=1)
        
        # Feed forward
        fused = self.norm(combined + self.ffn(combined))
        return fused

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class XrayReportModel(nn.Module):
    def __init__(self, vision_encoder, text_decoder, cross_attention, config):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.cross_attention = cross_attention
        self.config = config
        
        # Vision to text projection
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_encoder.output_dim, text_decoder.model.config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(text_decoder.model.config.hidden_size)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(text_decoder.model.config.hidden_size)
        
        # Regularization
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layer_norm = nn.LayerNorm(text_decoder.model.config.hidden_size)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, front_images, lateral_images, text=None, labels=None):
        """
        Enhanced forward pass with dual-view processing
        """
        device = front_images.device
        
        # Encode vision features
        vision_embeds = self.vision_encoder(front_images, lateral_images)
        vision_embeds = self.vision_proj(vision_embeds)
        
        # Add positional encoding
        vision_embeds = vision_embeds + self.pos_encoder(vision_embeds)
        vision_embeds = self.layer_norm(vision_embeds)
        vision_embeds = self.dropout(vision_embeds)

        if text is None:
            raise ValueError("Text input is required for training mode")
            
        # Process text inputs
        input_ids, attention_mask = self.text_decoder.encode_text(text)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Get text embeddings
        text_embeds = self.text_decoder.get_input_embeddings()(input_ids)
        text_embeds = text_embeds + self.pos_encoder(text_embeds)
        
        # Multimodal fusion
        fused = self.cross_attention(
            query=text_embeds,
            key=vision_embeds,
            value=vision_embeds,
            key_padding_mask=None
        )
        fused = self.layer_norm(fused + text_embeds)  # Residual connection
        
        return self.text_decoder(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            labels=labels if labels is not None else input_ids
        )

    def generate_report(self, front_images, lateral_images, prompt_text=None, **kwargs):
        """
        Enhanced generation with medical prompts
        """
        device = front_images.device
        batch_size = front_images.shape[0]
        
        # Encode vision features
        vision_embeds = self.vision_encoder(front_images, lateral_images)
        vision_embeds = self.vision_proj(vision_embeds)
        vision_embeds = vision_embeds + self.pos_encoder(vision_embeds)
        
        # Create medical prompts
        if prompt_text is None:
            prompt_text = ["Findings: Impression:"] * batch_size
            
        # Encode prompts
        input_ids, attention_mask = self.text_decoder.encode_text(prompt_text)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Get prompt embeddings
        text_embeds = self.text_decoder.get_input_embeddings()(input_ids)
        text_embeds = text_embeds + self.pos_encoder(text_embeds)
        
        # Multimodal fusion
        fused = self.cross_attention(
            query=text_embeds,
            key=vision_embeds,
            value=vision_embeds,
            key_padding_mask=None
        )
        fused = self.layer_norm(fused + text_embeds)  # Residual connection
        
        # Generation parameters
        gen_kwargs = {
            'inputs_embeds': fused,
            'attention_mask': attention_mask,
            'max_length': self.config.max_gen_length,
            'num_beams': self.config.num_beams,
            'repetition_penalty': self.config.repetition_penalty,
            'length_penalty': self.config.length_penalty,
            'early_stopping': True,
            'pad_token_id': self.text_decoder.tokenizer.eos_token_id
        }
        gen_kwargs.update(kwargs)
        
        generated_ids = self.text_decoder.model.generate(**gen_kwargs)
        return self.text_decoder.decode(generated_ids)