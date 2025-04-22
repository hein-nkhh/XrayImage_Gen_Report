import torch
import torch.nn as nn
from timm import create_model
from transformers import BioGptForCausalLM, BioGptTokenizer, BioGptConfig

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.projection(x)

class DualViewEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.front_encoder = create_model(config.vision_encoder_name, pretrained=True, num_classes=0)
        self.lateral_encoder = create_model(config.vision_encoder_name, pretrained=True, num_classes=0)

        # Freeze Swin encoders
        for param in self.front_encoder.parameters():
            param.requires_grad = False
        for param in self.lateral_encoder.parameters():
            param.requires_grad = False

        self.front_proj = ProjectionHead(self.front_encoder.num_features, config.vision_hidden_size)
        self.lateral_proj = ProjectionHead(self.lateral_encoder.num_features, config.vision_hidden_size)

        self.cross_attn = nn.MultiheadAttention(embed_dim=config.vision_hidden_size, num_heads=8, batch_first=True)
        self.fusion_proj = nn.Linear(config.vision_hidden_size * 2, config.vision_hidden_size)

    def forward(self, front, lateral):
        front_feat = self.front_encoder.forward_features(front)
        lateral_feat = self.lateral_encoder.forward_features(lateral)

        front_feat = front_feat.view(front_feat.size(0), -1, front_feat.size(-1))
        lateral_feat = lateral_feat.view(lateral_feat.size(0), -1, lateral_feat.size(-1))

        front_proj = self.front_proj(front_feat)
        lateral_proj = self.lateral_proj(lateral_feat)

        front_to_lateral, _ = self.cross_attn(front_proj, lateral_proj, lateral_proj)
        lateral_to_front, _ = self.cross_attn(lateral_proj, front_proj, front_proj)

        fused_feat = torch.cat([front_to_lateral, lateral_to_front], dim=-1)
        return self.fusion_proj(fused_feat)

class BioGPTDecoder(nn.Module):
    def __init__(self, model_name='microsoft/biogpt'):
        super().__init__()
        self.config = BioGptConfig.from_pretrained(model_name)
        self.model = BioGptForCausalLM.from_pretrained(model_name)
        self.tokenizer = BioGptTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def get_embedding_layer(self):
        return self.model.biogpt.embed_tokens

    def encode_text(self, texts, max_length=None):
        return self.tokenizer(
            texts, padding='max_length' if max_length else True,
            truncation=True, max_length=max_length, return_tensors='pt'
        )

class ImageTextFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_embeds, vision_embeds):
        attn_output, _ = self.cross_attn(text_embeds, vision_embeds, vision_embeds)
        return self.norm(text_embeds + self.dropout(attn_output))

class XrayReportModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_encoder = DualViewEncoder(config)
        self.biogpt = BioGPTDecoder()
        self.text_embed = self.biogpt.get_embedding_layer()

        self.vision_proj = nn.Linear(config.vision_dim, self.biogpt.config.hidden_size)
        self.fusion_layers = nn.ModuleList([
            ImageTextFusion(embed_dim=self.biogpt.config.hidden_size, num_heads=8) for _ in range(2)
        ])

    def forward(self, front, lateral, report, labels=None):
        vision_embeds = self.vision_encoder(front, lateral)
        vision_embeds = self.vision_proj(vision_embeds)

        encoding = self.biogpt.encode_text(report, max_length=self.config.max_len)
        input_ids = encoding['input_ids'].to(front.device)
        attention_mask = encoding['attention_mask'].to(front.device)

        text_embeds = self.text_embed(input_ids)
        for fusion_layer in self.fusion_layers:
            text_embeds = fusion_layer(text_embeds, vision_embeds)

        outputs = self.biogpt.model(inputs_embeds=text_embeds, attention_mask=attention_mask, labels=labels)
        return outputs
