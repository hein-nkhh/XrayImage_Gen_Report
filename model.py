# import torch
# import torch.nn as nn
# from timm import create_model
# from transformers import BioGptForCausalLM, BioGptTokenizer, BioGptConfig

# class ProjectionHead(nn.Module):
#     def __init__(self, input_dim, output_dim, dropout=0.1):
#         super().__init__()
#         self.projection = nn.Sequential(
#             nn.Linear(input_dim, output_dim),
#             nn.GELU(),
#             nn.LayerNorm(output_dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         return self.projection(x)

# class DualViewEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.front_encoder = create_model(config.vision_encoder_name, pretrained=True, num_classes=0)
#         self.lateral_encoder = create_model(config.vision_encoder_name, pretrained=True, num_classes=0)

#         # Freeze Swin encoders
#         for param in self.front_encoder.parameters():
#             param.requires_grad = False
#         for param in self.lateral_encoder.parameters():
#             param.requires_grad = False

#         self.front_proj = ProjectionHead(self.front_encoder.num_features, config.vision_hidden_size)
#         self.lateral_proj = ProjectionHead(self.lateral_encoder.num_features, config.vision_hidden_size)

#         self.cross_attn = nn.MultiheadAttention(embed_dim=config.vision_hidden_size, num_heads=8, batch_first=True)
#         self.fusion_proj = nn.Linear(config.vision_hidden_size * 2, config.vision_hidden_size)

#     def forward(self, front, lateral):
#         front_feat = self.front_encoder.forward_features(front)
#         lateral_feat = self.lateral_encoder.forward_features(lateral)

#         front_feat = front_feat.view(front_feat.size(0), -1, front_feat.size(-1))
#         lateral_feat = lateral_feat.view(lateral_feat.size(0), -1, lateral_feat.size(-1))

#         front_proj = self.front_proj(front_feat)
#         lateral_proj = self.lateral_proj(lateral_feat)

#         front_to_lateral, _ = self.cross_attn(front_proj, lateral_proj, lateral_proj)
#         lateral_to_front, _ = self.cross_attn(lateral_proj, front_proj, front_proj)

#         fused_feat = torch.cat([front_to_lateral, lateral_to_front], dim=-1)
#         return self.fusion_proj(fused_feat)

# class BioGPTDecoder(nn.Module):
#     def __init__(self, model_name='microsoft/biogpt'):
#         super().__init__()
#         self.config = BioGptConfig.from_pretrained(model_name)
#         self.model = BioGptForCausalLM.from_pretrained(model_name)
#         self.tokenizer = BioGptTokenizer.from_pretrained(model_name)
#         self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#         self.model.resize_token_embeddings(len(self.tokenizer))

#     def get_embedding_layer(self):
#         return self.model.biogpt.embed_tokens

#     def encode_text(self, texts, max_length=None):
#         return self.tokenizer(
#             texts, padding='max_length' if max_length else True,
#             truncation=True, max_length=max_length, return_tensors='pt'
#         )

# class ImageTextFusion(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
#         self.norm = nn.LayerNorm(embed_dim)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, text_embeds, vision_embeds):
#         attn_output, _ = self.cross_attn(text_embeds, vision_embeds, vision_embeds)
#         return self.norm(text_embeds + self.dropout(attn_output))

# class XrayReportModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.vision_encoder = DualViewEncoder(config)
#         self.biogpt = BioGPTDecoder()
#         self.text_embed = self.biogpt.get_embedding_layer()

#         self.vision_proj = nn.Linear(config.vision_dim, self.biogpt.config.hidden_size)
#         self.fusion_layers = nn.ModuleList([
#             ImageTextFusion(embed_dim=self.biogpt.config.hidden_size, num_heads=8) for _ in range(2)
#         ])

#     def forward(self, front, lateral, report, labels=None):
#         vision_embeds = self.vision_encoder(front, lateral)
#         vision_embeds = self.vision_proj(vision_embeds)

#         encoding = self.biogpt.encode_text(report, max_length=self.config.max_len)
#         input_ids = encoding['input_ids'].to(front.device)
#         attention_mask = encoding['attention_mask'].to(front.device)

#         text_embeds = self.text_embed(input_ids)
#         for fusion_layer in self.fusion_layers:
#             text_embeds = fusion_layer(text_embeds, vision_embeds)

#         outputs = self.biogpt.model(inputs_embeds=text_embeds, attention_mask=attention_mask, labels=labels)
#         return outputs
    
#     def generate(self, front, lateral, max_length=150):
#         vision_embeds = self.vision_encoder(front, lateral)
#         vision_embeds = self.vision_proj(vision_embeds)

#         batch_size = front.size(0)
#         device = front.device
#         bos_token_id = self.biogpt.tokenizer.bos_token_id
#         eos_token_id = self.biogpt.tokenizer.eos_token_id

#         past = None
#         cur_input = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
#         generated = []

#         for _ in range(max_length):
#             embed = self.text_embed(cur_input).squeeze(1).unsqueeze(1)

#             for fusion_layer in self.fusion_layers:
#                 embed = fusion_layer(embed, vision_embeds)

#             outputs = self.biogpt.model(inputs_embeds=embed, past_key_values=past, use_cache=True)
#             logits = outputs.logits
#             past = outputs.past_key_values

#             next_token = logits.argmax(dim=-1)[:, -1].unsqueeze(1)
#             generated.append(next_token)
#             cur_input = next_token

#             if (next_token == eos_token_id).all():
#                 break

#         return torch.cat(generated, dim=1)

#     def decode(self, generated_ids):
#         return self.biogpt.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from transformers import BartTokenizer, BartForConditionalGeneration

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

class CLIPVisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Load CLIP ViT model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # use projection_dim from CLIP config
        input_dim = self.clip_model.config.projection_dim
        self.proj_head = ProjectionHead(input_dim, config.vision_hidden_size)
        self.cross_attn = nn.MultiheadAttention(embed_dim=config.vision_hidden_size,
                                                num_heads=8,
                                                batch_first=True)

    def forward(self, front, lateral):
        # Process images
        front_inputs = self.clip_processor(images=front, return_tensors="pt", padding=True).to(front.device)
        lateral_inputs = self.clip_processor(images=lateral, return_tensors="pt", padding=True).to(lateral.device)

        front_feats = self.clip_model.get_image_features(**front_inputs)  # (B, projection_dim)
        lateral_feats = self.clip_model.get_image_features(**lateral_inputs)

        front_proj = self.proj_head(front_feats)      # (B, H)
        lateral_proj = self.proj_head(lateral_feats)  # (B, H)

        # Expand to sequence length=1 for attention
        f = front_proj.unsqueeze(1)   # (B,1,H)
        l = lateral_proj.unsqueeze(1) # (B,1,H)

        f2l, _ = self.cross_attn(f, l, l)
        l2f, _ = self.cross_attn(l, f, f)
        fused = torch.cat([f2l, l2f], dim=-1)  # (B,1,2H)
        return fused

class BioBARTDecoder(nn.Module):
    def __init__(self, model_name='GanjinZero/biobart-base'):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        # add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def get_embedding_layer(self):
        return self.model.model.shared

    def encode_reports(self, texts, max_length):
        return self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )

class ImageTextFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, vision_feats):
        attn_out, _ = self.cross_attn(x, vision_feats, vision_feats)
        return self.norm(x + self.dropout(attn_out))

class XrayReportModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_encoder = CLIPVisionEncoder(config)
        self.biobart = BioBARTDecoder(config.biobart_model_name)
        self.text_embed = self.biobart.get_embedding_layer()
        self.vision_proj = nn.Linear(config.vision_hidden_size * 2,
                                     self.biobart.model.config.d_model)
        self.fusion_layers = nn.ModuleList([
            ImageTextFusion(embed_dim=self.biobart.model.config.d_model,
                            num_heads=8)
            for _ in range(2)
        ])

    def forward(self, front, lateral, reports):
        fused_vis = self.vision_encoder(front, lateral)  # (B,1,2H)
        fuse = fused_vis.squeeze(1)                     # (B,2H)
        vis_ctx = self.vision_proj(fuse)                # (B,d_model)

        tok = self.biobart.encode_reports(reports, max_length=self.config.max_len)
        labels = tok['input_ids'].to(front.device)
        decoder_mask = (labels != self.biobart.tokenizer.pad_token_id).long().to(front.device)

        B, L = labels.size()
        dec_inputs = vis_ctx.unsqueeze(1).expand(-1, L, -1)  # (B,L,d_model)

        x = dec_inputs
        # Use projected vision context for fusion
        vis_context_seq = vis_ctx.unsqueeze(1)  # (B,1,d_model)
        for fusion in self.fusion_layers:
            x = fusion(x, vis_context_seq)

        outputs = self.biobart.model(
            inputs_embeds=x,
            labels=labels,
            decoder_attention_mask=decoder_mask
        )
        return outputs

    def generate(self, front, lateral, max_length=150, num_beams=4):
        fused_vis = self.vision_encoder(front, lateral)
        fuse = fused_vis.squeeze(1)
        vis_ctx = self.vision_proj(fuse)

        init_embed = vis_ctx.unsqueeze(1)
        gen_ids = self.biobart.model.generate(
            inputs_embeds=init_embed,
            max_length=max_length,
            num_beams=num_beams,
            pad_token_id=self.biobart.tokenizer.pad_token_id,
            eos_token_id=self.biobart.tokenizer.eos_token_id
        )
        return self.biobart.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
