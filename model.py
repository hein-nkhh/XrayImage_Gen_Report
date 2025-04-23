# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import CLIPProcessor, CLIPModel
# from transformers import BartTokenizer, BartForConditionalGeneration
# from config import Config
# import torch.nn.functional as F

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

# class CLIPVisionEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
#         for name, param in self.clip_model.vision_model.named_parameters():
#             param.requires_grad = False


#         self.hidden_size = self.clip_model.config.vision_config.hidden_size  # thường là 768
#         self.proj_head = ProjectionHead(self.hidden_size, config.vision_hidden_size)

#         self.cross_attn_f2l = nn.MultiheadAttention(embed_dim=config.vision_hidden_size,
#                                                     num_heads=8, batch_first=True)
#         self.cross_attn_l2f = nn.MultiheadAttention(embed_dim=config.vision_hidden_size,
#                                                     num_heads=8, batch_first=True)

#     def _extract_features(self, image_batch):
#         inputs = self.clip_processor(images=image_batch, return_tensors="pt")
#         inputs = {k: v.to(Config.device) for k, v in inputs.items()}
#         outputs = self.clip_model.vision_model(**inputs)  # lấy hidden states
#         features = outputs.last_hidden_state  # (B, num_patches+1, hidden_size)
#         features = features[:, 1:, :]  # bỏ [CLS] token, chỉ lấy patch tokens
#         return features

#     def forward(self, front, lateral):
#         front_feats = self._extract_features(front)      # (B, N, hidden)
#         lateral_feats = self._extract_features(lateral)  # (B, N, hidden)

#         front_proj = self.proj_head(front_feats)         # (B, N, H)
#         lateral_proj = self.proj_head(lateral_feats)     # (B, N, H)

#         # Cross attention: front attends to lateral, and vice versa
#         f2l, _ = self.cross_attn_f2l(front_proj, lateral_proj, lateral_proj)
#         l2f, _ = self.cross_attn_l2f(lateral_proj, front_proj, front_proj)

#         fused = torch.cat([f2l.mean(dim=1), l2f.mean(dim=1)], dim=-1)  # (B, 2H)
#         return fused.unsqueeze(1)  # (B, 1, 2H)


# class BioBARTDecoder(nn.Module):
#     def __init__(self, model_name='GanjinZero/biobart-base'):
#         super().__init__()
#         self.tokenizer = BartTokenizer.from_pretrained(model_name)
#         self.model = BartForConditionalGeneration.from_pretrained(model_name)
#         # add pad token if missing
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#             self.model.resize_token_embeddings(len(self.tokenizer))

#     def get_embedding_layer(self):
#         return self.model.model.shared

#     def encode_reports(self, texts, max_length):
#         return self.tokenizer(
#             texts,
#             add_special_tokens=True,
#             truncation=True,
#             max_length=max_length,
#             padding='max_length',
#             return_tensors='pt'
#         )

# class BiDirectionalFusion(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         self.attn_text2vis = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
#         self.attn_vis2text = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

#         self.ffn = nn.Sequential(
#             nn.Linear(embed_dim * 2, embed_dim),
#             nn.GELU(),
#             nn.LayerNorm(embed_dim)
#         )

#     def forward(self, x, vision_ctx):
#         # x: (B, L, d_model), vision_ctx: (B, 1, d_model)
#         t2v, _ = self.attn_text2vis(x, vision_ctx, vision_ctx)     # (B, L, d_model)
#         v2t, _ = self.attn_vis2text(vision_ctx, x, x)               # (B, 1, d_model)
#         v2t_expand = v2t.expand(-1, x.size(1), -1)                  # (B, L, d_model)

#         fused = torch.cat([t2v, v2t_expand], dim=-1)                # (B, L, 2*d_model)
#         return self.ffn(fused)                                      # (B, L, d_model)

# # class ImageTextFusion(nn.Module):
# #     def __init__(self, embed_dim, num_heads):
# #         super().__init__()
# #         self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim,
# #                                                 num_heads=num_heads,
# #                                                 batch_first=True)
# #         self.norm = nn.LayerNorm(embed_dim)
# #         self.dropout = nn.Dropout(0.1)

# #     def forward(self, x, vision_feats):
# #         attn_out, _ = self.cross_attn(x, vision_feats, vision_feats)
# #         return self.norm(x + self.dropout(attn_out))
    
# class XrayReportModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.vision_encoder = CLIPVisionEncoder(config)
#         self.biobart = BioBARTDecoder(config.biobart_model_name)
#         self.text_embed = self.biobart.get_embedding_layer()
#         self.vision_proj = nn.Linear(config.vision_hidden_size * 2,
#                                      self.biobart.model.config.d_model)
#         self.fusion_layers = nn.ModuleList([
#             BiDirectionalFusion(embed_dim=self.biobart.model.config.d_model,
#                             num_heads=8)
#             for _ in range(2)
#         ])
#         self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.biobart.tokenizer.pad_token_id, label_smoothing=0.1)


#     def forward(self, front, lateral, reports):
#         fused_vis = self.vision_encoder(front, lateral)  # (B,1,2H)
#         fuse = fused_vis.squeeze(1)                     # (B,2H)
#         vis_ctx = self.vision_proj(fuse)                # (B,d_model)

#         tok = self.biobart.encode_reports(reports, max_length=self.config.max_len)
#         labels = tok['input_ids'].to(Config.device)
#         decoder_mask = (labels != self.biobart.tokenizer.pad_token_id).long().to(Config.device)

#         B, L = labels.size()
#         dec_inputs = vis_ctx.unsqueeze(1).expand(-1, L, -1)  # (B,L,d_model)

#         x = dec_inputs
#         # Use projected vision context for fusion
#         vis_context_seq = vis_ctx.unsqueeze(1)  # (B,1,d_model)
#         for fusion in self.fusion_layers:
#             x = fusion(x, vis_context_seq)

#         outputs = self.biobart.model(
#             inputs_embeds=x,
#             labels=labels,
#             decoder_attention_mask=decoder_mask
#         )
#         logits = outputs.logits
#         loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
#         # Optional: Add coverage loss (simple penalty for repeated tokens)
#         probs = F.softmax(logits, dim=-1)
#         token_probs = probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # (B, L)
#         rep_penalty = torch.mean(torch.sum(token_probs, dim=1) / labels.size(1))  # normalize
#         loss += self.config.coverage_lambda * rep_penalty

#         outputs.loss = loss
#         return outputs

#         return outputs

#     def generate(self, front, lateral, max_length=150, num_beams=4):
#         fused_vis = self.vision_encoder(front, lateral)
#         fuse = fused_vis.squeeze(1)
#         vis_ctx = self.vision_proj(fuse)

#         init_embed = vis_ctx.unsqueeze(1)
#         gen_ids = self.biobart.model.generate(
#             inputs_embeds=init_embed,
#             max_length=max_length,
#             num_beams=num_beams,
#             pad_token_id=self.biobart.tokenizer.pad_token_id,
#             eos_token_id=self.biobart.tokenizer.eos_token_id
#         )
#         return self.biobart.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, BartTokenizer, BartForConditionalGeneration
from config import Config

class EnhancedCLIPVisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        # Freeze vision model
        for param in self.clip_model.vision_model.parameters():
            param.requires_grad = False

        self.hidden_dim = self.clip_model.config.vision_config.hidden_size  # 768
        self.proj_head = nn.Sequential(
            nn.Linear(self.hidden_dim, config.vision_hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.vision_hidden_size),
            nn.Dropout(0.1)
        )

        # Cross-view fusion
        self.cross_fusion = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config.vision_hidden_size,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=2
        )

        # Attention pooling query
        self.query = nn.Parameter(torch.randn(1, 1, config.vision_hidden_size))

    def _extract_features(self, images):
        with torch.no_grad():
            inputs = self.clip_processor(images=images, return_tensors="pt")
            inputs = {k: v.to(Config.device) for k, v in inputs.items()}
            features = self.clip_model.vision_model(**inputs).last_hidden_state  # [B, 197, 768]
        return features  # keep CLS token (197 tokens)

    def _attention_pooling(self, x):
        B = x.size(0)
        query = self.query.expand(B, -1, -1)  # [B, 1, D]
        attn_output, _ = nn.MultiheadAttention(
            embed_dim=x.size(-1),
            num_heads=8,
            batch_first=True
        )(query, x, x)
        return attn_output.squeeze(1)  # [B, D]

    def forward(self, front, lateral):
        # Extract features
        front_feats = self.proj_head(self._extract_features(front))  # [B, 197, D]
        lateral_feats = self.proj_head(self._extract_features(lateral))  # [B, 197, D]

        # Concatenate both views
        combined = torch.cat([front_feats, lateral_feats], dim=1)  # [B, 394, D]

        # Cross-view fusion
        fused = self.cross_fusion(combined)  # [B, 394, D]
        fused = fused.to(Config.device)
        # Attention pooling
        return self._attention_pooling(fused)  # [B, D]

class EnhancedBioBARTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained(config.biobart_model_name)
        self.model = BartForConditionalGeneration.frdeviceom_pretrained(config.biobart_model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.vis_proj = nn.Linear(config.vision_hidden_size, self.model.config.d_model)

        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.model.config.d_model, 
                num_heads=8, 
                batch_first=True)
            for _ in range(3)
        ])

        self.coverage_weights = nn.Parameter(torch.randn(1, 8, 1, 64))  # For 8 attention 
        self.coverage_loss_fn = nn.MSELoss()

    def forward(self, vis_features, reports):
        vis_ctx = self.vis_proj(vis_features).unsqueeze(1)

        inputs = self.tokenizer(
            reports,
            padding='max_length',
            max_length=Config.max_len,
            return_tensors='pt'
        ).to(Config.device)

        text_embeds = self.model.model.shared(inputs.input_ids)
        # Initialize coverage vector
        B, L = inputs.input_ids.shape
        coverage = torch.zeros(B, L, 1).to(Config.device)

        # Enhanced decoding with cross-modal attention
        for i, layer in enumerate(self.model.model.decoder.layers):
            # Self-attention
            text_embeds = layer(
                text_embeds, 
                attention_mask=inputs.attention_mask
            )[0]
            
            # Cross-modal attention
            if i < len(self.cross_attn_layers):
                attn_output, attn_weights = self.cross_attn_layers[i](
                    text_embeds,
                    vis_ctx.expand(-1, text_embeds.size(1), -1),
                    vis_ctx.expand(-1, text_embeds.size(1), -1)
                )
                text_embeds = text_embeds + attn_output
                
                # Update coverage
                coverage += attn_weights.mean(dim=1).unsqueeze(-1)
        
        # Calculate coverage loss
        cov_loss = self.coverage_loss(
            F.conv2d(coverage.permute(0,2,1).unsqueeze(1), self.coverage_weights),
            torch.ones_like(coverage)
        )
        
        # Final outputs
        logits = self.model.lm_head(text_embeds)
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            inputs.input_ids.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=0.1
        )
        
        return {
            'loss': ce_loss + 0.1 * cov_loss,
            'logits': logits
        }

    def generate(self, vis_features, max_length=150):
        vis_ctx = self.vis_proj(vis_features).unsqueeze(1)
        
        gen_ids = self.model.generate(
            inputs_embeds=vis_ctx,
            max_length=max_length,
            num_beams=4,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

class XrayReportModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = EnhancedCLIPVisionEncoder(config)
        self.decoder = EnhancedBioBARTDecoder(config)

    def forward(self, front, lateral, reports):
        vis_features = self.encoder(front, lateral)
        return self.decoder(vis_features, reports)

    def generate(self, front, lateral, max_length=150):
        vis_features = self.encoder(front, lateral)
        return self.decoder.generate(vis_features, max_length=max_length)

