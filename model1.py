import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, BartTokenizer, BartForConditionalGeneration
from config import Config

class BiomedCLIPVisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Using BiomedCLIP instead of standard CLIP for medical domain expertise
        self.vision_model = AutoModel.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        self.processor = AutoProcessor.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        
        # Freeze base vision model
        for param in self.vision_model.parameters():
            param.requires_grad = False

        self.hidden_dim = self.vision_model.config.vision_config.hidden_size  # Usually 768
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

        # Attention pooling for global features
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=config.vision_hidden_size,
            num_heads=8,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, config.vision_hidden_size))
        
        # Abnormality detection module
        self.abnormality_detector = AbnormalityDetector(config.vision_hidden_size)

    def _extract_features(self, images):
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(Config.device) for k, v in inputs.items()}
            # Extract visual features (patch embeddings)
            features = self.vision_model.vision_model(**inputs).last_hidden_state  # [B, num_patches, hidden_dim]
        return features

    def _attention_pooling(self, x):
        B = x.size(0)
        query = self.query.expand(B, -1, -1).to(x.device)
        attn_output, _ = self.attn_pool(query, x, x)
        return attn_output.squeeze(1)

    def forward(self, front, lateral):
        # Extract patch features from both views
        front_feats = self._extract_features(front)  # [B, num_patches, hidden_dim]
        lateral_feats = self._extract_features(lateral)  # [B, num_patches, hidden_dim]
        
        # Project features
        front_proj = self.proj_head(front_feats)  # [B, num_patches, vision_hidden_size]
        lateral_proj = self.proj_head(lateral_feats)  # [B, num_patches, vision_hidden_size]
        
        # Detect abnormality regions
        front_attn_map = self.abnormality_detector(front_proj)  # [B, num_patches, 1]
        lateral_attn_map = self.abnormality_detector(lateral_proj)  # [B, num_patches, 1]
        
        # Apply abnormality attention to features
        front_focused = front_proj * (1 + front_attn_map)
        lateral_focused = lateral_proj * (1 + lateral_attn_map)
        
        # Concatenate both views for cross-view fusion
        combined = torch.cat([front_focused, lateral_focused], dim=1)  # [B, 2*num_patches, vision_hidden_size]
        
        # Cross-view fusion through transformer layers
        fused = self.cross_fusion(combined)  # [B, 2*num_patches, vision_hidden_size]
        
        # Global feature extraction via attention pooling
        global_feature = self._attention_pooling(fused)  # [B, vision_hidden_size]
        
        # Keep local features for region-specific processing
        # Select top-k abnormal regions based on attention maps
        k = 5  # Number of regions to focus on
        front_scores, front_indices = torch.topk(front_attn_map.squeeze(-1), k=k, dim=1)  # [B, k]
        lateral_scores, lateral_indices = torch.topk(lateral_attn_map.squeeze(-1), k=k, dim=1)  # [B, k]
        
        # Gather top-k abnormal regions
        B = front_proj.size(0)
        local_features = []
        
        for b in range(B):
            # Get features for front view abnormal regions
            for idx in front_indices[b]:
                local_features.append(front_proj[b, idx].unsqueeze(0))  # [1, vision_hidden_size]
            
            # Get features for lateral view abnormal regions
            for idx in lateral_indices[b]:
                local_features.append(lateral_proj[b, idx].unsqueeze(0))  # [1, vision_hidden_size]
        
        # Stack local features [B, 2*k, vision_hidden_size]
        local_features = torch.cat(local_features, dim=0).reshape(B, 2*k, -1)
        
        return global_feature, local_features, front_attn_map, lateral_attn_map


class AbnormalityDetector(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Self-attention based abnormality detector
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.abnormality_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, num_patches, hidden_dim]
        
        # Project queries, keys, values
        queries = self.query_proj(x)
        keys = self.key_proj(x)
        values = self.value_proj(x)
        
        # Self-attention
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (x.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, values)
        
        # Score each region for abnormality
        abnormality_scores = self.abnormality_scorer(context)  # [B, num_patches, 1]
        
        return abnormality_scores


class FactualConsistencyModule(nn.Module):
    def __init__(self, hidden_dim, num_findings=14):
        super().__init__()
        # Define common findings in chest X-rays (like those in CheXpert dataset)
        self.num_findings = num_findings
        self.finding_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_findings),
            nn.Sigmoid()
        )
        
        # Learnable embeddings for each medical finding
        self.finding_embeddings = nn.Parameter(torch.randn(num_findings, hidden_dim))
        
    def forward(self, visual_features):
        # Predict presence of medical findings from visual features
        # visual_features: [B, hidden_dim]
        finding_probs = self.finding_classifier(visual_features)  # [B, num_findings]
        
        # Generate finding-specific embeddings using probabilities as weights
        B = visual_features.size(0)
        # [B, num_findings, 1] * [1, num_findings, hidden_dim] -> [B, num_findings, hidden_dim]
        weighted_findings = finding_probs.unsqueeze(-1) * self.finding_embeddings.unsqueeze(0)
        # Sum across findings dimension to get combined finding embedding
        finding_context = weighted_findings.sum(dim=1)  # [B, hidden_dim]
        
        return finding_probs, finding_context


class GlobalLocalFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)
        self.local_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, global_feat, local_feats):
        # global_feat: [B, hidden_dim], local_feats: [B, num_regions, hidden_dim]
        B, num_regions, D = local_feats.size()
        
        # Project features
        global_proj = self.global_proj(global_feat).unsqueeze(1)  # [B, 1, hidden_dim]
        local_proj = self.local_proj(local_feats)  # [B, num_regions, hidden_dim]
        
        # Expand global features to match local features shape
        global_expanded = global_proj.expand(-1, num_regions, -1)  # [B, num_regions, hidden_dim]
        
        # Compute fusion gates
        concat_feats = torch.cat([global_expanded, local_proj], dim=-1)  # [B, num_regions, 2*hidden_dim]
        gates = self.fusion_gate(concat_feats)  # [B, num_regions, hidden_dim]
        
        # Fuse global and local features using gates
        fused_features = gates * local_proj + (1 - gates) * global_expanded  # [B, num_regions, hidden_dim]
        
        # Average across regions dimension
        fused_output = fused_features.mean(dim=1)  # [B, hidden_dim]
        
        return fused_output


class EnhancedBioBARTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained(config.biobart_model_name)
        self.model = BartForConditionalGeneration.from_pretrained(config.biobart_model_name)
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Project visual features to text embedding space
        self.vis_proj = nn.Linear(config.vision_hidden_size, self.model.config.d_model)
        
        # Cross-attention layers for integrating visual context
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.model.config.d_model,
                num_heads=8,
                batch_first=True
            )
            for _ in range(3)
        ])
        
        # Additional finding-specific attention for factual consistency
        self.finding_attn = nn.MultiheadAttention(
            embed_dim=self.model.config.d_model,
            num_heads=4,
            batch_first=True
        )
        
        # Coverage mechanism to avoid repetition
        self.coverage_weights = nn.Linear(1, 1)
        self.coverage_loss_fn = nn.MSELoss()

    def forward(self, vis_features, finding_context, reports):
        # vis_features: [B, hidden_dim], finding_context: [B, hidden_dim]
        # Project visual features to text embedding space
        vis_ctx = self.vis_proj(vis_features).unsqueeze(1)  # [B, 1, d_model]
        
        # Project finding context to text embedding space
        finding_ctx = self.vis_proj(finding_context).unsqueeze(1)  # [B, 1, d_model]
        
        # Tokenize reports
        inputs = self.tokenizer(
            reports,
            padding=True,
            truncation=True,
            max_length=Config.max_len,
            return_tensors='pt'
        ).to(Config.device)
        
        # Get text embeddings
        text_embeds = self.model.model.shared(inputs.input_ids)  # [B, L, d_model]
        
        # Initialize coverage vector
        B, L = inputs.input_ids.shape
        coverage = torch.zeros(B, L, 1).to(Config.device)
        
        # Create attention mask
        attn_mask = inputs.attention_mask.bool()
        num_heads = self.model.config.decoder_attention_heads
        
        # Process through decoder layers with additional cross-modal attention
        for i, layer in enumerate(self.model.model.decoder.layers):
            # Self-attention
            text_embeds = layer(
                text_embeds,
                attention_mask=attn_mask
            )[0]
            
            # Cross-modal attention with visual context
            if i < len(self.cross_attn_layers):
                attn_output, attn_weights = self.cross_attn_layers[i](
                    text_embeds,
                    vis_ctx.expand(-1, text_embeds.size(1), -1),
                    vis_ctx.expand(-1, text_embeds.size(1), -1)
                )
                text_embeds = text_embeds + attn_output
                
                # Update coverage
                coverage += attn_weights.mean(dim=1).unsqueeze(-1)
                
            # Finding-specific attention (factual consistency)
            if i == 1:  # Apply in middle layer
                finding_attn_output, _ = self.finding_attn(
                    text_embeds,
                    finding_ctx.expand(-1, text_embeds.size(1), -1),
                    finding_ctx.expand(-1, text_embeds.size(1), -1)
                )
                text_embeds = text_embeds + finding_attn_output
        
        # Calculate coverage loss
        cov_out = self.coverage_weights(coverage)
        cov_loss = self.coverage_loss_fn(cov_out, coverage)
        
        # Final logits
        logits = self.model.lm_head(text_embeds)
        
        # Cross-entropy loss with label smoothing
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

    def generate(self, vis_features, finding_context, finding_probs=None, max_length=150):
        # Project features to embedding space
        vis_ctx = self.vis_proj(vis_features).unsqueeze(1)  # [B, 1, d_model]
        
        # Optional: Use finding probabilities to guide generation via forced_decoder_ids
        forced_decoder_ids = None
        if finding_probs is not None:
            # This would require additional implementation to convert finding_probs
            # to forced decoder ids based on your specific tokenizer/vocabulary
            pass
        
        # Generate report
        gen_ids = self.model.generate(
            inputs_embeds=vis_ctx,
            max_length=max_length,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            forced_decoder_ids=forced_decoder_ids
        )
        
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


class XrayReportModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Improved vision encoder with BiomedCLIP and abnormality detection
        self.encoder = BiomedCLIPVisionEncoder(config)
        
        # Factual consistency module
        self.factual_module = FactualConsistencyModule(config.vision_hidden_size)
        
        # Global-local fusion module
        self.fusion_module = GlobalLocalFusion(config.vision_hidden_size)
        
        # Enhanced decoder with factual consistency integration
        self.decoder = EnhancedBioBARTDecoder(config)
        
        # Medical finding classifier for auxiliary supervision
        self.findings_classifier = nn.Linear(config.vision_hidden_size, 14)  # Common findings

    def forward(self, front, lateral, reports=None, finding_labels=None):
        # Extract visual features with abnormality detection
        global_feat, local_feats, front_attn_map, lateral_attn_map = self.encoder(front, lateral)
        
        # Factual consistency prediction
        finding_probs, finding_context = self.factual_module(global_feat)
        
        # Fuse global and local features
        fused_visual = self.fusion_module(global_feat, local_feats)
        
        # Compute losses
        outputs = {
            'finding_probs': finding_probs,
            'front_attn_map': front_attn_map,
            'lateral_attn_map': lateral_attn_map
        }
        
        # Auxiliary finding classification loss if labels provided
        if finding_labels is not None:
            finding_loss = F.binary_cross_entropy_with_logits(
                self.findings_classifier(fused_visual),
                finding_labels
            )
            outputs['finding_loss'] = finding_loss
        
        # Generate report if reports are provided (training) or None (inference)
        if reports is not None:
            report_outputs = self.decoder(fused_visual, finding_context, reports)
            outputs.update(report_outputs)
        
        return outputs

    def generate(self, front, lateral, max_length=150):
        # Extract visual features with abnormality detection
        global_feat, local_feats, _, _ = self.encoder(front, lateral)
        
        # Get factual consistency predictions and context
        finding_probs, finding_context = self.factual_module(global_feat)
        
        # Fuse global and local features
        fused_visual = self.fusion_module(global_feat, local_feats)
        
        # Generate report
        reports = self.decoder.generate(fused_visual, finding_context, finding_probs, max_length)
        
        return {
            'reports': reports,
            'finding_probs': finding_probs
        }