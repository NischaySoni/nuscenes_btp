# ------------------------------------------------------------------
# NuScenes-QA MCAN Model — Ultimate Fusion Architecture (v2)
# Supports: Single-feature (BEV/YOLO) and Dual-Encoder Fusion
#
# Key improvements over v1:
#   1. Separate MCA_ED backbones for BEV and YOLO (no interference)
#   2. Cross-modal attention between modalities
#   3. Vector-level gating conditioned on lang+both visuals
#   4. Separate language representations per modality
#   5. Improved count head with attention
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ops.fc import MLP
from src.ops.layer_norm import LayerNorm
from src.models.mcan.mca import MCA_ED, SGA


# ------------------------------------------------
# Mask creator
# ------------------------------------------------

def make_mask(feature):
    """
    feature: (B, N, D)
    return: (B,1,1,N)
    """
    return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)


# ------------------------------------------------
# Adapter (visual projection)
# ------------------------------------------------

class Adapter(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(Adapter, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, feat):

        feat = feat.to(torch.float32)

        mask = make_mask(feat)

        feat = self.fc(feat)

        return feat, mask


# ------------------------------------------------
# YOLO Class-Aware Adapter
# ------------------------------------------------

class YOLOClassAdapter(nn.Module):
    """
    Projects YOLO features (13-dim) to hidden_dim AND embeds
    the class_id (dim 9) through a learnable embedding table.
    This gives the model semantic identity for each detection.
    """

    def __init__(self, input_dim, hidden_dim, num_classes=80, emb_dim=128):
        super(YOLOClassAdapter, self).__init__()

        # Standard projection for the raw 13-dim features
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable class embedding: class_id -> emb_dim -> hidden_dim
        self.class_emb = nn.Embedding(num_classes, emb_dim)
        self.class_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat):
        """
        feat: (B, 80, 13) — raw YOLO features
        dim 9 = class_id (float, 0-79)
        """
        feat = feat.to(torch.float32)

        mask = make_mask(feat)

        # Extract class IDs before projection (dim 9)
        class_ids = feat[:, :, 9].long().clamp(0, 79)  # (B, 80)

        # Project raw features
        proj = self.fc(feat)  # (B, 80, hidden_dim)

        # Embed class IDs and add to projection
        cls_emb = self.class_emb(class_ids)       # (B, 80, emb_dim)
        cls_proj = self.class_proj(cls_emb)        # (B, 80, hidden_dim)

        # Additive fusion: features + class identity
        proj = proj + cls_proj

        return proj, mask


# ------------------------------------------------
# Attention Flattening
# ------------------------------------------------

class AttFlat(nn.Module):

    def __init__(self, __C):
        super(AttFlat, self).__init__()

        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):

        att = self.mlp(x)

        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e4
        )

        att = F.softmax(att, dim=1)
        self.last_attention = att.detach().cpu()

        att_list = []

        for i in range(self.__C.FLAT_GLIMPSES):

            att_list.append(
                torch.sum(att[:, :, i:i+1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)

        x_atted = self.linear_merge(x_atted)

        return x_atted


# ------------------------------------------------
# Cross-Modal Attention Block
# ------------------------------------------------

class CrossModalAttention(nn.Module):
    """
    Bi-directional cross-attention between two modalities.
    BEV attends to YOLO, YOLO attends to BEV.
    This enables each modality to gather complementary information.
    """

    def __init__(self, __C, num_layers=2):
        super(CrossModalAttention, self).__init__()

        # BEV attends to YOLO
        self.bev_to_yolo = nn.ModuleList([SGA(__C) for _ in range(num_layers)])
        # YOLO attends to BEV
        self.yolo_to_bev = nn.ModuleList([SGA(__C) for _ in range(num_layers)])

    def forward(self, bev_feat, yolo_feat, bev_mask, yolo_mask):
        """
        bev_feat:  (B, N, D)
        yolo_feat: (B, N, D)
        Returns: enhanced bev_feat and yolo_feat
        """
        for sga_b2y, sga_y2b in zip(self.bev_to_yolo, self.yolo_to_bev):
            # BEV gathers info from YOLO
            bev_enhanced = sga_b2y(bev_feat, yolo_feat, bev_mask, yolo_mask)
            # YOLO gathers info from BEV
            yolo_enhanced = sga_y2b(yolo_feat, bev_feat, yolo_mask, bev_mask)

            bev_feat = bev_enhanced
            yolo_feat = yolo_enhanced

        return bev_feat, yolo_feat


# ------------------------------------------------
# MLP Concat Fusion
# ------------------------------------------------

class MLPConcatFusion(nn.Module):
    """
    Fuses two independent visual modalities by concatenation
    and non-linear MLP projection.
    Input: bev_vec (B, D), yolo_vec (B, D)
    """

    def __init__(self, __C):
        super(MLPConcatFusion, self).__init__()

        D = __C.FLAT_OUT_SIZE  # 1024

        self.fuse = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.ReLU(inplace=True),
            nn.Dropout(__C.DROPOUT_R),
            nn.Linear(D, D)
        )

    def forward(self, bev_vec, yolo_vec):
        combined = torch.cat([bev_vec, yolo_vec], dim=-1)  # (B, 2D)
        return self.fuse(combined)


# ------------------------------------------------
# Count Prediction Head (V2)
# ------------------------------------------------

class CountHead(nn.Module):
    """
    Predicts count classes 0..10.
    """

    def __init__(self, __C):
        super(CountHead, self).__init__()

        D = __C.FLAT_OUT_SIZE

        self.count_mlp = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.ReLU(inplace=True),
            nn.Dropout(__C.DROPOUT_R),
            nn.Linear(D, D // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(__C.DROPOUT_R),
            nn.Linear(D // 2, 11)   # 11 classes: 0..10
        )

    def forward(self, fused_feat, lang_feat):
        """
        fused_feat: (B, D)
        lang_feat:  (B, D)
        """
        combined = torch.cat([fused_feat, lang_feat], dim=-1)  # (B, 2D)
        return self.count_mlp(combined)


# ------------------------------------------------
# Main MCAN Model — Ultimate Fusion Architecture
# ------------------------------------------------

class Net(nn.Module):

    def __init__(self, __C, pretrained_emb, token_size, answer_size):

        super(Net, self).__init__()

        self.__C = __C
        self.is_fusion = (getattr(__C, 'VISUAL_FEATURE', 'bev') == 'fusion')

        # Determine feature dimensions
        bev_dim = __C.FEAT_SIZE['OBJ_FEAT_SIZE'][1]   # 69 for BEV

        if self.is_fusion:
            yolo_dim = __C.FEAT_SIZE['BBOX_FEAT_SIZE'][1]  # 13 for YOLO
            print(f"  [MCAN] FUSION mode: BEV dim={bev_dim}, YOLO dim={yolo_dim}")
        else:
            # Single-feature mode (backward compatible)
            feat_dim = bev_dim
            if feat_dim == 69:
                self.img_dim = 64
                self.radar_dim = 5
                self.feat_mode = 'bev'
            elif feat_dim == 13:
                self.img_dim = 10
                self.radar_dim = 3
                self.feat_mode = 'yolo'
            else:
                self.radar_dim = min(3, feat_dim)
                self.img_dim = feat_dim - self.radar_dim
                self.feat_mode = 'generic'

            self.radar_expand = max(60, self.radar_dim)
            print(f"  [MCAN] feat_mode={self.feat_mode}, feat_dim={feat_dim}, "
                  f"img_dim={self.img_dim}, radar_dim={self.radar_dim}")

        # ---------------- Language ----------------

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        self.embedding.weight.data.copy_(
            torch.from_numpy(pretrained_emb)
        )

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        # ================================================
        # FUSION MODE — ULTIMATE ARCHITECTURE
        # ================================================
        if self.is_fusion:

            # --- Visual Adapters ---
            self.bev_adapter = Adapter(bev_dim, __C.HIDDEN_SIZE)
            self.yolo_adapter = YOLOClassAdapter(yolo_dim, __C.HIDDEN_SIZE)

            # --- SEPARATE MCAN backbones (key fix: no shared weights) ---
            self.backbone_bev = MCA_ED(__C)
            self.backbone_yolo = MCA_ED(__C)

            # --- Cross-Modal Attention ---
            cross_layers = getattr(__C, 'CROSS_MODAL_LAYERS', 2)
            self.cross_modal = CrossModalAttention(__C, num_layers=cross_layers)

            # --- Separate Flatten layers ---
            self.attflat_lang_bev = AttFlat(__C)
            self.attflat_lang_yolo = AttFlat(__C)
            self.attflat_bev = AttFlat(__C)
            self.attflat_yolo = AttFlat(__C)

            # --- Language fusion (concat + project instead of averaging) ---
            self.lang_fusion = nn.Sequential(
                nn.Linear(__C.FLAT_OUT_SIZE * 2, __C.FLAT_OUT_SIZE),
                nn.ReLU(inplace=True),
                nn.Dropout(__C.DROPOUT_R),
            )

            # --- MLP Fusion (Replaces Element-Wise Gate) ---
            self.fusion_mlp = MLPConcatFusion(__C)

            # --- BEV residual weight (learnable, init 0.3) ---
            self.bev_residual_weight = nn.Parameter(torch.tensor(0.3))

            # --- Count head (improved) ---
            self.count_head = CountHead(__C)

            # --- Classification head ---
            self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
            self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

            # Print parameter count
            total_params = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  [MCAN Fusion] Total params: {total_params:,}, Trainable: {trainable:,}")

        # ================================================
        # SINGLE-FEATURE MODE (backward compatible)
        # ================================================
        else:

            self.img_adapter = Adapter(self.img_dim, __C.HIDDEN_SIZE)
            self.radar_adapter = Adapter(self.radar_expand, __C.HIDDEN_SIZE)

            self.backbone_img = MCA_ED(__C)
            self.backbone_radar = MCA_ED(__C)

            self.attflat_lang = AttFlat(__C)
            self.attflat_img = AttFlat(__C)
            self.attflat_radar = AttFlat(__C)

            self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
            self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, obj_feat, bbox_feat, ques_ix):
        """
        Fusion mode:
            obj_feat  : (B, 80, 69) BEV features
            bbox_feat : (B, 80, 13) YOLO features
        Single mode:
            obj_feat  : (B, 80, feat_dim) visual features
            bbox_feat : (B, 80, 4) dummy / unused
        ques_ix       : (B, T) question token indices
        """

        # ------------------------------------------------
        # Language (shared for both modes)
        # ------------------------------------------------

        lang_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        if self.is_fusion:
            return self._forward_fusion(obj_feat, bbox_feat, lang_feat, lang_mask)
        else:
            return self._forward_single(obj_feat, lang_feat, lang_mask)


    def _forward_fusion(self, bev_feat, yolo_feat, lang_feat, lang_mask):
        """Ultimate dual-encoder fusion path."""

        # ------------------------------------------------
        # Adapt features (richer 2-layer projection)
        # ------------------------------------------------

        bev_proj, bev_mask = self.bev_adapter(bev_feat)      # (B, 80, 512)
        yolo_proj, yolo_mask = self.yolo_adapter(yolo_feat)  # (B, 80, 512)

        # ------------------------------------------------
        # SEPARATE MCAN co-attention for BEV
        # (dedicated backbone — no interference with YOLO)
        # ------------------------------------------------

        lang_bev, bev_out = self.backbone_bev(
            lang_feat.clone(),
            bev_proj,
            lang_mask,
            bev_mask
        )

        # ------------------------------------------------
        # SEPARATE MCAN co-attention for YOLO
        # (dedicated backbone — no interference with BEV)
        # ------------------------------------------------

        lang_yolo, yolo_out = self.backbone_yolo(
            lang_feat.clone(),
            yolo_proj,
            lang_mask,
            yolo_mask
        )

        # ------------------------------------------------
        # Cross-Modal Attention
        # BEV and YOLO exchange information
        # ------------------------------------------------

        bev_out, yolo_out = self.cross_modal(
            bev_out, yolo_out,
            bev_mask, yolo_mask
        )

        # ------------------------------------------------
        # Flatten — separate paths for each modality
        # ------------------------------------------------

        lang_bev_vec = self.attflat_lang_bev(lang_bev, lang_mask)    # (B, 1024)
        lang_yolo_vec = self.attflat_lang_yolo(lang_yolo, lang_mask) # (B, 1024)

        bev_vec = self.attflat_bev(bev_out, bev_mask)    # (B, 1024)
        yolo_vec = self.attflat_yolo(yolo_out, yolo_mask) # (B, 1024)

        # ------------------------------------------------
        # Language Fusion (concat + project, not avg)
        # ------------------------------------------------

        lang_vec = self.lang_fusion(
            torch.cat([lang_bev_vec, lang_yolo_vec], dim=-1)
        )  # (B, 1024)

        # ------------------------------------------------
        # Late Fusion: MLP instead of Element-Wise Gating
        # ------------------------------------------------

        fused_visual = self.fusion_mlp(bev_vec, yolo_vec)

        # ------------------------------------------------
        # Classify (with BEV residual)
        # ------------------------------------------------

        bev_residual = self.bev_residual_weight * bev_vec
        proj_feat = lang_vec + fused_visual + bev_residual
        proj_feat = self.proj_norm(proj_feat)
        logits = self.proj(proj_feat)

        # ------------------------------------------------
        # Count head (auxiliary output)
        # ------------------------------------------------

        count_logits = self.count_head(fused_visual, lang_vec)

        # Store for the training engine to use
        self._count_logits = count_logits
        self._fusion_gate_mean = 0.5  # dummy for diagnostics

        return logits


    def _forward_single(self, obj_feat, lang_feat, lang_mask):
        """Single-feature path (backward compatible with BEV/YOLO only)."""

        # Split Image + Radar
        img_feat = obj_feat[:, :, :self.img_dim]
        radar_feat = obj_feat[:, :, self.img_dim:]

        # Radar confidence weighting on image features
        radar_conf = radar_feat[:, :, -1].unsqueeze(-1)
        img_feat = img_feat * (1 + radar_conf)

        # Expand radar to target dimension via repeat
        if self.radar_dim < self.radar_expand:
            repeat_factor = (self.radar_expand // self.radar_dim) + 1
            radar_feat = radar_feat.repeat(1, 1, repeat_factor)[:, :, :self.radar_expand]

        radar_feat = radar_feat / (torch.norm(radar_feat, dim=-1, keepdim=True) + 1e-6)

        # Visual adapters
        img_feat, img_mask = self.img_adapter(img_feat)
        radar_feat, radar_mask = self.radar_adapter(radar_feat)

        # MCAN for Image
        lang_img, img_feat = self.backbone_img(
            lang_feat,
            img_feat,
            lang_mask,
            img_mask
        )

        # MCAN for Radar
        lang_rad, radar_feat = self.backbone_radar(
            lang_feat,
            radar_feat,
            lang_mask,
            radar_mask
        )

        # Flatten
        lang_vec = self.attflat_lang(lang_feat, lang_mask)
        img_vec = self.attflat_img(img_feat, img_mask)
        radar_vec = self.attflat_radar(radar_feat, radar_mask)

        # Late Fusion
        proj_feat = lang_vec + img_vec + 0.1 * radar_vec

        presence = radar_feat[:, :, 0].sum(dim=1, keepdim=True)
        presence = presence.expand(-1, proj_feat.shape[1])
        proj_feat = proj_feat + 0.2 * presence

        proj_feat = self.proj_norm(proj_feat)
        logits = self.proj(proj_feat)

        return logits
