# ------------------------------------------------------------------
# NuScenes-QA MCAN Model
# Supports: Single-feature (BEV/YOLO) and Dual-Encoder Fusion
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ops.fc import MLP
from src.ops.layer_norm import LayerNorm
from src.models.mcan.mca import MCA_ED


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

        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, feat):

        feat = feat.to(torch.float32)

        mask = make_mask(feat)

        feat = self.linear(feat)

        return feat, mask


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
# Learned Fusion Gate
# ------------------------------------------------

class FusionGate(nn.Module):
    """
    Question-conditioned gating: decides how much to weight
    BEV vs YOLO features for each sample/question.

    Input:  lang_vec (B, FLAT_OUT_SIZE)
    Output: alpha    (B, 1) in [0, 1]

    Final fused = alpha * bev_vec + (1-alpha) * yolo_vec
    """

    def __init__(self, __C):
        super(FusionGate, self).__init__()

        self.gate = nn.Sequential(
            nn.Linear(__C.FLAT_OUT_SIZE, __C.FLAT_OUT_SIZE // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(__C.DROPOUT_R),
            nn.Linear(__C.FLAT_OUT_SIZE // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, lang_vec):
        return self.gate(lang_vec)


# ------------------------------------------------
# Count Prediction Head
# ------------------------------------------------

class CountHead(nn.Module):
    """
    Auxiliary head for count-type questions.
    Predicts a soft count in range [0, 10].
    """

    def __init__(self, __C):
        super(CountHead, self).__init__()

        self.count_mlp = nn.Sequential(
            nn.Linear(__C.FLAT_OUT_SIZE, __C.FLAT_OUT_SIZE // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(__C.DROPOUT_R),
            nn.Linear(__C.FLAT_OUT_SIZE // 2, 11)   # 11 classes: 0..10
        )

    def forward(self, fused_feat):
        return self.count_mlp(fused_feat)


# ------------------------------------------------
# Main MCAN Model — with Fusion support
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
        # FUSION MODE
        # ================================================
        if self.is_fusion:

            # --- Visual Adapters ---
            self.bev_adapter = Adapter(bev_dim, __C.HIDDEN_SIZE)
            self.yolo_adapter = Adapter(yolo_dim, __C.HIDDEN_SIZE)

            # --- Shared MCAN backbone ---
            self.backbone = MCA_ED(__C)

            # --- Separate Flatten layers ---
            self.attflat_lang = AttFlat(__C)
            self.attflat_bev = AttFlat(__C)
            self.attflat_yolo = AttFlat(__C)

            # --- Fusion gate ---
            self.fusion_gate = FusionGate(__C)

            # --- Count head ---
            self.count_head = CountHead(__C)

            # --- Classification head ---
            self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
            self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

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
        """Dual-encoder fusion path."""

        # ------------------------------------------------
        # Adapt features
        # ------------------------------------------------

        bev_proj, bev_mask = self.bev_adapter(bev_feat)      # (B, 80, 512)
        yolo_proj, yolo_mask = self.yolo_adapter(yolo_feat)  # (B, 80, 512)

        # ------------------------------------------------
        # MCAN co-attention for BEV
        # (shared backbone, but called sequentially)
        # ------------------------------------------------

        lang_bev, bev_out = self.backbone(
            lang_feat.clone(),
            bev_proj,
            lang_mask,
            bev_mask
        )

        # ------------------------------------------------
        # MCAN co-attention for YOLO
        # ------------------------------------------------

        lang_yolo, yolo_out = self.backbone(
            lang_feat.clone(),
            yolo_proj,
            lang_mask,
            yolo_mask
        )

        # ------------------------------------------------
        # Flatten
        # ------------------------------------------------

        # Use the average of both lang representations
        lang_combined = (lang_bev + lang_yolo) / 2.0
        lang_vec = self.attflat_lang(lang_combined, lang_mask)

        bev_vec = self.attflat_bev(bev_out, bev_mask)
        yolo_vec = self.attflat_yolo(yolo_out, yolo_mask)

        # ------------------------------------------------
        # Learned Gating
        # ------------------------------------------------

        alpha = self.fusion_gate(lang_vec)   # (B, 1)
        fused_visual = alpha * bev_vec + (1.0 - alpha) * yolo_vec   # (B, FLAT_OUT_SIZE)

        # ------------------------------------------------
        # Classify
        # ------------------------------------------------

        proj_feat = lang_vec + fused_visual
        proj_feat = self.proj_norm(proj_feat)
        logits = self.proj(proj_feat)

        # ------------------------------------------------
        # Count head (auxiliary output)
        # ------------------------------------------------

        count_logits = self.count_head(proj_feat)

        # Store for the training engine to use
        self._count_logits = count_logits
        self._fusion_alpha = alpha.detach().mean().item()

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
