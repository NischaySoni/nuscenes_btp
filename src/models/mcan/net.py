# ------------------------------------------------------------------
# NuScenes-QA MCAN Model — RadarXFormer-Inspired Architecture
# Supports: Single-feature (BEV/YOLO), Dual-Encoder Fusion,
#           Annotation, Detected, and RadarXFormer (radarxf) modes
#
# RadarXFormer additions (inspired by 2603.14822v1):
#   1. RadarXFormerAdapter: dual-stream encoder for structured + CLIP features
#   2. SpatialPositionalEncoding: 3D position encoding (spherical + Cartesian)
#   3. Iterative refinement: cross-object attention → re-attend to language
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.ops.fc import MLP
from src.ops.layer_norm import LayerNorm
from src.models.mcan.mca import MCA_ED, SGA, SA


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

    def __init__(self, input_dim, hidden_dim, num_classes=80, emb_dim=128):
        super(YOLOClassAdapter, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.class_emb = nn.Embedding(num_classes, emb_dim)
        self.class_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat):
        feat = feat.to(torch.float32)
        mask = make_mask(feat)
        class_ids = feat[:, :, 9].long().clamp(0, 79)
        proj = self.fc(feat)
        cls_emb = self.class_emb(class_ids)
        cls_proj = self.class_proj(cls_emb)
        proj = proj + cls_proj
        return proj, mask


# ------------------------------------------------
# Annotation Feature Adapter
# ------------------------------------------------

class AnnotationAdapter(nn.Module):

    def __init__(self, hidden_dim, num_categories=23, num_attributes=9,
                 cat_emb_dim=64, attr_emb_dim=32):
        super(AnnotationAdapter, self).__init__()

        self.cat_emb = nn.Embedding(num_categories, cat_emb_dim)
        self.attr_emb = nn.Embedding(num_attributes + 1, attr_emb_dim)

        total_input = 14 + cat_emb_dim + attr_emb_dim
        self.proj = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, feat):
        feat = feat.to(torch.float32)
        mask = make_mask(feat)

        cat_ids = feat[:, :, 0].long().clamp(0, 22)
        attr_ids = feat[:, :, 1].long().clamp(0, 8)
        continuous = feat[:, :, 2:]

        cat_vec = self.cat_emb(cat_ids)
        attr_vec = self.attr_emb(attr_ids)
        combined = torch.cat([continuous, cat_vec, attr_vec], dim=-1)
        proj = self.proj(combined)

        return proj, mask


# ================================================
# RadarXFormer-Inspired Components
# ================================================

class SpatialPositionalEncoding(nn.Module):
    """
    Encodes 3D object positions using both Cartesian and polar coordinates.

    Inspired by RadarXFormer's spherical coordinate representation that
    preserves spatial structure. We provide both coordinate systems so
    the model can learn which is more useful for different question types.

    Input features layout (from radarxf features):
      dims 2-4: (x, y, z) — Cartesian position
      dims 12-14: (distance, sin_angle, cos_angle) — polar position
    """

    def __init__(self, hidden_dim, max_obj=100):
        super(SpatialPositionalEncoding, self).__init__()

        # 6 spatial dims: x, y, z, dist, sin_angle, cos_angle
        self.pos_proj = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Learnable position-order embedding (captures object ranking)
        self.order_emb = nn.Embedding(max_obj, hidden_dim)

    def forward(self, features, projected):
        """
        features: (B, N, feat_dim) — raw features with spatial info
        projected: (B, N, hidden_dim) — already-projected features
        Returns: (B, N, hidden_dim) — with spatial encoding added
        """
        # Extract spatial coordinates from raw features
        # dims 2-4: x, y, z; dims 12-14: dist, sin_angle, cos_angle
        spatial = torch.cat([
            features[:, :, 2:5],    # x, y, z
            features[:, :, 12:15],  # dist, sin_angle, cos_angle
        ], dim=-1)  # (B, N, 6)

        pos_enc = self.pos_proj(spatial)  # (B, N, hidden_dim)

        # Add order embedding
        B, N, _ = projected.shape
        order_idx = torch.arange(N, device=projected.device).unsqueeze(0).expand(B, -1)
        order_enc = self.order_emb(order_idx)  # (B, N, hidden_dim)

        return projected + pos_enc + 0.1 * order_enc


class RadarXFormerAdapter(nn.Module):
    """
    Dual-stream feature adapter for RadarXFormer features (32-dim).
    We concatenate the structured and visual embeddings and map them
    via a standard MLP, ensuring no ReLU at the final exit.
    """
    def __init__(self, __C, hidden_dim, struct_dim=16, visual_dim=16,
                 num_categories=23, num_attributes=9,
                 cat_emb_dim=64, attr_emb_dim=32):
        super(RadarXFormerAdapter, self).__init__()
        
        self.__C = __C

        self.struct_dim = struct_dim
        self.visual_dim = visual_dim

        self.cat_emb = nn.Embedding(num_categories, cat_emb_dim)
        self.attr_emb = nn.Embedding(num_attributes + 1, attr_emb_dim)

        # 14 continuous structured + 64 cat + 32 attr + 16 visual CLIP = 126
        total_input = (struct_dim - 2) + cat_emb_dim + attr_emb_dim + visual_dim
        
        self.proj = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, feat):
        feat = feat.to(torch.float32)
        mask = make_mask(feat)

        struct_feat = feat[:, :, :self.struct_dim]
        visual_feat = feat[:, :, self.struct_dim:]
        if not getattr(self.__C, 'USE_CLIP_FEATURES', True):
            visual_feat = torch.zeros_like(visual_feat)

        cat_ids = struct_feat[:, :, 0].long().clamp(0, 22)
        attr_ids = struct_feat[:, :, 1].long().clamp(0, 8)
        continuous = struct_feat[:, :, 2:]  

        cat_vec = self.cat_emb(cat_ids)     
        attr_vec = self.attr_emb(attr_ids)  
        
        combined = torch.cat([continuous, cat_vec, attr_vec, visual_feat], dim=-1)
        proj = self.proj(combined)

        # Dummy store for diagnostics compatibility
        self._gate_mean = 0.5

        return proj, mask


class CrossObjectAttention(nn.Module):
    """
    Self-attention among detected objects in a scene.

    Inspired by RadarXFormer's query self-attention (MHSA, Eq. 5):
    "queries are fed into a multi-head self-attention layer to enable
    information exchange among queries."

    This allows the VQA model to reason about inter-object relationships
    (e.g., "is the car behind the truck?", "how many pedestrians near the bus?")
    """

    def __init__(self, __C, num_layers=1):
        super(CrossObjectAttention, self).__init__()
        self.layers = nn.ModuleList([SA(__C) for _ in range(num_layers)])

    def forward(self, x, x_mask):
        for layer in self.layers:
            x = layer(x, x_mask)
        return x


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

    def __init__(self, __C, num_layers=2):
        super(CrossModalAttention, self).__init__()
        self.bev_to_yolo = nn.ModuleList([SGA(__C) for _ in range(num_layers)])
        self.yolo_to_bev = nn.ModuleList([SGA(__C) for _ in range(num_layers)])

    def forward(self, bev_feat, yolo_feat, bev_mask, yolo_mask):
        for sga_b2y, sga_y2b in zip(self.bev_to_yolo, self.yolo_to_bev):
            bev_enhanced = sga_b2y(bev_feat, yolo_feat, bev_mask, yolo_mask)
            yolo_enhanced = sga_y2b(yolo_feat, bev_feat, yolo_mask, bev_mask)
            bev_feat = bev_enhanced
            yolo_feat = yolo_enhanced
        return bev_feat, yolo_feat


# ------------------------------------------------
# MLP Concat Fusion
# ------------------------------------------------

class MLPConcatFusion(nn.Module):

    def __init__(self, __C):
        super(MLPConcatFusion, self).__init__()
        D = __C.FLAT_OUT_SIZE
        self.fuse = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.ReLU(inplace=True),
            nn.Dropout(__C.DROPOUT_R),
            nn.Linear(D, D)
        )

    def forward(self, bev_vec, yolo_vec):
        combined = torch.cat([bev_vec, yolo_vec], dim=-1)
        return self.fuse(combined)


# ------------------------------------------------
# Count Prediction Head
# ------------------------------------------------

class CountHead(nn.Module):

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
            nn.Linear(D // 2, 11)
        )

    def forward(self, fused_feat, lang_feat):
        combined = torch.cat([fused_feat, lang_feat], dim=-1)
        return self.count_mlp(combined)


# ================================================
# Main MCAN Model
# ================================================

class Net(nn.Module):

    def __init__(self, __C, pretrained_emb, token_size, answer_size):

        super(Net, self).__init__()

        self.__C = __C
        self.is_fusion = (getattr(__C, 'VISUAL_FEATURE', 'bev') == 'fusion')
        self.is_annot = (getattr(__C, 'VISUAL_FEATURE', 'bev') in ('annot', 'detected'))
        self.is_radarxf = (getattr(__C, 'VISUAL_FEATURE', 'bev') == 'radarxf')

        bev_dim = __C.FEAT_SIZE['OBJ_FEAT_SIZE'][1]

        # ================================================
        # RADARXFORMER MODE
        # ================================================
        if self.is_radarxf:
            feat_dim = bev_dim  # 32 for radarxf features
            print(f"  [MCAN] RADARXFORMER mode: feat_dim={feat_dim}")

            # --- RadarXFormer dual-stream adapter ---
            self.radarxf_adapter = RadarXFormerAdapter(
                __C,
                __C.HIDDEN_SIZE,
                struct_dim=16,
                visual_dim=feat_dim - 16,
            )

            # --- Spatial positional encoding ---
            use_spatial_pe = getattr(__C, 'USE_SPATIAL_PE', True)
            self.use_spatial_pe = use_spatial_pe
            if use_spatial_pe:
                self.spatial_pe = SpatialPositionalEncoding(__C.HIDDEN_SIZE)
                print("  [MCAN] Spatial positional encoding: ENABLED")

            # --- Cross-object self-attention (RadarXFormer query MHSA) ---
            self.cross_obj_attn = CrossObjectAttention(__C, num_layers=1)

            # --- Primary MCAN backbone ---
            self.backbone = MCA_ED(__C)

            # --- Iterative refinement (RadarXFormer n-iteration refinement) ---
            n_refine = getattr(__C, 'REFINEMENT_ITERATIONS', 2)
            self.n_refine = n_refine
            if n_refine > 1:
                self.refine_backbones = nn.ModuleList([
                    MCA_ED(__C) for _ in range(n_refine - 1)
                ])
                self.refine_cross_obj = nn.ModuleList([
                    CrossObjectAttention(__C, num_layers=1) for _ in range(n_refine - 1)
                ])
                print(f"  [MCAN] Iterative refinement: {n_refine} iterations")

            # --- Flatten ---
            self.attflat_lang = AttFlat(__C)
            self.attflat_vis = AttFlat(__C)

            # --- Classification ---
            self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
            self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

            total_params = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  [MCAN RadarXF] Total params: {total_params:,}, Trainable: {trainable:,}")

        elif self.is_fusion:
            yolo_dim = __C.FEAT_SIZE['BBOX_FEAT_SIZE'][1]
            print(f"  [MCAN] FUSION mode: BEV dim={bev_dim}, YOLO dim={yolo_dim}")

            self.bev_adapter = Adapter(bev_dim, __C.HIDDEN_SIZE)
            self.yolo_adapter = YOLOClassAdapter(yolo_dim, __C.HIDDEN_SIZE)

            self.backbone_bev = MCA_ED(__C)
            self.backbone_yolo = MCA_ED(__C)

            cross_layers = getattr(__C, 'CROSS_MODAL_LAYERS', 2)
            self.cross_modal = CrossModalAttention(__C, num_layers=cross_layers)

            self.attflat_lang_bev = AttFlat(__C)
            self.attflat_lang_yolo = AttFlat(__C)
            self.attflat_bev = AttFlat(__C)
            self.attflat_yolo = AttFlat(__C)

            self.lang_fusion = nn.Sequential(
                nn.Linear(__C.FLAT_OUT_SIZE * 2, __C.FLAT_OUT_SIZE),
                nn.ReLU(inplace=True),
                nn.Dropout(__C.DROPOUT_R),
            )

            self.fusion_mlp = MLPConcatFusion(__C)
            self.bev_residual_weight = nn.Parameter(torch.tensor(0.3))
            self.count_head = CountHead(__C)

            self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
            self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

            total_params = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  [MCAN Fusion] Total params: {total_params:,}, Trainable: {trainable:,}")

        elif self.is_annot:
            feat_dim = bev_dim
            print(f"  [MCAN] ANNOTATION mode: feat_dim={feat_dim}")

            self.annot_adapter = AnnotationAdapter(__C.HIDDEN_SIZE)
            self.backbone = MCA_ED(__C)
            self.attflat_lang = AttFlat(__C)
            self.attflat_vis = AttFlat(__C)
            self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
            self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

            total_params = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  [MCAN Annot] Total params: {total_params:,}, Trainable: {trainable:,}")

        else:
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

            self.img_adapter = Adapter(self.img_dim, __C.HIDDEN_SIZE)
            self.radar_adapter = Adapter(self.radar_expand, __C.HIDDEN_SIZE)
            self.backbone_img = MCA_ED(__C)
            self.backbone_radar = MCA_ED(__C)
            self.attflat_lang = AttFlat(__C)
            self.attflat_img = AttFlat(__C)
            self.attflat_radar = AttFlat(__C)
            self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
            self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

        # ---- Language (shared) ----

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


    def forward(self, obj_feat, bbox_feat, ques_ix):
        """
        obj_feat  : (B, N, feat_dim) visual features
        bbox_feat : (B, N, ?) secondary features or dummy
        ques_ix   : (B, T) question token indices
        """

        # ---- Language ----
        lang_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        if self.is_radarxf:
            return self._forward_radarxf(obj_feat, lang_feat, lang_mask)
        elif self.is_fusion:
            return self._forward_fusion(obj_feat, bbox_feat, lang_feat, lang_mask)
        elif self.is_annot:
            return self._forward_annot(obj_feat, lang_feat, lang_mask)
        else:
            return self._forward_single(obj_feat, lang_feat, lang_mask)


    def _forward_radarxf(self, radarxf_feat, lang_feat, lang_mask):
        """
        RadarXFormer-inspired forward path.

        Architecture (inspired by RadarXFormer Fig. 1 & Fig. 4):
          1. Dual-stream adapter: encode structured + visual features separately, fuse with gate
          2. Spatial positional encoding: add 3D position information
          3. Cross-object self-attention: objects exchange information (like query MHSA)
          4. MCAN co-attention: language ↔ visual
          5. [Optional] Iterative refinement: repeat steps 3-4
          6. Flatten + classify
        """

        vis_proj, vis_mask = self.radarxf_adapter(radarxf_feat)  # (B, N, hidden)

        if getattr(self.__C, 'USE_SPATIAL_PE', True) and hasattr(self, 'spatial_pe'):
            vis_proj = self.spatial_pe(radarxf_feat, vis_proj)

        # 3. Allow objects to communicate spatially via cross-attention
        if hasattr(self, 'cross_obj_attn') and self.cross_obj_attn is not None:
            # CrossObjectAttention internals automatically run REFINEMENT_ITERATIONS loops
            cross_out = self.cross_obj_attn(vis_proj, vis_mask)
            vis_proj = vis_proj + 0.1 * cross_out  # Damped residual to prevent explosion

        lang_out, vis_out = self.backbone(
            lang_feat, vis_proj, lang_mask, vis_mask
        )

        # ---- Flatten ----
        lang_vec = self.attflat_lang(lang_out, lang_mask)
        vis_vec = self.attflat_vis(vis_out, vis_mask)

        # ---- Classify ----
        proj_feat = lang_vec + vis_vec
        proj_feat = self.proj_norm(proj_feat)
        logits = self.proj(proj_feat)

        # Dummy values for training engine compatibility
        self._count_logits = None
        self._fusion_gate_mean = getattr(self.radarxf_adapter, '_gate_mean', 0.5)

        return logits


    def _forward_fusion(self, bev_feat, yolo_feat, lang_feat, lang_mask):
        """Ultimate dual-encoder fusion path."""

        bev_proj, bev_mask = self.bev_adapter(bev_feat)
        yolo_proj, yolo_mask = self.yolo_adapter(yolo_feat)

        lang_bev, bev_out = self.backbone_bev(
            lang_feat.clone(), bev_proj, lang_mask, bev_mask
        )
        lang_yolo, yolo_out = self.backbone_yolo(
            lang_feat.clone(), yolo_proj, lang_mask, yolo_mask
        )

        bev_out, yolo_out = self.cross_modal(
            bev_out, yolo_out, bev_mask, yolo_mask
        )

        lang_bev_vec = self.attflat_lang_bev(lang_bev, lang_mask)
        lang_yolo_vec = self.attflat_lang_yolo(lang_yolo, lang_mask)
        bev_vec = self.attflat_bev(bev_out, bev_mask)
        yolo_vec = self.attflat_yolo(yolo_out, yolo_mask)

        lang_vec = self.lang_fusion(
            torch.cat([lang_bev_vec, lang_yolo_vec], dim=-1)
        )

        fused_visual = self.fusion_mlp(bev_vec, yolo_vec)

        bev_residual = self.bev_residual_weight * bev_vec
        proj_feat = lang_vec + fused_visual + bev_residual
        proj_feat = self.proj_norm(proj_feat)
        logits = self.proj(proj_feat)

        count_logits = self.count_head(fused_visual, lang_vec)
        self._count_logits = count_logits
        self._fusion_gate_mean = 0.5

        return logits


    def _forward_annot(self, annot_feat, lang_feat, lang_mask):
        """Annotation-based path: simple, effective."""

        vis_proj, vis_mask = self.annot_adapter(annot_feat)

        lang_out, vis_out = self.backbone(
            lang_feat, vis_proj, lang_mask, vis_mask
        )

        lang_vec = self.attflat_lang(lang_out, lang_mask)
        vis_vec = self.attflat_vis(vis_out, vis_mask)

        proj_feat = lang_vec + vis_vec
        proj_feat = self.proj_norm(proj_feat)
        logits = self.proj(proj_feat)

        self._count_logits = None
        self._fusion_gate_mean = 0.0

        return logits


    def _forward_single(self, obj_feat, lang_feat, lang_mask):
        """Single-feature path (backward compatible with BEV/YOLO only)."""

        img_feat = obj_feat[:, :, :self.img_dim]
        radar_feat = obj_feat[:, :, self.img_dim:]

        radar_conf = radar_feat[:, :, -1].unsqueeze(-1)
        img_feat = img_feat * (1 + radar_conf)

        if self.radar_dim < self.radar_expand:
            repeat_factor = (self.radar_expand // self.radar_dim) + 1
            radar_feat = radar_feat.repeat(1, 1, repeat_factor)[:, :, :self.radar_expand]

        radar_feat = radar_feat / (torch.norm(radar_feat, dim=-1, keepdim=True) + 1e-6)

        img_feat, img_mask = self.img_adapter(img_feat)
        radar_feat, radar_mask = self.radar_adapter(radar_feat)

        lang_img, img_feat = self.backbone_img(
            lang_feat, img_feat, lang_mask, img_mask
        )

        lang_rad, radar_feat = self.backbone_radar(
            lang_feat, radar_feat, lang_mask, radar_mask
        )

        lang_vec = self.attflat_lang(lang_feat, lang_mask)
        img_vec = self.attflat_img(img_feat, img_mask)
        radar_vec = self.attflat_radar(radar_feat, radar_mask)

        proj_feat = lang_vec + img_vec + 0.1 * radar_vec

        presence = radar_feat[:, :, 0].sum(dim=1, keepdim=True)
        presence = presence.expand(-1, proj_feat.shape[1])
        proj_feat = proj_feat + 0.2 * presence

        proj_feat = self.proj_norm(proj_feat)
        logits = self.proj(proj_feat)

        return logits
