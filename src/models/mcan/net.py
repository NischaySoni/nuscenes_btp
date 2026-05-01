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
        self.is_centerpoint_fusion = (getattr(__C, 'VISUAL_FEATURE', 'bev') == 'centerpoint_fusion')
        self.is_centerpoint_only = (getattr(__C, 'VISUAL_FEATURE', 'bev') == 'centerpoint_only')
        self.is_radarxf_fusion = (getattr(__C, 'VISUAL_FEATURE', 'bev') in ('radarxf_fusion', 'trimodal_fusion', 'centerpoint_fusion'))
        self.is_trimodal_fusion = (getattr(__C, 'VISUAL_FEATURE', 'bev') == 'trimodal_fusion')
        self.is_annot = (getattr(__C, 'VISUAL_FEATURE', 'bev') in ('annot', 'detected'))
        self.is_radarxf = (getattr(__C, 'VISUAL_FEATURE', 'bev') == 'radarxf')

        bev_dim = __C.FEAT_SIZE['OBJ_FEAT_SIZE'][1]

        # ================================================
        # RADARXFORMER MODE
        # ================================================
        if self.is_centerpoint_only:
            # ================================================
            # CENTERPOINT-ONLY MODE (matches official paper EXACTLY)
            # Official Adapter: bbox_linear(7→2048), cat(512,2048)=2560, obj_linear(2560→512)
            # ================================================
            obj_dim = bev_dim  # 512
            bbox_dim = __C.FEAT_SIZE['BBOX_FEAT_SIZE'][1]  # 7
            use_bbox = getattr(__C, 'USE_BBOX_FEAT', False)
            bbox_emb_size = getattr(__C, 'BBOXFEAT_EMB_SIZE', 2048)

            obj_feat_linear_size = obj_dim
            if use_bbox:
                self.bbox_linear = nn.Linear(bbox_dim, bbox_emb_size)
                obj_feat_linear_size += bbox_emb_size  # 512 + 2048 = 2560

            self.obj_linear = nn.Linear(obj_feat_linear_size, __C.HIDDEN_SIZE)

            self._use_bbox = use_bbox
            print(f"  [MCAN] CENTERPOINT_ONLY mode: obj_dim={obj_dim}, bbox_dim={bbox_dim}, "
                  f"use_bbox={use_bbox}, bbox_emb={bbox_emb_size}, "
                  f"adapter: {obj_feat_linear_size}→{__C.HIDDEN_SIZE}")

            # Single MCAN backbone
            self.backbone = MCA_ED(__C)

            # Flatten
            self.attflat_lang = AttFlat(__C)
            self.attflat_vis = AttFlat(__C)

            # Classify
            self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
            self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

            total_params = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  [MCAN CenterPoint] Total params: {total_params:,}, Trainable: {trainable:,}")

        elif self.is_radarxf:
            feat_dim = bev_dim  # 48 for radarxf v2 features
            print(f"  [MCAN] RADARXFORMER mode: feat_dim={feat_dim}")

            # --- RadarXFormer adapter (same architecture as AnnotationAdapter) ---
            self.radarxf_adapter = RadarXFormerAdapter(
                __C,
                __C.HIDDEN_SIZE,
                struct_dim=16,
                visual_dim=feat_dim - 16,
            )

            # --- Primary MCAN backbone (same as annotation model) ---
            self.backbone = MCA_ED(__C)

            # --- Flatten ---
            self.attflat_lang = AttFlat(__C)
            self.attflat_vis = AttFlat(__C)

            # --- Classification ---
            self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
            self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

            total_params = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  [MCAN RadarXF] Total params: {total_params:,}, Trainable: {trainable:,}")

        elif self.is_radarxf_fusion:
            # ================================================
            # RADARXF + BEV FUSION MODE
            # ================================================
            rxf_dim = __C.FEAT_SIZE['BBOX_FEAT_SIZE'][1]  # 48
            mode_name = 'CENTERPOINT_FUSION' if self.is_centerpoint_fusion else ('TRIMODAL_FUSION' if self.is_trimodal_fusion else 'RADARXF_FUSION')
            print(f"  [MCAN] {mode_name} mode: BEV dim={bev_dim}, RadarXF dim={rxf_dim}")

            # Adapters
            self.bev_adapter = Adapter(bev_dim, __C.HIDDEN_SIZE)
            self.radarxf_adapter = RadarXFormerAdapter(
                __C, __C.HIDDEN_SIZE,
                struct_dim=16,
                visual_dim=rxf_dim - 16,
            )

            # Dual backbones
            self.backbone_bev = MCA_ED(__C)
            self.backbone_rxf = MCA_ED(__C)

            # Cross-modal attention
            cross_layers = getattr(__C, 'CROSS_MODAL_LAYERS', 2)
            self.cross_modal = CrossModalAttention(__C, num_layers=cross_layers)

            # Flatten
            self.attflat_lang_bev = AttFlat(__C)
            self.attflat_lang_rxf = AttFlat(__C)
            self.attflat_bev = AttFlat(__C)
            self.attflat_rxf = AttFlat(__C)

            # Lang fusion
            self.lang_fusion = nn.Sequential(
                nn.Linear(__C.FLAT_OUT_SIZE * 2, __C.FLAT_OUT_SIZE),
                nn.ReLU(inplace=True),
                nn.Dropout(__C.DROPOUT_R),
            )

            # Visual fusion
            self.fusion_mlp = MLPConcatFusion(__C)

            # Classify
            self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
            self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

            # Dedicated count head (11 classes for counts 0-10)
            self.count_head = nn.Sequential(
                nn.Linear(__C.FLAT_OUT_SIZE, __C.FLAT_OUT_SIZE // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(__C.DROPOUT_R),
                nn.Linear(__C.FLAT_OUT_SIZE // 2, 11),
            )

            total_params = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  [MCAN RadarXF Fusion] Total params: {total_params:,}, Trainable: {trainable:,}")

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

        if self.is_centerpoint_only:
            return self._forward_centerpoint(obj_feat, bbox_feat, lang_feat, lang_mask)
        elif self.is_radarxf:
            return self._forward_radarxf(obj_feat, lang_feat, lang_mask)
        elif self.is_radarxf_fusion:
            return self._forward_radarxf_fusion(obj_feat, bbox_feat, lang_feat, lang_mask)
        elif self.is_fusion:
            return self._forward_fusion(obj_feat, bbox_feat, lang_feat, lang_mask)
        elif self.is_annot:
            return self._forward_annot(obj_feat, lang_feat, lang_mask)
        else:
            return self._forward_single(obj_feat, lang_feat, lang_mask)


    def _forward_centerpoint(self, obj_feat, bbox_feat, lang_feat, lang_mask):
        """
        CenterPoint-only forward path (matches official NuScenes-QA paper EXACTLY).
        Official Adapter: bbox_linear(7→2048), cat(512,2048)=2560, obj_linear(2560→512).
        """
        obj_feat = obj_feat.to(torch.float32)
        bbox_feat = bbox_feat.to(torch.float32)

        # Create mask from raw object features (before any projection)
        vis_mask = make_mask(obj_feat)

        # Official adapter: concatenate bbox embedding with obj features, then project
        if self._use_bbox:
            bbox_emb = self.bbox_linear(bbox_feat)        # (B, N, 7) → (B, N, 2048)
            obj_feat = torch.cat((obj_feat, bbox_emb), dim=-1)  # (B, N, 2560)

        vis_feat = self.obj_linear(obj_feat)              # (B, N, 2560) → (B, N, 512)

        # MCAN encoder
        lang_out, vis_out = self.backbone(
            lang_feat, vis_feat, lang_mask, vis_mask
        )

        # Flatten
        lang_vec = self.attflat_lang(lang_out, lang_mask)
        vis_vec = self.attflat_vis(vis_out, vis_mask)

        # Classify
        proj_feat = lang_vec + vis_vec
        proj_feat = self.proj_norm(proj_feat)
        logits = self.proj(proj_feat)

        # Compatibility with training engine
        self._count_logits = None
        self._fusion_gate_mean = 0.5

        return logits


    def _forward_radarxf(self, radarxf_feat, lang_feat, lang_mask):
        """
        RadarXFormer-inspired forward path.
        Simplified to match the proven Annotation model architecture.
        """

        vis_proj, vis_mask = self.radarxf_adapter(radarxf_feat)

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


    def _forward_radarxf_fusion(self, bev_feat, rxf_feat, lang_feat, lang_mask):
        """BEV + RadarXFormer Dual-Encoder Fusion path."""

        # Adapt both modalities
        bev_proj, bev_mask = self.bev_adapter(bev_feat)
        rxf_proj, rxf_mask = self.radarxf_adapter(rxf_feat)

        # Independent language-visual co-attention
        lang_bev, bev_out = self.backbone_bev(
            lang_feat.clone(), bev_proj, lang_mask, bev_mask
        )
        lang_rxf, rxf_out = self.backbone_rxf(
            lang_feat.clone(), rxf_proj, lang_mask, rxf_mask
        )

        # Cross-modal attention between modalities
        bev_out, rxf_out = self.cross_modal(
            bev_out, rxf_out, bev_mask, rxf_mask
        )

        # Flatten
        lang_bev_vec = self.attflat_lang_bev(lang_bev, lang_mask)
        lang_rxf_vec = self.attflat_lang_rxf(lang_rxf, lang_mask)
        bev_vec = self.attflat_bev(bev_out, bev_mask)
        rxf_vec = self.attflat_rxf(rxf_out, rxf_mask)

        # Fuse language streams
        lang_vec = self.lang_fusion(
            torch.cat([lang_bev_vec, lang_rxf_vec], dim=-1)
        )

        # Fuse visual streams
        fused_visual = self.fusion_mlp(bev_vec, rxf_vec)

        # Classify
        proj_feat = lang_vec + fused_visual
        proj_feat = self.proj_norm(proj_feat)
        logits = self.proj(proj_feat)

        # Dedicated count head
        self._count_logits = self.count_head(proj_feat)
        self._fusion_gate_mean = 0.5

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
