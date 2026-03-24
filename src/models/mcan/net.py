# # ------------------------------------------------------------------
# # Modified for Image + Radar BEV features (80 × 69)
# # Based on NuScenes-QA MCAN implementation 
# # ------------------------------------------------------------------

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from src.ops.fc import MLP
# from src.ops.layer_norm import LayerNorm
# from src.models.mcan.mca import MCA_ED


# # ------------------------------
# # ---- Utility: Mask creator ---
# # ------------------------------

# def make_mask(feature):
#     """
#     feature: (B, N, D)
#     returns: (B, 1, 1, N)
#     """
#     return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)


# # ------------------------------
# # ---- Adapter (Visual Input) --
# # ------------------------------

# class Adapter(nn.Module):
#     """
#     Projects BEV object features (69-dim) → MCAN hidden size (512)
#     """
#     def __init__(self, __C):
#         super(Adapter, self).__init__()
#         self.__C = __C

#         obj_dim = __C.FEAT_SIZE['OBJ_FEAT_SIZE'][1]  # = 69
#         self.obj_linear = nn.Linear(obj_dim, __C.HIDDEN_SIZE)

#     def forward(self, obj_feat):
#         """
#         obj_feat: (B, 80, 69)
#         """
#         obj_feat = obj_feat.to(torch.float32)
#         obj_feat_mask = make_mask(obj_feat)
#         obj_feat = self.obj_linear(obj_feat)
#         return obj_feat, obj_feat_mask


# # ------------------------------
# # ---- Attention Flattening ----
# # ------------------------------

# class AttFlat(nn.Module):
#     def __init__(self, __C):
#         super(AttFlat, self).__init__()
#         self.__C = __C

#         self.mlp = MLP(
#             in_size=__C.HIDDEN_SIZE,
#             mid_size=__C.FLAT_MLP_SIZE,
#             out_size=__C.FLAT_GLIMPSES,
#             dropout_r=__C.DROPOUT_R,
#             use_relu=True
#         )

#         self.linear_merge = nn.Linear(
#             __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
#             __C.FLAT_OUT_SIZE
#         )

#     def forward(self, x, x_mask):
#         att = self.mlp(x)
#         att = att.masked_fill(
#             x_mask.squeeze(1).squeeze(1).unsqueeze(2),
#             -1e4
#         )
#         att = F.softmax(att, dim=1)

#         att_list = []
#         for i in range(self.__C.FLAT_GLIMPSES):
#             att_list.append(torch.sum(att[:, :, i:i+1] * x, dim=1))

#         x_atted = torch.cat(att_list, dim=1)
#         x_atted = self.linear_merge(x_atted)
#         return x_atted


# # -------------------------
# # ---- Main MCAN Model ----
# # -------------------------

# class Net(nn.Module):
#     def __init__(self, __C, pretrained_emb, token_size, answer_size):
#         super(Net, self).__init__()
#         self.__C = __C

#         # -------- Language embedding --------
#         self.embedding = nn.Embedding(
#             num_embeddings=token_size,
#             embedding_dim=__C.WORD_EMBED_SIZE
#         )
#         self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

#         self.lstm = nn.LSTM(
#             input_size=__C.WORD_EMBED_SIZE,
#             hidden_size=__C.HIDDEN_SIZE,
#             num_layers=1,
#             batch_first=True
#         )

#         # -------- Visual adapter --------
#         self.adapter = Adapter(__C)

#         # -------- Co-attention backbone --------
#         self.backbone = MCA_ED(__C)

#         # -------- Flatten layers --------
#         self.attflat_lang = AttFlat(__C)
#         self.attflat_img = AttFlat(__C)

#         # -------- Classification head --------
#         self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
#         self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

#     def forward(self, obj_feat, bbox_feat, ques_ix):
#         """
#         obj_feat: (B, 80, 69)
#         ques_ix:  (B, T)
#         """

#         # ---- Language ----
#         lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
#         lang_feat = self.embedding(ques_ix)
#         lang_feat, _ = self.lstm(lang_feat)

#         # ---- Visual ----
#         obj_feat, obj_feat_mask = self.adapter(obj_feat)

#         # ---- Co-attention ----
#         lang_feat, obj_feat = self.backbone(
#             lang_feat,
#             obj_feat,
#             lang_feat_mask,
#             obj_feat_mask
#         )

#         # ---- Flatten ----
#         lang_feat = self.attflat_lang(lang_feat, lang_feat_mask)
#         obj_feat = self.attflat_img(obj_feat, obj_feat_mask)

#         # ---- Fuse & classify ----
#         proj_feat = lang_feat + obj_feat
#         proj_feat = self.proj_norm(proj_feat)
#         logits = self.proj(proj_feat)

#         return logits




# ------------------------------------------------------------------
# Modified for Image + Radar BEV features (80 × 69)
# Late Fusion MCAN Architecture
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
        # att = torch.sigmoid(att) # each object gets independent weight

        att_list = []

        for i in range(self.__C.FLAT_GLIMPSES):

            att_list.append(
                torch.sum(att[:, :, i:i+1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)

        # obj_count_feat = torch.sum(att, dim=1) # how many objects received attention

        x_atted = self.linear_merge(x_atted)

        # x_atted = torch.cat([x_atted, obj_count_feat], dim=1)

        return x_atted


# ------------------------------------------------
# Main MCAN Model
# ------------------------------------------------

class Net(nn.Module):

    def __init__(self, __C, pretrained_emb, token_size, answer_size):

        super(Net, self).__init__()

        self.__C = __C

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

        # ---------------- Visual Adapters ----------------

        # Image = 64 dims
        self.img_adapter = Adapter(64, __C.HIDDEN_SIZE)

        # Radar = 5 dims
        self.radar_adapter = Adapter(60, __C.HIDDEN_SIZE)

        # ---------------- MCAN backbone ----------------

        self.backbone_img = MCA_ED(__C)
        self.backbone_radar = MCA_ED(__C)

        # ---------------- Flatten ----------------

        self.attflat_lang = AttFlat(__C)

        self.attflat_img = AttFlat(__C)

        self.attflat_radar = AttFlat(__C)

        # ---------------- Fusion Head ----------------

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        # self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE + __C.FLAT_GLIMPSES)

        self.proj = nn.Linear(
            __C.FLAT_OUT_SIZE,
            answer_size
        )
        # self.proj = nn.Linear(
        #     __C.FLAT_OUT_SIZE + __C.FLAT_GLIMPSES,
        #     answer_size
        # )


    def forward(self, obj_feat, bbox_feat, ques_ix):

        """
        obj_feat : (B,80,69)
        """

        # ------------------------------------------------
        # Language
        # ------------------------------------------------

        lang_mask = make_mask(ques_ix.unsqueeze(2))

        lang_feat = self.embedding(ques_ix)

        lang_feat, _ = self.lstm(lang_feat)

        # ------------------------------------------------
        # Split Image + Radar
        # ------------------------------------------------

        img_feat = obj_feat[:, :, :64]

        radar_feat = obj_feat[:, :, 64:]

        radar_conf = radar_feat[:, :, -1].unsqueeze(-1) # radar object confidence
        # object_count = radar_conf.sum(dim=1)
        # object_count = object_count.unsqueeze(-1)
        radar_feat = radar_feat.repeat(1,1,12)[:, :, :60]
        img_feat = img_feat * (1 + radar_conf) # weight image features using radar

        radar_feat = radar_feat / (torch.norm(radar_feat, dim=-1, keepdim=True) + 1e-6)

        # ------------------------------------------------
        # Visual adapters
        # ------------------------------------------------

        img_feat, img_mask = self.img_adapter(img_feat)

        radar_feat, radar_mask = self.radar_adapter(radar_feat)

        # ------------------------------------------------
        # MCAN for Image
        # ------------------------------------------------

        lang_img, img_feat = self.backbone_img(
            lang_feat,
            img_feat,
            lang_mask,
            img_mask
        )

        # ------------------------------------------------
        # MCAN for Radar
        # ------------------------------------------------

        lang_rad, radar_feat = self.backbone_radar(
            lang_feat,
            radar_feat,
            lang_mask,
            radar_mask
        )

        # ------------------------------------------------
        # Flatten
        # ------------------------------------------------

        lang_vec = self.attflat_lang(lang_feat, lang_mask)

        img_vec = self.attflat_img(img_feat, img_mask)

        radar_vec = self.attflat_radar(radar_feat, radar_mask)

        # ------------------------------------------------
        # Late Fusion
        # ------------------------------------------------
        # radar_strength = radar_feat[:,:,-1].mean(dim=1, keepdim=True)
        proj_feat = lang_vec + img_vec + 0.1 * radar_vec  

        presence = radar_feat[:, :, 0].sum(dim=1, keepdim=True)
        presence = presence.expand(-1,proj_feat.shape[1])
        proj_feat = proj_feat + 0.2*presence  

        # proj_feat = lang_vec + img_vec + radar_vec

        proj_feat = self.proj_norm(proj_feat)

        logits = self.proj(proj_feat)

        return logits

