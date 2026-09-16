"""DeCAPS-Net: multimodal fusion model for ASD assessment.

Pipeline:
  Parsing branch  : colorized parsing maps [B, T, 3, H, W]
                    -> frame encoder (torchvision backbone)
                    -> temporal Transformer -> frame tokens
  Skeleton branch : 3D skeleton sequence [B, C, T, V, M]
                    -> DeCAPSSkeleton trunk -> temporal skeleton tokens
  Fusion          : bidirectional cross-attention between the two token sets
                    -> fusion Transformer (CLS) -> MLP -> 1 logit

The branch logits (parse_logit, skel_logit) are returned in `aux` when
return_aux=True; ParseBranch and SkeletonBranch can also be used standalone
for single-modality ablations.
"""

import torch
import torch.nn as nn
from torchvision import models as tv_models

from .skeleton.decaps_skeleton import DeCAPSSkeleton


# ---------------------------------------------------------------------------
# Parsing branch
# ---------------------------------------------------------------------------


class FrameEncoder(nn.Module):
    """Pretrained frame encoder for colorized parsing maps.

    Input:  [B, 3, H, W]
    Output: [B, out_dim]
    Supported backbones: resnet18, mobilenet_v3_small, efficientnet_b0,
    vgg16, vit_b_16.
    """

    def __init__(self, out_dim=256, backbone_name="resnet18", pretrained=True, dropout=0.2):
        super().__init__()
        backbone_name = backbone_name.lower()

        if backbone_name == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = tv_models.resnet18(weights=weights)
            feat_dim = backbone.fc.in_features          # 512
            backbone.fc = nn.Identity()

        elif backbone_name == "mobilenet_v3_small":
            weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = tv_models.mobilenet_v3_small(weights=weights)
            feat_dim = backbone.classifier[0].in_features  # 576
            backbone.classifier = nn.Identity()

        elif backbone_name == "efficientnet_b0":
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = tv_models.efficientnet_b0(weights=weights)
            feat_dim = backbone.classifier[-1].in_features  # 1280
            backbone.classifier = nn.Identity()

        elif backbone_name == "vgg16":
            weights = tv_models.VGG16_Weights.DEFAULT if pretrained else None
            backbone = tv_models.vgg16(weights=weights)
            feat_dim = 4096                             # penultimate feature
            backbone.classifier = nn.Sequential(*list(backbone.classifier.children())[:-1])

        elif backbone_name == "vit_b_16":
            weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
            backbone = tv_models.vit_b_16(weights=weights)
            feat_dim = backbone.hidden_dim              # 768
            backbone.heads = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        self.backbone_name = backbone_name
        self.backbone = backbone
        self.feat_dim = feat_dim

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.proj(self.backbone(x))


class TemporalTransformer(nn.Module):
    """Transformer encoder over frame tokens with a CLS token.

    Input:  x [B, T, D]
    Output: cls_out [B, D]; tok_out [B, T, D] if return_tokens=True
    """

    def __init__(self, d_model=256, nhead=4, num_layers=2, dropout=0.1, max_len=256):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_tokens=False):
        B, T, D = x.shape
        cls = self.cls_token.expand(B, 1, D)
        z = torch.cat([cls, x], dim=1)              # [B, T+1, D]
        z = z + self.pos_emb[:, :T + 1, :]
        z = self.encoder(z)

        cls_out = z[:, 0]
        tok_out = z[:, 1:]

        if return_tokens:
            return cls_out, tok_out
        return cls_out


class ParseBranch(nn.Module):
    """Parsing branch: frame encoder + temporal Transformer + binary head.

    Input:  parse_x [B, T, 3, H, W]
    Output: z_parse [B, D], parse_logit [B], and frame tokens [B, T, D]
            when return_tokens=True
    """

    def __init__(self, backbone_name="resnet18", pretrained=True,
                 frame_feat_dim=256, dropout=0.2):
        super().__init__()

        self.frame_encoder = FrameEncoder(
            out_dim=frame_feat_dim,
            backbone_name=backbone_name,
            pretrained=pretrained,
            dropout=dropout,
        )
        self.temporal = TemporalTransformer(
            d_model=frame_feat_dim, nhead=4, num_layers=2, dropout=0.1, max_len=256
        )

        self.parse_feat_dim = frame_feat_dim
        self.token_dim = frame_feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(frame_feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, parse_x, return_tokens=False):
        B, T, C, H, W = parse_x.shape
        x = parse_x.view(B * T, C, H, W)
        f = self.frame_encoder(x).view(B, T, -1)     # [B, T, D]

        if return_tokens:
            z, tokens = self.temporal(f, return_tokens=True)
        else:
            z, tokens = self.temporal(f), None

        parse_logit = self.classifier(z).squeeze(1)  # [B]

        if return_tokens:
            return z, parse_logit, tokens
        return z, parse_logit


# ---------------------------------------------------------------------------
# Skeleton branch
# ---------------------------------------------------------------------------


class SkeletonBranch(nn.Module):
    """Skeleton branch: DeCAPSSkeleton trunk + binary head.

    Input:  skel_x [B, C, T, V, M]  (or [B, T, V*C])
    Output: z_skel [B, 512], skel_logit [B], and temporal tokens
            [B, T', 256] when return_tokens=True
            (stage-2 maps averaged over streams and joints, time kept)
    """

    def __init__(self, num_point=25, num_person=1, k=8, eta=4,
                 graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0.0, tsm_impl="htm"):
        super().__init__()

        self.backbone = DeCAPSSkeleton(
            num_class=1,
            num_point=num_point,
            num_person=num_person,
            k=k,
            eta=eta,
            num_stream=2,
            graph=graph,
            graph_args=graph_args,
            in_channels=in_channels,
            drop_out=drop_out,
            tsm_impl=tsm_impl,
        )

        self.skel_feat_dim = 256 * 2   # two pooled stream features concatenated
        self.skel_token_dim = 256      # stage-2 output channels

    def forward(self, x, return_tokens=False):
        m = self.backbone
        stream_maps, N, M = m.extract_stream_features(x)   # each [N*M, 256, T', V]

        feats = []
        logits = []
        stage2_maps = []
        for y, fc in zip(stream_maps, m.fc):
            c_new, t_new, v_new = y.size(1), y.size(2), y.size(3)
            y_map = y.view(N, M, c_new, t_new, v_new).mean(dim=1)   # [N, 256, T', V]
            stage2_maps.append(y_map)

            feat = y_map.flatten(2).mean(dim=2)                     # [N, 256]
            feat = m.drop_out(feat)
            feats.append(feat)
            logits.append(fc(feat).squeeze(1))                      # [N]

        z_skel = torch.cat(feats, dim=1)                            # [N, 512]
        skel_logit = torch.stack(logits, dim=0).mean(dim=0)         # [N]

        if return_tokens:
            stage2_map = torch.stack(stage2_maps, dim=0).mean(dim=0)     # [N, 256, T', V]
            skel_tokens = stage2_map.mean(dim=-1).transpose(1, 2).contiguous()  # [N, T', 256]
            return z_skel, skel_logit, skel_tokens

        return z_skel, skel_logit


# ---------------------------------------------------------------------------
# Cross-attention fusion
# ---------------------------------------------------------------------------


class FeedForwardBlock(nn.Module):
    def __init__(self, dim, hidden_mult=2.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * hidden_mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional cross-attention: parsing queries skeleton and vice versa."""

    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.parse_to_skel = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.skel_to_parse = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm_p1 = nn.LayerNorm(dim)
        self.norm_s1 = nn.LayerNorm(dim)
        self.ffn_p = FeedForwardBlock(dim=dim, hidden_mult=2.0, dropout=dropout)
        self.ffn_s = FeedForwardBlock(dim=dim, hidden_mult=2.0, dropout=dropout)
        self.norm_p2 = nn.LayerNorm(dim)
        self.norm_s2 = nn.LayerNorm(dim)

    def forward(self, parse_tokens, skel_tokens):
        # keep the originals for symmetric bidirectional attention
        p0 = parse_tokens
        s0 = skel_tokens

        # parsing attends to skeleton
        p_attn, _ = self.parse_to_skel(query=p0, key=s0, value=s0, need_weights=False)
        p = self.norm_p1(p0 + p_attn)
        p = self.norm_p2(p + self.ffn_p(p))

        # skeleton attends to parsing
        s_attn, _ = self.skel_to_parse(query=s0, key=p0, value=p0, need_weights=False)
        s = self.norm_s1(s0 + s_attn)
        s = self.norm_s2(s + self.ffn_s(s))

        return p, s


class FusionTransformerHead(nn.Module):
    """Fusion Transformer over the concatenated fused tokens.

    CLS token + positional embeddings -> Transformer encoder -> CLS -> MLP.
    """

    def __init__(self, d_model=256, nhead=4, num_layers=1, dropout=0.1,
                 max_len=512, cls_hidden=256, cls_dropout=0.2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden, 1),
        )

    def forward(self, tokens):
        # tokens: [B, L, D]
        B, L, D = tokens.shape
        cls = self.cls_token.expand(B, 1, D)
        z = torch.cat([cls, tokens], dim=1)         # [B, L+1, D]
        z = z + self.pos_emb[:, :L + 1, :]
        z = self.encoder(z)

        z_cls = z[:, 0]
        logit = self.classifier(z_cls).squeeze(1)   # [B]
        return z_cls, logit


# ---------------------------------------------------------------------------
# Full fusion model
# ---------------------------------------------------------------------------


class DeCAPSFusion(nn.Module):
    """DeCAPS-Net: parsing branch + skeleton branch + cross-attention fusion.

    forward(parse_x, skel_x) -> logit [B]
    """

    def __init__(
        self,
        # parsing branch
        parse_backbone_name="resnet18",
        parse_pretrained=True,
        parse_frame_feat_dim=256,
        parse_dropout=0.2,
        # skeleton branch
        skel_num_point=25,
        skel_num_person=1,
        skel_k=8,
        skel_eta=4,
        skel_graph=None,
        skel_graph_args=dict(),
        skel_in_channels=3,
        skel_drop_out=0.0,
        skel_tsm_impl="htm",
        # fusion
        xattn_dim=256,
        xattn_heads=4,
        xattn_dropout=0.1,
        fusion_num_layers=1,
        fusion_max_len=512,
        fusion_hidden=256,
        fusion_dropout=0.2,
    ):
        super().__init__()

        self.parse_branch = ParseBranch(
            backbone_name=parse_backbone_name,
            pretrained=parse_pretrained,
            frame_feat_dim=parse_frame_feat_dim,
            dropout=parse_dropout,
        )
        self.skel_branch = SkeletonBranch(
            num_point=skel_num_point,
            num_person=skel_num_person,
            k=skel_k,
            eta=skel_eta,
            graph=skel_graph,
            graph_args=skel_graph_args,
            in_channels=skel_in_channels,
            drop_out=skel_drop_out,
            tsm_impl=skel_tsm_impl,
        )

        self.parse_token_proj = (
            nn.Identity()
            if self.parse_branch.token_dim == xattn_dim
            else nn.Linear(self.parse_branch.token_dim, xattn_dim)
        )
        self.skel_token_proj = (
            nn.Identity()
            if self.skel_branch.skel_token_dim == xattn_dim
            else nn.Linear(self.skel_branch.skel_token_dim, xattn_dim)
        )

        self.bi_xattn = BidirectionalCrossAttention(
            dim=xattn_dim, num_heads=xattn_heads, dropout=xattn_dropout
        )
        self.fusion_transformer_head = FusionTransformerHead(
            d_model=xattn_dim,
            nhead=xattn_heads,
            num_layers=fusion_num_layers,
            dropout=xattn_dropout,
            max_len=fusion_max_len,
            cls_hidden=fusion_hidden,
            cls_dropout=fusion_dropout,
        )

    def forward(self, parse_x, skel_x, return_aux=False):
        z_parse, parse_logit, parse_tokens = self.parse_branch(parse_x, return_tokens=True)
        z_skel, skel_logit, skel_tokens = self.skel_branch(skel_x, return_tokens=True)

        # project to a common token dimension
        parse_tokens = self.parse_token_proj(parse_tokens)   # [B, T_parse, D]
        skel_tokens = self.skel_token_proj(skel_tokens)      # [B, T_skel, D]

        # bidirectional cross-attention
        parse_tokens_fused, skel_tokens_fused = self.bi_xattn(parse_tokens, skel_tokens)

        # fusion Transformer + CLS + MLP
        fused_tokens = torch.cat([parse_tokens_fused, skel_tokens_fused], dim=1)
        z_fuse, logit = self.fusion_transformer_head(fused_tokens)

        if not return_aux:
            return logit

        aux = {
            "parse_logit": parse_logit,
            "skel_logit": skel_logit,
            "z_fuse": z_fuse,
        }
        return logit, aux


# Alias for configs
Model = DeCAPSFusion
