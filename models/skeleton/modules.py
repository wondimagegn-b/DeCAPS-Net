"""Building blocks of the DeCAPS-Net skeleton branch.

Contents (execution order in a Basic_Block):
  Spatial:
    - ST_GC                          standard spatial graph convolution
                                     (used in the first block and as the
                                     non-CAD-SGC ablation)
    - JointContextTransformer        joint-context encoder inside CAD-SGC
    - ContextAwareDeformableSpatialGC  CAD-SGC: context-aware top-k joint
                                     selection + decoupled aggregation
  Temporal:
    - TemporalConv                   plain temporal convolution (ablation)
    - DeTGC                          deformable temporal convolution
    - SingleScale_DeTGC              single DeTGC wrapper (+DeTGC ablation)
    - HTMBranch / MultiScale_TemporalModeling
                                     MTM: 4 parallel DW-TConv(k) + DeTGC
                                     branches, kernels (3, 7, 15, 31)
    - MultiScale_TemporalModeling_Legacy
                                     earlier 2x DeTGC + maxpool + 1x1 variant
                                     (used for the GFBMD experiments)
  Block:
    - Basic_Block                    spatial module + temporal module + residuals

All tensors are (N, C, T, V): batch, channels, frames, joints.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LEAKY_ALPHA = 0.1


def init_param(modules):
    """Kaiming init for conv/linear layers; unit weight / zero bias for norms."""
    for m in modules:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, a=LEAKY_ALPHA, mode="fan_out", nonlinearity="leaky_relu"
            )
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.constant_(m.weight, 1)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Basic temporal convolutions
# ---------------------------------------------------------------------------


class TemporalConv(nn.Module):
    """Plain temporal convolution + BN (ablation baseline)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        super().__init__()

        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class PointWiseTCN(nn.Module):
    """1x1 channel projection + BN (optionally strided in time)."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class DWTemporalConv(nn.Module):
    """Depthwise temporal convolution + BN (no bias)."""

    def __init__(self, channels, kernel_size, stride=1, dilation=1):
        super().__init__()

        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.conv(x))


# ---------------------------------------------------------------------------
# Spatial modules
# ---------------------------------------------------------------------------


class ST_GC(nn.Module):
    """Spatial graph convolution with a learnable adjacency (A is (Nh, V, V))."""

    def __init__(self, in_channels, out_channels, A):
        super().__init__()

        A = torch.from_numpy(A.astype(np.float32))
        self.A = nn.Parameter(A)
        self.Nh = A.size(0)

        self.conv = nn.Conv2d(in_channels, out_channels * self.Nh, 1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, C, T, V = x.size()
        v = self.conv(x).view(N, self.Nh, -1, T, V)
        weights = self.A.to(v.dtype)

        x = torch.einsum("hvu,nhctu->nctv", weights, v)
        return self.bn(x)


class JointContextTransformer(nn.Module):
    """Transformer-style joint-context encoder used by CAD-SGC.

    The input features are first averaged over time, then each joint is
    encoded with global joint context via multi-head self-attention.

    Input : x_mean (N, C, V)
    Output: ctx    (N, C, V)
    """

    def __init__(
        self,
        in_channels,
        num_joint=25,
        context_dim=None,
        num_heads=4,
        mlp_ratio=2.0,
        drop=0.1,
    ):
        super().__init__()

        if context_dim is None:
            context_dim = int(math.ceil(in_channels / num_heads) * num_heads)
            context_dim = max(context_dim, num_heads)

        if context_dim % num_heads != 0:
            raise ValueError(f"context_dim ({context_dim}) must be divisible by num_heads ({num_heads})")

        self.in_channels = in_channels
        self.context_dim = context_dim
        self.num_joint = num_joint

        self.in_proj = nn.Conv1d(in_channels, context_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv1d(context_dim, in_channels, kernel_size=1, bias=False)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_joint, context_dim))

        self.norm1 = nn.LayerNorm(context_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=context_dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,
        )

        hidden_dim = int(context_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(context_dim)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, context_dim),
            nn.Dropout(drop),
        )

        nn.init.normal_(self.pos_embed, std=0.02)
        init_param(self.modules())

    def forward(self, x_mean):
        # x_mean: (N, C, V)
        N, C, V = x_mean.shape
        if V > self.num_joint:
            raise ValueError(f"Input V={V} exceeds configured num_joint={self.num_joint}")

        # (N, C, V) -> (N, V, context_dim)
        tok = self.in_proj(x_mean).transpose(1, 2).contiguous()
        tok = tok + self.pos_embed[:, :V, :]

        # Transformer block
        y = self.norm1(tok)
        y, _ = self.attn(y, y, y, need_weights=False)
        tok = tok + y

        y = self.norm2(tok)
        y = self.mlp(y)
        tok = tok + y

        # back to (N, C, V)
        return self.out_proj(tok.transpose(1, 2).contiguous())


class ContextAwareDeformableSpatialGC(nn.Module):
    """CAD-SGC: context-aware deformable spatial graph convolution.

    Pipeline:
      1) time-average x over T to build joint tokens
      2) encode joint context with JointContextTransformer
      3) shared q/k projections produce:
           - the selection score pi(i, j)
           - the aggregation relation w(i, j)
      4) top-k source joints per center joint from pi
         (plus the graph adjacency prior A)
      5) calibrated differentiable distributions over ALL joints
      6) sampled values + sampled dynamic aggregation weights
         (decoupled aggregation with its own prior B)
      7) aggregate over the sampled neighbors

    Input : x (N, Cin, T, V)
    Output: y (N, Cout, T, V)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        k=8,
        num_scale=4,
        num_joint=25,
        selector_heads=4,
        context_dim=None,
        selector_drop=0.1,
        calib_delta=10.0,
        force_self=True,
        self_topk_boost=1e4,
    ):
        super().__init__()

        A = torch.from_numpy(A.astype(np.float32))
        self.Nh = A.size(0)                      # graph heads / partitions
        self.A = nn.Parameter(A.clone())         # selection prior
        self.B = nn.Parameter(A.clone())         # aggregation prior

        self.k = int(k)
        self.num_scale = int(num_scale)
        self.num_joint = int(num_joint)
        self.calib_delta = float(calib_delta)
        self.force_self = bool(force_self)
        self.self_topk_boost = float(self_topk_boost)

        if out_channels % self.num_scale != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by num_scale ({self.num_scale})")
        if in_channels % self.num_scale != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by num_scale ({self.num_scale})")

        self.Cout_scale = out_channels // self.num_scale

        # make rel_total a multiple of num_scale so reshape is safe
        rel_total = 8 if in_channels == 3 else max(8, in_channels // 8)
        rel_total = int(math.ceil(rel_total / self.num_scale) * self.num_scale)
        self.rel_total = rel_total
        self.rel_per_sh = rel_total // self.num_scale

        # 1) Transformer-style selector context encoder
        self.selector_ctx = JointContextTransformer(
            in_channels=in_channels,
            num_joint=num_joint,
            context_dim=context_dim,
            num_heads=selector_heads,
            mlp_ratio=2.0,
            drop=selector_drop,
        )

        # shared q/k for both selection and aggregation
        self.q_proj = nn.Conv1d(
            in_channels, self.rel_total * self.Nh, kernel_size=1, groups=self.num_scale, bias=False
        )
        self.k_proj = nn.Conv1d(
            in_channels, self.rel_total * self.Nh, kernel_size=1, groups=self.num_scale, bias=False
        )

        # value projection
        self.v_proj = nn.Conv2d(
            in_channels, out_channels * self.Nh, kernel_size=1, groups=self.num_scale, bias=False
        )

        # dynamic aggregation projection:
        # diff(q_i, k_j) in R^(rel_per_sh) -> channel weights in R^(Cout_scale)
        self.agg_proj = nn.Parameter(
            torch.empty(self.num_scale, self.Nh, self.Cout_scale, self.rel_per_sh)
        )

        self.alpha = nn.Parameter(torch.zeros(1))   # selection dynamic scale
        self.beta = nn.Parameter(torch.zeros(1))    # aggregation dynamic scale

        self.value_act = nn.LeakyReLU(LEAKY_ALPHA, inplace=True)
        self.tanh = nn.Tanh()
        self.bn = nn.BatchNorm2d(out_channels)

        nn.init.kaiming_normal_(
            self.agg_proj, a=LEAKY_ALPHA, mode="fan_out", nonlinearity="leaky_relu"
        )
        init_param(self.modules())

    def forward(self, x):
        # x: (N, Cin, T, V)
        N, Cin, T, V = x.shape
        dtype = x.dtype
        device = x.device
        k_eff = min(self.k, V)

        # 1) Joint-context encoding
        x_mean = x.mean(dim=2)                              # (N, Cin, V)
        ctx = self.selector_ctx(x_mean)                     # (N, Cin, V)

        q = self.q_proj(ctx).view(N, self.num_scale, self.Nh, self.rel_per_sh, V)
        k = self.k_proj(ctx).view(N, self.num_scale, self.Nh, self.rel_per_sh, V)

        # 2) Selection pathway: cosine similarity + adjacency prior
        q_sel = F.normalize(q, p=2, dim=3)
        k_sel = F.normalize(k, p=2, dim=3)

        # sim: (N, scale, Nh, V_center, V_source)
        sim = torch.einsum("nshrv,nshru->nshvu", q_sel, k_sel)
        A_sel = self.A.to(dtype).view(1, 1, self.Nh, V, V)
        pi = A_sel + self.alpha.to(dtype) * sim

        # force the self-joint to always appear in the top-k
        if self.force_self:
            eye = torch.eye(V, device=device, dtype=dtype).view(1, 1, 1, V, V)
            pi_topk = pi + self.self_topk_boost * eye
        else:
            pi_topk = pi

        _, topk_idx = torch.topk(pi_topk, k=k_eff, dim=-1)     # (N, scale, Nh, V, k)

        # 3) Calibrated differentiable top-k over ALL joints
        #    sel_prob_full: (N, scale, Nh, V_center, k, V_source)
        one_hot = F.one_hot(topk_idx, num_classes=V).to(dtype)
        sel_logits = pi.unsqueeze(-2) + self.calib_delta * one_hot
        sel_prob_full = torch.softmax(sel_logits, dim=-1)

        # 4) Sample values using the calibrated distributions
        v_feat = self.value_act(self.v_proj(x))
        v_feat = v_feat.view(N, self.num_scale, self.Nh, self.Cout_scale, T, V)

        # v_sampled[n,s,h,c,t,v,k] = sum_u v_feat[n,s,h,c,t,u] * sel_prob_full[n,s,h,v,k,u]
        v_sampled = torch.einsum("nshctu,nshvku->nshctvk", v_feat, sel_prob_full)

        # 5) Aggregation pathway: dynamic graph weights
        #    shared q/k context, separate aggregation prior B
        diff = self.tanh(q.unsqueeze(-1) - k.unsqueeze(-2))    # (N, scale, Nh, rel, V, V)

        # w_dyn[n,s,h,c,v,u] = sum_r diff[n,s,h,r,v,u] * agg_proj[s,h,c,r]
        w_dyn = torch.einsum("nshrvu,shcr->nshcvu", diff, self.agg_proj.to(dtype))

        B_agg = self.B.to(dtype).view(1, 1, self.Nh, 1, V, V)
        w_dense = B_agg + self.beta.to(dtype) * w_dyn

        # sampled aggregation weights using the SAME calibrated selector
        w_sampled = torch.einsum("nshcvu,nshvku->nshcvk", w_dense, sel_prob_full)

        # 6) Final sampled aggregation
        y = (v_sampled * w_sampled.unsqueeze(4)).sum(dim=-1)   # (N, scale, Nh, Cout_s, T, V)

        # sum over graph heads / partitions
        y = y.sum(dim=2).contiguous()                          # (N, scale, Cout_s, T, V)
        y = y.view(N, self.num_scale * self.Cout_scale, T, V)  # (N, Cout, T, V)
        return self.bn(y)


# ---------------------------------------------------------------------------
# Temporal modules
# ---------------------------------------------------------------------------


class DeTGC(nn.Module):
    """Deformable temporal graph convolution.

    Learns `eta` temporal sampling offsets (initialized uniformly in
    [-ref, +ref]), samples frames by linear interpolation, and aggregates
    the eta samples with a Conv3d.
    """

    def __init__(self, in_channels, out_channels, eta, kernel_size=1, stride=1, padding=0,
                 dilation=1, num_scale=1, num_frame=64):
        super().__init__()

        self.ks, self.stride, self.dilation = kernel_size, stride, dilation
        self.T = num_frame
        self.num_scale = num_scale

        self.eta = eta
        ref = (self.ks + (self.ks - 1) * (self.dilation - 1) - 1) // 2
        tr = torch.linspace(-ref, ref, self.eta)
        self.tr = nn.Parameter(tr)

        self.conv_out = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(self.eta, 1, 1)),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        N, C, T, V = x.size()
        Tout = T // self.stride
        dtype = x.dtype

        # learnable sampling locations
        t0 = torch.arange(0, T, self.stride, dtype=dtype, device=x.device)
        tr = self.tr.to(dtype)
        t0, tr = t0.view(1, 1, -1).expand(-1, self.eta, -1), tr.view(1, self.eta, 1)
        t = t0 + tr
        t = t.view(1, 1, -1, 1)

        # indexing (clamped to the runtime T, not the configured num_frame)
        tdn = t.detach().floor()
        tup = tdn + 1
        index1 = torch.clamp(tdn, 0, T - 1).long()
        index2 = torch.clamp(tup, 0, T - 1).long()
        index1, index2 = index1.expand(N, C, -1, V), index2.expand(N, C, -1, V)

        # linear-interpolation sampling
        alpha = tup - t
        x1, x2 = x.gather(-2, index=index1), x.gather(-2, index=index2)
        x = x1 * alpha + x2 * (1 - alpha)
        x = x.view(N, C, self.eta, Tout, V)

        # aggregate the eta samples
        x = self.conv_out(x).squeeze(2)
        return x


class SingleScale_DeTGC(nn.Module):
    """Single DeTGC block (the +DeTGC ablation stage).

    Input / output shape: (N, C, T, V)
    """

    def __init__(self, channels, eta, kernel_size=5, stride=1, dilation=1, num_scale=1, num_frame=64):
        super().__init__()
        self.block = DeTGC(
            in_channels=channels,
            out_channels=channels,
            eta=eta,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            num_scale=num_scale,
            num_frame=num_frame,
        )

    def forward(self, x):
        return self.block(x)


class HTMBranch(nn.Module):
    """One MTM branch: channel projection -> DW-TConv(dw_kernel) -> DeTGC."""

    def __init__(
        self,
        in_channels,
        out_channels,
        eta,
        dw_kernel,
        detgc_kernel_size=5,
        stride=1,
        num_scale=1,
        num_frame=64,
    ):
        super().__init__()

        self.branch = nn.Sequential(
            PointWiseTCN(in_channels, out_channels),              # internal projection
            nn.LeakyReLU(LEAKY_ALPHA, inplace=True),
            DWTemporalConv(out_channels, kernel_size=dw_kernel),  # scale-specific local temporal modeling
            nn.LeakyReLU(LEAKY_ALPHA, inplace=True),
            DeTGC(
                in_channels=out_channels,
                out_channels=out_channels,
                eta=eta,
                kernel_size=detgc_kernel_size,
                stride=stride,
                dilation=1,
                num_scale=num_scale,
                num_frame=num_frame,
            ),
        )

        init_param(self.modules())

    def forward(self, x):
        return self.branch(x)


class MultiScale_TemporalModeling(nn.Module):
    """Multi-scale temporal modeling (MTM) described in the paper.

    Four parallel branches, each DW-TConv(k) -> DeTGC with
    k in (3, 7, 15, 31). Each branch first projects channels so the
    concatenated output has exactly out_channels.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        eta,
        kernel_size=5,                 # used inside DeTGC
        stride=1,
        dilations=(3, 7, 15, 31),      # MTM branch kernels
        num_scale=1,
        num_frame=64,
    ):
        super().__init__()

        if not isinstance(dilations, (list, tuple)):
            raise ValueError("dilations must be a tuple/list of branch kernels, e.g. (3, 7, 15, 31).")

        self.branch_kernels = tuple(dilations)
        self.num_branches = len(self.branch_kernels)

        if out_channels % self.num_branches != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by number of branches ({self.num_branches})"
            )

        scale_channels = out_channels // self.num_branches

        self.branches = nn.ModuleList(
            [
                HTMBranch(
                    in_channels=in_channels,
                    out_channels=scale_channels,
                    eta=eta,
                    dw_kernel=k,
                    detgc_kernel_size=kernel_size,
                    stride=stride,
                    num_scale=num_scale,
                    num_frame=num_frame,
                )
                for k in self.branch_kernels
            ]
        )

        init_param(self.modules())

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        return torch.cat(outs, dim=1)


class MultiScale_TemporalModeling_Legacy(nn.Module):
    """Earlier temporal block: 2x DeTGC (dilation 1 and 2) + maxpool + 1x1.

    Used for the GFBMD experiments (selected with tsm_impl="legacy").
    """

    def __init__(self, in_channels, out_channels, eta, kernel_size=5, stride=1, dilations=1,
                 num_scale=1, num_frame=64):
        super().__init__()

        scale_channels = out_channels // num_scale
        self.num_scale = num_scale if in_channels != 3 else 1

        self.tcn1 = nn.Sequential(
            PointWiseTCN(in_channels, scale_channels),
            nn.LeakyReLU(LEAKY_ALPHA),
            DeTGC(scale_channels, scale_channels, eta,
                  kernel_size=5, stride=stride, dilation=1,
                  num_scale=num_scale, num_frame=num_frame),
        )
        self.tcn2 = nn.Sequential(
            PointWiseTCN(in_channels, scale_channels),
            nn.LeakyReLU(LEAKY_ALPHA),
            DeTGC(scale_channels, scale_channels, eta,
                  kernel_size=5, stride=stride, dilation=2,
                  num_scale=num_scale, num_frame=num_frame),
        )
        self.maxpool3x1 = nn.Sequential(
            PointWiseTCN(in_channels, scale_channels),
            nn.LeakyReLU(LEAKY_ALPHA),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(scale_channels),
        )
        self.conv1x1 = PointWiseTCN(in_channels, scale_channels, stride=stride)

    def forward(self, x):
        return torch.cat(
            [self.tcn1(x), self.tcn2(x), self.maxpool3x1(x), self.conv1x1(x)], 1
        )


# ---------------------------------------------------------------------------
# Basic block
# ---------------------------------------------------------------------------


class Basic_Block(nn.Module):
    """One DeCAPS-Net skeleton block: spatial module + temporal module.

    Spatial: ST_GC for the first block (in_channels == 3) or when
    use_cadsgc=False (ablation); CAD-SGC otherwise.
    Temporal (progressive ablation):
      use_tsm=True                       -> MTM (tsm_impl "htm" or "legacy")
      use_tsm=False, use_detgc=True      -> single DeTGC
      both False                         -> plain temporal conv
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        k,
        eta,
        kernel_size=5,
        stride=1,
        dilations=(3, 7, 15, 31),
        num_frame=64,
        num_joint=25,
        residual=True,
        use_cadsgc=True,
        use_detgc=True,
        use_tsm=True,
        tsm_impl="htm",
    ):
        super().__init__()

        base_num_scale = 4
        self.num_scale = base_num_scale if in_channels != 3 else 1

        self.use_cadsgc = bool(use_cadsgc)
        self.use_detgc = bool(use_detgc)
        self.use_tsm = bool(use_tsm)

        # --- Spatial module ---
        # The very first block (in_channels == 3) always uses ST_GC; the
        # deformable grouped spatial operator is not applied to raw input.
        if in_channels == 3:
            self.gcn = ST_GC(in_channels, out_channels, A)
        else:
            if self.use_cadsgc:
                self.gcn = ContextAwareDeformableSpatialGC(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    A=A,
                    k=k,
                    num_scale=self.num_scale,
                    num_joint=num_joint,
                    selector_heads=4,
                    context_dim=None,
                    selector_drop=0.1,
                    calib_delta=10.0,
                    force_self=True,
                )
            else:
                self.gcn = ST_GC(in_channels, out_channels, A)

        # --- Temporal module ---
        if self.use_tsm:
            TSMClass = (
                MultiScale_TemporalModeling
                if tsm_impl == "htm"
                else MultiScale_TemporalModeling_Legacy
            )
            self.tcn = TSMClass(
                out_channels,
                out_channels,
                eta,
                kernel_size=kernel_size,
                stride=stride,
                dilations=dilations,
                num_scale=base_num_scale,
                num_frame=num_frame,
            )
        elif self.use_detgc:
            self.tcn = SingleScale_DeTGC(
                channels=out_channels,
                eta=eta,
                kernel_size=kernel_size,
                stride=stride,
                dilation=1,
                num_scale=base_num_scale,
                num_frame=num_frame,
            )
        else:
            self.tcn = TemporalConv(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=1,
                groups=1,
            )

        # --- Residual paths ---
        if in_channels != out_channels:
            self.residual1 = PointWiseTCN(in_channels, out_channels, groups=self.num_scale)
        else:
            self.residual1 = nn.Identity()

        if not residual:
            self.residual2 = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual2 = nn.Identity()
        else:
            self.residual2 = PointWiseTCN(in_channels, out_channels, stride=stride, groups=self.num_scale)

        self.relu = nn.LeakyReLU(LEAKY_ALPHA, inplace=True)
        init_param(self.modules())

    def forward(self, x):
        res = x

        x = self.gcn(x)
        x = self.relu(x + self.residual1(res))

        x = self.tcn(x)
        x = self.relu(x + self.residual2(res))

        return x
