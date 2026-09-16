"""DeCAPS-Net skeleton model.

Single implementation covering both datasets:
  - GFBMD   (Kinect v2, 25 joints): tsm_impl="legacy", num_frame=64
  - ASDPose (COCO-17 keypoints):    tsm_impl="htm",    num_frame=200

Architecture (multi-stream J/B/JM/BM design):
  Stage 1 : one primitive encoder stream per selected modality
            (joint, bone, joint_motion, bone_motion)
  Fusion  : static side  = joint + bone
            dynamic side = joint_motion + bone_motion
            final fusion = static + dynamic
  Mid     : refinement stacks on the static and dynamic sides
  Stage 2 : `num_stream` deeper stacks, each with its own classifier head
            (the training loss sums the head logits)

Input : (N, C, T, V, M) — or (N, T, V*C), reshaped internally
Output: list of `num_stream` logit tensors, each (N, num_class)

Modality-subset ablations are controlled with `modalities`, e.g.
  modalities="joint"  or  modalities=["joint", "bone"]
"""

import importlib
import math

import numpy as np
import torch
import torch.nn as nn

from .modules import Basic_Block

AVAILABLE_MODALITIES = ("joint", "bone", "joint_motion", "bone_motion")


def import_class(name):
    """Import a class from a dotted path, e.g. "graphs.kinect25.Graph"."""
    mod_str, _, cls_str = name.rpartition(".")
    if mod_str == "":
        raise ImportError(f"Invalid class path: {name}")
    mod = importlib.import_module(mod_str)
    return getattr(mod, cls_str)


def normalize_modalities(modalities):
    """Normalize the `modalities` argument to an ordered list.

    Accepts:
      - None                    -> all 4 modalities
      - "all" / "all4" / "full" -> all 4
      - "joint,bone" or "joint+bone+joint_motion"
      - ["joint", "bone"]
    """
    if modalities is None:
        mods = list(AVAILABLE_MODALITIES)
    elif isinstance(modalities, str):
        text = modalities.strip().lower()
        if text in {"all", "all4", "full", "default"}:
            mods = list(AVAILABLE_MODALITIES)
        else:
            text = text.replace("+", ",")
            mods = [m.strip().lower() for m in text.split(",") if m.strip()]
    else:
        mods = [str(m).strip().lower() for m in modalities if str(m).strip()]

    if len(mods) == 0:
        raise ValueError("modalities cannot be empty")

    seen = set()
    ordered = []
    for m in mods:
        if m not in AVAILABLE_MODALITIES:
            raise ValueError(f"Unknown modality '{m}'. Supported: {AVAILABLE_MODALITIES}")
        if m not in seen:
            ordered.append(m)
            seen.add(m)

    return ordered


def _bone_pairs_ntu25():
    """Bone pairs for the Kinect v2 25-joint layout (1-based)."""
    return [
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
        (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
        (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12),
    ]


def _bone_pairs_coco17():
    """Bone pairs for the COCO-17 keypoint layout (1-based)."""
    return [
        (1, 1),
        (2, 1), (3, 1),
        (4, 2), (5, 3),
        (12, 12),
        (13, 12),
        (6, 12),
        (7, 13),
        (8, 6), (10, 8),
        (9, 7), (11, 9),
        (14, 12), (16, 14),
        (15, 13), (17, 15),
    ]


def _expand_A_to_3(A):
    """Replicate a single-partition adjacency (V,V) / (1,V,V) into 3 partitions."""
    A = np.asarray(A, dtype=np.float32)
    if A.ndim == 2:
        A = A[None, ...]
    if A.shape[0] == 1:
        A = np.repeat(A, 3, axis=0)
    return A


class BlockStack(nn.Sequential):
    """A stack of Basic_Blocks built from a compact argument list.

    block_args entries: [in_channels, out_channels, stride, residual, num_frame, num_joint]
    """

    def __init__(self, block_args, A, k, eta,
                 use_cadsgc=True, use_detgc=True, use_tsm=True, tsm_impl="htm"):
        super().__init__()
        for i, (in_channels, out_channels, stride, residual, num_frame, num_joint) in enumerate(block_args):
            self.add_module(
                f"block-{i}",
                Basic_Block(
                    in_channels,
                    out_channels,
                    A,
                    k,
                    eta,
                    stride=stride,
                    num_frame=num_frame,
                    num_joint=num_joint,
                    residual=residual,
                    use_cadsgc=use_cadsgc,
                    use_detgc=use_detgc,
                    use_tsm=use_tsm,
                    tsm_impl=tsm_impl,
                ),
            )


class FusionBlock(nn.Module):
    """Fuse two feature maps [N*M, C, T, V].

    Modes:
      - add:    x1 + x2 -> BN -> LeakyReLU
      - concat: cat(x1, x2) -> 1x1 conv -> BN -> LeakyReLU
    Output always has shape [N*M, C, T, V].
    """

    def __init__(self, channels, mode="add", act_slope=0.1):
        super().__init__()
        mode = str(mode).lower()
        if mode not in ["add", "concat"]:
            raise ValueError(f"fusion mode must be 'add' or 'concat', got {mode}")
        self.mode = mode

        if self.mode == "concat":
            self.fuse = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(act_slope, inplace=True),
            )
        else:
            self.fuse = nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(act_slope, inplace=True),
            )

    def forward(self, x1, x2):
        if self.mode == "concat":
            return self.fuse(torch.cat([x1, x2], dim=1))
        return self.fuse(x1 + x2)


class DeCAPSSkeleton(nn.Module):
    def __init__(
        self,
        num_class=60,
        num_point=25,
        num_person=1,
        k=8,
        eta=4,
        num_stream=2,
        graph=None,
        graph_args=dict(),
        in_channels=3,
        drop_out=0,
        num_frame=64,
        fusion_mode="add",
        modalities=None,
        use_cadsgc=True,
        use_detgc=True,
        use_tsm=True,
        tsm_impl="htm",
    ):
        super().__init__()

        if graph is None:
            raise ValueError("graph must be provided")

        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        A = _expand_A_to_3(self.graph.A)

        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
        self.num_stream = int(num_stream)
        self.fusion_mode = str(fusion_mode).lower()

        if self.fusion_mode not in ["add", "concat"]:
            raise ValueError(f"fusion_mode must be 'add' or 'concat', got {fusion_mode}")

        self.modalities = normalize_modalities(modalities)
        self.num_active_modal = len(self.modalities)

        self.use_cadsgc = bool(use_cadsgc)
        self.use_detgc = bool(use_detgc)
        self.use_tsm = bool(use_tsm)

        if num_point == 25:
            self.bone_pairs = _bone_pairs_ntu25()
        elif num_point == 17:
            self.bone_pairs = _bone_pairs_coco17()
        else:
            raise ValueError(
                f"Unsupported num_point={num_point}. "
                "Add a bone-pair definition for this skeleton layout."
            )

        # BN over the selected modalities only
        self.data_bn = nn.BatchNorm1d(
            num_person * in_channels * num_point * self.num_active_modal
        )

        base_channel = 64
        base_frame = int(num_frame)
        self.base_channel = base_channel

        block_args = dict(
            use_cadsgc=self.use_cadsgc,
            use_detgc=self.use_detgc,
            use_tsm=self.use_tsm,
            tsm_impl=tsm_impl,
        )

        # Stage 1: primitive encoders
        self.blockargs1 = [
            [in_channels, base_channel, 1, False, base_frame, num_point],
            [base_channel, base_channel, 1, True, base_frame, num_point],
            [base_channel, base_channel, 1, True, base_frame, num_point],
        ]

        # Mid refinement: streams_mid[0] -> static side, streams_mid[1] -> dynamic side
        self.blockargs_mid = [
            [base_channel, base_channel, 1, True, base_frame, num_point],
            [base_channel, base_channel, 1, True, base_frame, num_point],
            [base_channel, base_channel, 1, True, base_frame, num_point],
        ]

        # Stage 2: deeper stack (one per output stream)
        self.blockargs2 = [
            [base_channel, base_channel, 1, True, base_frame, num_point],
            [base_channel, base_channel * 2, 2, True, base_frame, num_point],
            [base_channel * 2, base_channel * 2, 1, True, max(1, base_frame // 2), num_point],
            [base_channel * 2, base_channel * 2, 1, True, max(1, base_frame // 2), num_point],
            [base_channel * 2, base_channel * 4, 2, True, max(1, base_frame // 2), num_point],
            [base_channel * 4, base_channel * 4, 1, True, max(1, base_frame // 4), num_point],
            [base_channel * 4, base_channel * 4, 1, True, max(1, base_frame // 4), num_point],
        ]

        self.streams1 = nn.ModuleDict(
            {name: BlockStack(self.blockargs1, A, k, eta, **block_args) for name in AVAILABLE_MODALITIES}
        )
        self.streams_mid = nn.ModuleList(
            [BlockStack(self.blockargs_mid, A, k, eta, **block_args) for _ in range(2)]
        )
        self.streams2 = nn.ModuleList(
            [BlockStack(self.blockargs2, A, k, eta, **block_args) for _ in range(self.num_stream)]
        )

        # Fusion blocks
        self.fuse_static = FusionBlock(base_channel, mode=self.fusion_mode)
        self.fuse_dynamic = FusionBlock(base_channel, mode=self.fusion_mode)
        self.fuse_final = FusionBlock(base_channel, mode=self.fusion_mode)

        self.bn_mid = nn.BatchNorm2d(base_channel)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.fc = nn.ModuleList(
            [nn.Linear(base_channel * 4, num_class) for _ in range(self.num_stream)]
        )

        for fc in self.fc:
            nn.init.normal_(fc.weight, 0, math.sqrt(2.0 / num_class))
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0)

        nn.init.constant_(self.data_bn.weight, 1)
        nn.init.constant_(self.data_bn.bias, 0)

        self.drop_out = nn.Dropout(drop_out) if drop_out else nn.Identity()

    # ------------------------------------------------------------------
    # Primitive modality construction
    # ------------------------------------------------------------------

    def _build_bone(self, x):
        # x: [N, C, T, V, M]
        x_bone = torch.zeros_like(x)
        for v1, v2 in self.bone_pairs:
            x_bone[:, :, :, v1 - 1, :] = x[:, :, :, v1 - 1, :] - x[:, :, :, v2 - 1, :]
        return x_bone

    def _build_motion(self, x):
        # x: [N, C, T, V, M]
        x_motion = torch.zeros_like(x)
        x_motion[:, :, :-1, :, :] = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        x_motion[:, :, -1, :, :] = 0
        return x_motion

    def _encode_selected_modalities(self, x):
        # Build all primitive modalities once
        x_joint = x
        x_bone = self._build_bone(x_joint)
        x_joint_motion = self._build_motion(x_joint)
        x_bone_motion = self._build_motion(x_bone)

        primitive = {
            "joint": x_joint,
            "bone": x_bone,
            "joint_motion": x_joint_motion,
            "bone_motion": x_bone_motion,
        }

        # Select only the requested modalities
        selected = [primitive[m] for m in self.modalities]

        # Concat selected modalities along channels
        x_all = torch.cat(selected, dim=1)

        # Shared normalization over the selected modalities only
        N, C_all, T, V, M = x_all.size()
        x_all = x_all.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C_all, T)
        x_all = self.data_bn(x_all)
        x_all = x_all.view(N, M, V, C_all, T).permute(0, 1, 3, 4, 2).contiguous()
        x_all = x_all.view(N * M, C_all, T, V)

        # Split back according to the selected modalities
        chunks = x_all.chunk(self.num_active_modal, dim=1)

        feats = {}
        for name, x_mod in zip(self.modalities, chunks):
            feats[name] = self.streams1[name](x_mod)

        return feats, N, M

    def _merge_branch(self, feat_list, fuse_block, refine_block):
        """feat_list has length 0, 1, or 2 (one branch holds at most 2 modalities)."""
        if len(feat_list) == 0:
            return None
        if len(feat_list) == 1:
            x = feat_list[0]
        elif len(feat_list) == 2:
            x = fuse_block(feat_list[0], feat_list[1])
        else:
            raise ValueError("Each branch supports at most 2 modalities.")
        return refine_block(x)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def extract_stream_features(self, x):
        """Run the trunk up to stage 2.

        Returns:
            stream_maps: list of `num_stream` tensors [N*M, C_final, T', V]
            N, M: batch size and number of persons
        (Used by the fusion model to build temporal skeleton tokens.)
        """
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)

        # Stage 1: encode the selected primitive modalities
        feat_dict, N, M = self._encode_selected_modalities(x)

        # Static (joint/bone) and dynamic (joint_motion/bone_motion) sides
        static_feats = [feat_dict[m] for m in ("joint", "bone") if m in feat_dict]
        dynamic_feats = [feat_dict[m] for m in ("joint_motion", "bone_motion") if m in feat_dict]

        f_static = self._merge_branch(static_feats, self.fuse_static, self.streams_mid[0])
        f_dynamic = self._merge_branch(dynamic_feats, self.fuse_dynamic, self.streams_mid[1])

        # Final fusion
        if f_static is not None and f_dynamic is not None:
            x_fused = self.fuse_final(f_static, f_dynamic)
        elif f_static is not None:
            x_fused = f_static
        elif f_dynamic is not None:
            x_fused = f_dynamic
        else:
            raise RuntimeError("No active branch was built. Check modalities.")

        x_fused = self.relu(self.bn_mid(x_fused))

        # Stage 2 stacks
        stream_maps = [stream(x_fused) for stream in self.streams2]
        return stream_maps, N, M

    def forward(self, x):
        stream_maps, N, M = self.extract_stream_features(x)

        out = []
        for y, fc in zip(stream_maps, self.fc):
            c_new = y.size(1)
            y = y.view(N, M, c_new, -1)
            y = y.mean(3).mean(1)               # pool over (T' * V) and persons
            out.append(fc(self.drop_out(y)))

        return out
