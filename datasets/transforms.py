"""Input transforms shared by the dataset feeders.

Both functions operate on skeleton arrays of shape (C, T, V, M):
C = channels (e.g. x, y, z), T = frames, V = joints, M = bodies.

- valid_crop_resize: deterministic temporal center-crop to the valid
  frames followed by bilinear resize to a fixed window (GFBMD).
- random_rot: random 3D rotation augmentation, applied at training
  time only (GFBMD).
"""

import numpy as np
import torch
import torch.nn.functional as F


def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    """Center-crop the valid frames, then resize the temporal axis to `window`.

    Args:
        data_numpy: (C, T, V, M) input sequence.
        valid_frame_num: number of valid (non-padded) frames, starting at 0.
        p_interval: crop ratio interval. With a single value p, a centered
            crop of ratio p is used (p=1.0 keeps all valid frames).
        window: output temporal length.

    Returns:
        (C, window, V, M) numpy array (bilinear interpolation over time).
    """
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # Temporal crop (with p_interval=[1.0] this is the full valid range).
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias:end - bias, :, :]
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(
            np.maximum(int(np.floor(valid_size * p)), 64), valid_size
        )
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]

    # Bilinear resize of the temporal axis to `window`.
    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(
        data, size=(C * V * M, window), mode="bilinear", align_corners=False
    ).squeeze()
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data


def _rotation_matrices(rot):
    """Build per-frame 3D rotation matrices from (T, 3) axis angles."""
    cos_r, sin_r = rot.cos(), rot.sin()  # (T, 3)
    zeros = torch.zeros(rot.shape[0], 1)
    ones = torch.ones(rot.shape[0], 1)

    rx = torch.cat(
        (
            torch.stack((ones, zeros, zeros), dim=-1),
            torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), dim=-1),
            torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1),
        ),
        dim=1,
    )
    ry = torch.cat(
        (
            torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), dim=-1),
            torch.stack((zeros, ones, zeros), dim=-1),
            torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1),
        ),
        dim=1,
    )
    rz = torch.cat(
        (
            torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), dim=-1),
            torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), dim=-1),
            torch.stack((zeros, zeros, ones), dim=-1),
        ),
        dim=1,
    )
    return rz.matmul(ry).matmul(rx)  # (T, 3, 3)


def random_rot(data_numpy, theta=0.3):
    """Apply one random 3D rotation (angles ~ U(-theta, theta) per axis).

    The same rotation is applied to every frame of the sequence.
    Training-time augmentation only.

    Args:
        data_numpy: (C, T, V, M) input sequence, C must be 3 (x, y, z).

    Returns:
        Rotated sequence as a torch tensor of shape (C, T, V, M).
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V * M)
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot] * T, dim=0)
    rot = _rotation_matrices(rot)  # (T, 3, 3)
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    return data_torch
