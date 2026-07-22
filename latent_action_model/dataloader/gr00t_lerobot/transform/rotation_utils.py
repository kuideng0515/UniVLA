# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rotation math utilities for rotation-aware relative / delta action processing.

This module provides SE(3)-composition primitives keyed by ``RotationType`` so
that relative / delta action modes compose rotations on SO(3) (``R_ref^T @ R_t``)
instead of doing (mathematically wrong) elementwise subtraction on the rotation
sub-vector.

Design notes:
    * The quaternion / rotation_6d / matrix conversions are VENDORED verbatim from
      PyTorch3D (and identical to the user-verified reference implementation
      ``xr2_eef/core/utils/rotation_torch_utils.py``). rot6d follows the pytorch3d
      convention: the first two ROWS of the rotation matrix (``matrix[..., :2, :]``),
      reconstructed via Gram-Schmidt. This is the SAME convention as
      ``RotationTransform`` in ``state_action.py`` (which imports pytorch3d), so
      the two are interchangeable.
    * euler_angles / axis_angle conversions defer to ``pytorch3d.transforms`` (only
      imported lazily, when such a representation is actually used), since the
      verified reference did not need them.
    * Functions accept numpy arrays (used by the offline statistics functions) or
      torch tensors (used by the runtime transform) and return the same type they
      were given, so both call sites share one source of truth.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..schema import RotationType

_QUATERNION = "quaternion"
_AXIS_ANGLE = "axis_angle"
_ROTATION_6D = "rotation_6d"
_MATRIX = "matrix"

# Dimensionality of each representation's flat vector.
REPRESENTATION_DIMS: dict[str, int] = {
    _QUATERNION: 4,
    _AXIS_ANGLE: 3,
    _ROTATION_6D: 6,
    _MATRIX: 9,
    "euler_angles": 3,
}


# ---------------------------------------------------------------------------
# Vendored PyTorch3D conversions (verbatim from xr2_eef rotation_torch_utils.py,
# itself copied from facebookresearch/pytorch3d). Kept local so the verified
# rot6d/quaternion path has no hard pytorch3d import dependency.
# ---------------------------------------------------------------------------
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Quaternions (real part first, ``(..., 4)``) -> rotation matrices ``(..., 3, 3)``."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D rotation (Zhou et al.) ``(..., 6)`` -> rotation matrices ``(..., 3, 3)``."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(..., 3, 3)`` -> 6D rotation ``(..., 6)`` (first two rows)."""
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(..., 3, 3)`` -> quaternions (real part first, ``(..., 4)``)."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )
    return standardize_quaternion(out)


# ---------------------------------------------------------------------------
# RotationType-keyed dispatch
# ---------------------------------------------------------------------------
def _parse_rotation_type(rotation_type: "RotationType | str") -> tuple[str, str | None]:
    """Split a RotationType into (base rep, euler convention or None).

    Convention parsing mirrors ``RotationTransform.__init__`` (state_action.py):
    r->X, p->Y, y->Z. One shared source of truth for the euler letter mapping.
    """
    rep = rotation_type.value if isinstance(rotation_type, RotationType) else str(rotation_type)
    if rep.startswith("euler_angles"):
        convention = rep.split("_")[-1]
        convention = convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        return "euler_angles", convention
    return rep, None


def representation_dim(rotation_type: "RotationType | str") -> int:
    """Flat vector length for a rotation representation."""
    base, _ = _parse_rotation_type(rotation_type)
    return REPRESENTATION_DIMS[base]


def _to_matrix_torch(t: torch.Tensor, base: str, convention: str | None) -> torch.Tensor:
    if base == _QUATERNION:
        return quaternion_to_matrix(t)
    if base == _ROTATION_6D:
        return rotation_6d_to_matrix(t)
    if base == _MATRIX:
        return t.reshape(t.shape[:-1] + (3, 3))
    # euler_angles / axis_angle -> defer to pytorch3d
    import pytorch3d.transforms as pt

    if base == "euler_angles":
        return pt.euler_angles_to_matrix(t, convention)
    return pt.axis_angle_to_matrix(t)


def _from_matrix_torch(m: torch.Tensor, base: str, convention: str | None) -> torch.Tensor:
    if base == _QUATERNION:
        return matrix_to_quaternion(m)
    if base == _ROTATION_6D:
        return matrix_to_rotation_6d(m)
    if base == _MATRIX:
        return m.reshape(m.shape[:-2] + (9,))
    import pytorch3d.transforms as pt

    if base == "euler_angles":
        return pt.matrix_to_euler_angles(m, convention)
    return pt.matrix_to_axis_angle(m)


def _as_tensor(arr) -> tuple[torch.Tensor, bool]:
    """Return (float64 tensor, was_numpy)."""
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float64), True
    if isinstance(arr, torch.Tensor):
        return arr, False
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(arr)}")


def _restore(t: torch.Tensor, was_numpy: bool, ref):
    if was_numpy:
        return t.cpu().numpy().astype(np.asarray(ref).dtype)
    return t.to(ref.dtype)


def rotation_to_matrix(arr, rotation_type: "RotationType | str"):
    """Flat rotation representation ``(..., D)`` -> matrices ``(..., 3, 3)``."""
    base, convention = _parse_rotation_type(rotation_type)
    t, was_numpy = _as_tensor(arr)
    mat = _to_matrix_torch(t, base, convention)
    return _restore(mat, was_numpy, arr) if was_numpy else mat


def matrix_to_rotation(mat, rotation_type: "RotationType | str"):
    """Matrices ``(..., 3, 3)`` -> flat rotation representation ``(..., D)``."""
    base, convention = _parse_rotation_type(rotation_type)
    t, was_numpy = _as_tensor(mat)
    out = _from_matrix_torch(t, base, convention)
    return _restore(out, was_numpy, mat) if was_numpy else out


def relative_rotation(arr_t, arr_ref, rotation_type: "RotationType | str"):
    """Relative rotation of ``arr_t`` w.r.t. ``arr_ref``: ``R_ref^T @ R_t``.

    Args:
        arr_t: rotations to transform, shape ``(..., D)`` in ``rotation_type``.
        arr_ref: reference rotation(s), broadcastable to ``arr_t``, same rep.
        rotation_type: representation of both inputs (and the output).

    Returns:
        Relative rotations in the SAME representation as the inputs, same type
        (numpy / torch) as ``arr_t``.
    """
    t, was_numpy = _as_tensor(arr_t)
    ref, _ = _as_tensor(arr_ref)
    base, convention = _parse_rotation_type(rotation_type)
    R_t = _to_matrix_torch(t, base, convention)
    R_ref = _to_matrix_torch(ref, base, convention)
    R_rel = torch.matmul(R_ref.transpose(-1, -2), R_t)  # R_ref^T @ R_t
    out = _from_matrix_torch(R_rel, base, convention)
    return _restore(out, was_numpy, arr_t)


def absolute_rotation(arr_rel, arr_ref, rotation_type: "RotationType | str"):
    """Inverse of :func:`relative_rotation`: ``R_ref @ R_rel``."""
    rel, was_numpy = _as_tensor(arr_rel)
    ref, _ = _as_tensor(arr_ref)
    base, convention = _parse_rotation_type(rotation_type)
    R_rel = _to_matrix_torch(rel, base, convention)
    R_ref = _to_matrix_torch(ref, base, convention)
    R_abs = torch.matmul(R_ref, R_rel)
    out = _from_matrix_torch(R_abs, base, convention)
    return _restore(out, was_numpy, arr_rel)


def delta_rotation(arr, ref, rotation_type: "RotationType | str"):
    """Per-step delta rotation along a chunk.

    Mirrors the elementwise-delta convention used for translations:
      * step 0: relative to ``ref`` (the state reference), ``R_ref^T @ R_0``
      * step t>0: relative to previous step, ``R_{t-1}^T @ R_t``

    Args:
        arr: chunk of rotations, shape ``(T, D)`` in ``rotation_type``.
        ref: reference rotation, shape ``(D,)`` (state at t=0), same rep.
        rotation_type: representation of inputs and output.

    Returns:
        Delta rotations, shape ``(T, D)``, same type as ``arr``.
    """
    t, was_numpy = _as_tensor(arr)
    ref_t, _ = _as_tensor(ref)
    base, convention = _parse_rotation_type(rotation_type)
    R = _to_matrix_torch(t, base, convention)  # (T, 3, 3)
    R_ref = _to_matrix_torch(ref_t, base, convention)  # (3, 3)
    prev = torch.cat([R_ref.unsqueeze(0), R[:-1]], dim=0)  # [ref, R_0, ..., R_{T-2}]
    R_delta = torch.matmul(prev.transpose(-1, -2), R)
    out = _from_matrix_torch(R_delta, base, convention)
    return _restore(out, was_numpy, arr)
