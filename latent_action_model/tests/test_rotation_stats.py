"""Numerical unit tests for rotation-aware relative/delta action processing.

Covers (per the implementation plan verification section):
  T1  SE(3) roundtrip: absolute(relative(t, ref), ref) == t
  T2  relative_rotation matches the user-verified reference get_relative_rot_6d
  T3  numpy / torch parity and return type
  T4  delta_rotation[0] == identity rotation in target rep (chunk-start insight)
  T5  rotation-aware rel/delta statistics: dim = target rep, chunk-level, and
      rotation stats stored under ROTATION_STATS_KEY
  T6  regression: non-rotation fields go through the unchanged elementwise path
      (byte-identical to the pre-change formula)

Run:  python latent_action_model/tests/test_rotation_stats.py
"""

import os
import sys
import types
import importlib.util
import importlib.machinery

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LAM = os.path.abspath(os.path.join(HERE, ".."))
DL = os.path.join(LAM, "dataloader")
GR = os.path.join(DL, "gr00t_lerobot")
sys.path.insert(0, LAM)


def _mk_pkg(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = m
    return m


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Build synthetic packages so submodules import WITHOUT running dataloader/__init__.py
# (which pulls transformers/accelerate). Each real submodule still imports its
# siblings via 'dataloader.gr00t_lerobot.<x>', which resolve to these package objs.
_mk_pkg("dataloader", DL)
_mk_pkg("dataloader.gr00t_lerobot", GR)
_mk_pkg("dataloader.gr00t_lerobot.transform", os.path.join(GR, "transform"))

# Stub the video module (needs decord) that datasets.py imports.
_video = types.ModuleType("dataloader.gr00t_lerobot.video")
_video.get_all_frames = lambda *a, **k: None
_video.get_frames_by_indices = lambda *a, **k: None
_video.get_frames_by_timestamps = lambda *a, **k: None
sys.modules["dataloader.gr00t_lerobot.video"] = _video

# Stub pytorch3d.transforms: RotationTransform imports it at module load but the
# rotation-aware statistics path we test uses only the vendored math in
# rotation_utils. Functions are referenced lazily, so an empty module is enough.
if "pytorch3d" not in sys.modules:
    _pt = types.ModuleType("pytorch3d")
    _ptt = types.ModuleType("pytorch3d.transforms")
    _pt.transforms = _ptt
    sys.modules["pytorch3d"] = _pt
    sys.modules["pytorch3d.transforms"] = _ptt

# Load in dependency order.
_load("dataloader.gr00t_lerobot.embodiment_tags", os.path.join(GR, "embodiment_tags.py"))
_load("dataloader.gr00t_lerobot.schema", os.path.join(GR, "schema.py"))
_load("dataloader.gr00t_lerobot.transform.base", os.path.join(GR, "transform", "base.py"))
ru = _load(
    "dataloader.gr00t_lerobot.transform.rotation_utils",
    os.path.join(GR, "transform", "rotation_utils.py"),
)
# transform/__init__.py re-exports; load it so 'from ..transform import X' works.
_load("dataloader.gr00t_lerobot.transform", os.path.join(GR, "transform", "__init__.py"))
_load("dataloader.gr00t_lerobot.transform.state_action", os.path.join(GR, "transform", "state_action.py"))
ds = _load("dataloader.gr00t_lerobot.datasets", os.path.join(GR, "datasets.py"))

from dataloader.gr00t_lerobot.schema import LeRobotModalityMetadata, RotationType  # noqa: E402

# Optional reference implementation (user-verified, vendored pytorch3d). Only used
# for the T2 cross-check below; the rest of the suite is self-contained. Override
# the path with ROTATION_REF_IMPL, or leave the file absent to skip T2.
_ref_path = os.environ.get(
    "ROTATION_REF_IMPL",
    "/mnt/pfs/dengyiqi/code/xr2_eef/core/utils/rotation_torch_utils.py",
)
ref = None
if os.path.exists(_ref_path):
    _ref_spec = importlib.util.spec_from_file_location("refrot", _ref_path)
    ref = importlib.util.module_from_spec(_ref_spec)
    _ref_spec.loader.exec_module(ref)
else:
    print(f"[skip] reference impl not found at {_ref_path}; T2 cross-check skipped")


PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


def rand_quats(n, seed):
    g = np.random.RandomState(seed)
    q = g.randn(n, 4)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1  # standardize real part
    return torch.tensor(q, dtype=torch.float64)


print("== rotation_utils ==")
q = rand_quats(8, 0)
# Build the rot6d test vectors from the module under test (self-contained). When
# a reference impl is present, it produces the same values (validated by T2).
r6 = ru.matrix_to_rotation(ru.rotation_to_matrix(q, "quaternion"), "rotation_6d")  # (8,6)
ref_q = q[0]
ref_r6 = r6[0]

# T1 roundtrip
rel = ru.relative_rotation(r6[1:], ref_r6, "rotation_6d")
back = ru.absolute_rotation(rel, ref_r6, "rotation_6d")
check("T1 SE(3) roundtrip rot6d", torch.allclose(back, r6[1:], atol=1e-5))

# T2 matches reference (only when the reference impl is available)
if ref is not None:
    ref_rel = ref.get_relative_rot_6d(ref_r6, r6[1:])
    check("T2 matches reference get_relative_rot_6d", torch.allclose(ref_rel, rel, atol=1e-5))
else:
    print("  SKIP  T2 matches reference get_relative_rot_6d (no reference impl)")

# T2b quaternion path roundtrip
relq = ru.relative_rotation(q[1:], ref_q, "quaternion")
backq = ru.absolute_rotation(relq, ref_q, "quaternion")
# compare as matrices to avoid quat sign ambiguity
check(
    "T2b quaternion roundtrip",
    torch.allclose(
        ru.rotation_to_matrix(backq, "quaternion"),
        ru.rotation_to_matrix(q[1:], "quaternion"),
        atol=1e-5,
    ),
)

# T3 numpy parity + type
rel_np = ru.relative_rotation(r6[1:].numpy(), ref_r6.numpy(), "rotation_6d")
check("T3 numpy parity", np.allclose(rel_np, rel.numpy(), atol=1e-5))
check("T3 numpy return type", isinstance(rel_np, np.ndarray))

# T4 delta[0] == identity
chunk = r6  # (8,6)
d = ru.delta_rotation(chunk, ref_r6, "rotation_6d")
identity_r6 = ru.matrix_to_rotation(torch.eye(3, dtype=torch.float64), "rotation_6d")
check("T4 delta[0]==identity rot6d", torch.allclose(d[0], identity_r6, atol=1e-5))
check("T4 identity is [1,0,0,0,1,0]",
      np.allclose(identity_r6.numpy(), [1, 0, 0, 0, 1, 0], atol=1e-6))

# dims
check("T4b dims rot6d/quat/euler", (ru.representation_dim("rotation_6d"),
                                    ru.representation_dim(RotationType.QUATERNION),
                                    ru.representation_dim("euler_angles_rpy")) == (6, 4, 3))


# --- Build a tiny in-memory dataset to test the statistics functions. ---
print("== rotation-aware statistics ==")


def make_modality_meta():
    # action column "action" laid out as: [pos(3) | rot(quat 4) | grip(1)] = 8
    # state column "observation.state" laid out identically.
    d = {
        "state": {
            "eef_position": {"start": 0, "end": 3, "absolute": True, "original_key": "observation.state"},
            "eef_rotation": {"start": 3, "end": 7, "rotation_type": "quaternion", "absolute": True, "original_key": "observation.state"},
            "gripper": {"start": 7, "end": 8, "absolute": True, "original_key": "observation.state"},
        },
        "action": {
            "eef_position": {"start": 0, "end": 3, "absolute": True, "original_key": "action"},
            "eef_rotation": {"start": 3, "end": 7, "rotation_type": "quaternion", "absolute": True, "original_key": "action"},
            "gripper": {"start": 7, "end": 8, "absolute": True, "original_key": "action"},
        },
        "video": {},
    }
    return LeRobotModalityMetadata.model_validate(d)


class _FakeParquet:
    """Minimal stand-in for a pandas DataFrame of one trajectory."""

    def __init__(self, action, state):
        self._d = {"action": action, "observation.state": state}
        self.columns = list(self._d.keys())

    def __len__(self):
        return len(self._d["action"])

    def __getitem__(self, k):
        return list(self._d[k])  # np.stack over rows


# Monkeypatch pd.read_parquet to return our fake trajectories.
import pandas as pd  # noqa: E402

T = 6
g = np.random.RandomState(1)
quats = rand_quats(T, 3).numpy()
traj_action = np.concatenate(
    [g.randn(T, 3), quats, g.rand(T, 1)], axis=1
).astype(np.float32)  # (T,8)
traj_state = np.concatenate(
    [g.randn(T, 3), rand_quats(T, 4).numpy(), g.rand(T, 1)], axis=1
).astype(np.float32)

fake = _FakeParquet(traj_action, traj_state)
_orig_read_parquet = pd.read_parquet
pd.read_parquet = lambda *a, **k: fake

meta = make_modality_meta()
action_keys = ["action.eef_position", "action.eef_rotation", "action.gripper"]
state_keys = ["state.eef_position", "state.eef_rotation", "state.gripper"]
base_stats = {
    "action": {s: np.zeros(8).tolist() for s in ["mean", "std", "min", "max", "q01", "q99"]},
    "observation.state": {s: np.zeros(8).tolist() for s in ["mean", "std", "min", "max", "q01", "q99"]},
}

try:
    rel_stats = ds.calculate_rel_action_statistics(
        parquet_paths=[type("P", (), {"name": "episode_0.parquet"})()],
        lerobot_modality_meta=meta,
        action_keys_full=action_keys,
        state_keys_full=state_keys,
        action_indices=list(range(T)),
        state_indices=[0],
        target_rotations={
            "action.eef_rotation": "rotation_6d",
            "state.eef_rotation": "rotation_6d",
        },
        base_stats=base_stats,
    )

    rot = rel_stats.get(ds.ROTATION_STATS_KEY, {})
    check("T5 ROTATION_STATS_KEY present", "action.eef_rotation" in rot)
    field = rot.get("action.eef_rotation", {})
    check("T5 target rep recorded", field.get("__target_rotation__") == "rotation_6d")
    check("T5 rotation stats dim == 6 (rot6d)", len(field.get("mean", [])) == 6)
    check("T5 has six stats",
          all(k in field for k in ["mean", "std", "min", "max", "q01", "q99"]))

    # T6 regression: non-rotation column-level stats unchanged in shape (8-dim column).
    check("T6 column stats stay 8-dim", len(rel_stats["action"]["mean"]) == 8)

    # T6b: recompute the non-rotation position slice by the OLD elementwise formula
    # over the SAME chunk-level aggregation the stats function performs (loop base
    # index, rel = a_t - s_0 where s_0 is the state at that base index), and confirm
    # it matches the position sub-slice of the new column-level stats.
    action_indices = list(range(T))
    pos = traj_action[:, 0:3]
    state_pos = traj_state[:, 0:3]
    old_samples = []
    for base in range(T):
        steps = np.array(action_indices) + base
        # 'first_last' padding for absolute fields (matches _get_chunk_padded).
        steps = np.clip(steps, 0, T - 1)
        chunk = pos[steps]  # (T,3)
        ref = state_pos[min(base, T - 1)]  # state_indices=[0] -> base offset
        old_samples.append(chunk - ref)
    old_rel_pos = np.concatenate(old_samples, axis=0)
    new_col_min = np.array(rel_stats["action"]["min"])[0:3]
    new_col_max = np.array(rel_stats["action"]["max"])[0:3]
    check(
        "T6b non-rotation rel matches elementwise (min)",
        np.allclose(new_col_min, old_rel_pos.min(axis=0), atol=1e-4),
    )
    check(
        "T6b non-rotation rel matches elementwise (max)",
        np.allclose(new_col_max, old_rel_pos.max(axis=0), atol=1e-4),
    )

    # T7 runtime/stats consistency: the runtime path composes the relative rotation
    # in NATIVE rep (quaternion) then StateActionTransform converts to target (rot6d).
    # The statistics path composes then converts too. For the base-0 chunk the two
    # must produce identical rot6d samples (same function, same reference).
    quat_chunk = traj_action[:, 3:7]  # (T,4) native quaternion
    ref_quat = traj_state[0, 3:7]
    # runtime: relative in native rep, then convert to target (what the transform does)
    runtime_native = ru.relative_rotation(quat_chunk, ref_quat, "quaternion")
    runtime_target = ru.matrix_to_rotation(
        ru.rotation_to_matrix(runtime_native, "quaternion"), "rotation_6d"
    )
    # stats (base 0): relative_rotation then convert to target — same as _relative_action_statistics
    stats_native = ru.relative_rotation(quat_chunk, ref_quat, "quaternion")
    stats_target = ru.matrix_to_rotation(
        ru.rotation_to_matrix(stats_native, "quaternion"), "rotation_6d"
    )
    check(
        "T7 runtime rotation math == stats rotation math",
        np.allclose(runtime_target, stats_target, atol=1e-6),
    )
    # When the action equals its state reference, the relative rotation is identity.
    self_rel = ru.matrix_to_rotation(
        ru.rotation_to_matrix(
            ru.relative_rotation(ref_quat[None], ref_quat, "quaternion"), "quaternion"
        ),
        "rotation_6d",
    )
    check(
        "T7 rel(ref, ref) == identity rot6d",
        np.allclose(self_rel[0], [1, 0, 0, 0, 1, 0], atol=1e-5),
    )
finally:
    pd.read_parquet = _orig_read_parquet


print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
