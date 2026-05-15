"""Step-3 smoke test: instantiate every sensor client in replay mode
against a real episode and assert get_latest() shapes/types.

This does NOT yet check bit-equality with RealImageTactileDataset —
that's step 4. The point here is to lock the replay-client API and
catch missing files / wrong dtypes early.

Run:
    conda activate robo_flow
    cd trainflow
    pytest -q tests/test_replay_clients.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from trainflow.env.ur3_bir.sensor_clients import (
    GelsightClient,
    RealsenseClient,
    UR3Client,
)

REPO = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO / "trainflow" / "dataset" / "peg_hole_tac"
HW_CFG = REPO / "trainflow" / "config" / "hardware" / "ur3_bir_default.yaml"


def _first_episode() -> Path:
    if not DATASET_DIR.exists():
        pytest.skip(f"missing dataset dir: {DATASET_DIR}")
    eps = sorted(p for p in DATASET_DIR.iterdir() if p.is_dir())
    if not eps:
        pytest.skip(f"no episodes under {DATASET_DIR}")
    return eps[0]


@pytest.fixture(scope="module")
def hw_cfg():
    return OmegaConf.load(HW_CFG)


@pytest.fixture(scope="module")
def episode():
    return _first_episode()


@pytest.fixture
def idx_ref():
    return [0]


def test_ur3_client_replay(hw_cfg, episode, idx_ref):
    client = UR3Client.from_npy_replay(hw_cfg.robot.ur3, episode, idx_ref)
    obs = client.get_latest()
    # Client publishes raw fields only — no virtual eef_state7 key.
    # Concat is expressed declaratively in obs_sources via the `concat` op.
    assert obs["tcp_pose6"].shape == (6,)
    assert obs["gripper_width"].shape == (1,)
    assert obs["eef_force"].shape == (6,)
    assert obs["joint_state"].shape == (7,)
    assert isinstance(obs["ts"], float)
    # Advance and verify the read tracks idx_ref
    idx_ref[0] = 5
    obs2 = client.get_latest()
    assert (not np.allclose(obs["tcp_pose6"], obs2["tcp_pose6"])) or obs["ts"] != obs2["ts"]


def test_realsense_client_replay_platform(hw_cfg, episode, idx_ref):
    client = RealsenseClient.from_npy_replay(
        hw_cfg.cameras.platform_realsense, episode, idx_ref, replay_key="rgb"
    )
    obs = client.get_latest()
    rgb = obs["rgb"]
    assert rgb.ndim == 3 and rgb.shape[-1] == 3, rgb.shape
    assert rgb.dtype == np.uint8
    assert isinstance(obs["ts"], float)


def test_realsense_client_replay_hand(hw_cfg, episode, idx_ref):
    client = RealsenseClient.from_npy_replay(
        hw_cfg.cameras.hand_realsense, episode, idx_ref, replay_key="rgb_hand"
    )
    obs = client.get_latest()
    assert obs["rgb"].ndim == 3 and obs["rgb"].shape[-1] == 3


def test_gelsight_client_replay(hw_cfg, episode, idx_ref):
    for slot in (0, 1):
        client = GelsightClient.from_npy_replay(
            hw_cfg.tactile.gelsight, episode, idx_ref, slot=slot
        )
        obs = client.get_latest()
        assert obs["rgb"].ndim == 3 and obs["rgb"].shape[-1] == 3
        assert obs["rgb"].dtype == np.uint8
        assert isinstance(obs["ts"], float)


def test_live_mode_unimplemented(hw_cfg):
    """Live-mode init is deferred to step 5+. Construction via __init__
    succeeds (cfg only), but calling get_latest() before replay setup
    raises NotImplementedError so we don't accidentally ship a live
    runner without finishing the wiring."""
    client = UR3Client(hw_cfg.robot.ur3)
    with pytest.raises(NotImplementedError):
        client.get_latest()


def test_clients_share_idx_ref(hw_cfg, episode):
    """All clients constructed with the same idx_ref read the same frame."""
    idx = [3]
    robot = UR3Client.from_npy_replay(hw_cfg.robot.ur3, episode, idx)
    cam = RealsenseClient.from_npy_replay(
        hw_cfg.cameras.platform_realsense, episode, idx, replay_key="rgb"
    )
    obs_r1 = robot.get_latest()
    obs_c1 = cam.get_latest()
    idx[0] = 7
    obs_r2 = robot.get_latest()
    obs_c2 = cam.get_latest()
    assert obs_r1["ts"] != obs_r2["ts"]
    assert obs_c1["ts"] != obs_c2["ts"]
    # Same source for ts (frame_timestamp.npy), so they must match across clients.
    assert obs_r1["ts"] == obs_c1["ts"]
    assert obs_r2["ts"] == obs_c2["ts"]
