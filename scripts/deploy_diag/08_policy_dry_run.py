"""Phase H — trained policy, inference only (no execution).

Loads a trained checkpoint, runs it against LIVE obs from `Ur3BirEnv`
at `--inference-fps` (default = task control_fps), decodes each
predicted action via the task's `shape_meta.action.out` chain, and
logs both the raw policy output and the decoded per-sink values.

The robot is never commanded. RTDEReceiveInterface is read-only;
RTDEControlInterface is never instantiated by this script. Run with
the robot powered on + e-stop pressed if your task's shape_meta.obs
includes any `robot.ur3.*` source.

The script does NOT use `Ur3BirRunner` — we run inference inline so
Phase H can land without the action_executor module. When Phase I
needs real dispatch, the runner takes over with a real executor.

Usage:
    python -m scripts.deploy_diag.08_policy_dry_run \\
        --task real_ee2_dice \\
        --run-dir /path/to/data/outputs/<date>/<time>_<workspace>_<task>/ \\
        --duration 60 \\
        --out /tmp/policy_dry_run

If your checkpoint was trained against an older task cfg (different
shape_meta), the script will refuse to start; the obs surface the
model expects must match what the live env produces.

See scripts/deploy_diag/README.md (Phase H) for go/no-go criteria.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

from trainflow.env.ur3_bir.action_decoder import decode_action
from trainflow.env.ur3_bir.ur3_bir_env import Ur3BirEnv


REPO = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / "trainflow" / "config"


def _load_task(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        cfg = compose(config_name=f"task/{name}")
    return cfg.task


def _find_best_ckpt(run_dir: Path) -> Path:
    """Lowest-loss top-k ckpt by filename, else latest.ckpt. Mirrors
    eval_offline.find_best_ckpt."""
    topk = list((run_dir / "checkpoints").glob("epoch=*.ckpt"))
    def loss_of(p: Path) -> float:
        try:
            return float(p.stem.split("=")[-1])
        except ValueError:
            return float("inf")
    if topk:
        return min(topk, key=loss_of)
    return run_dir / "checkpoints" / "latest.ckpt"


def _build_policy(run_dir: Path, ckpt_path: Path, device: str):
    """Load the workspace + checkpoint payload + pick EMA weights.
    Pattern copied from eval_offline.py."""
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    WorkspaceCls = hydra.utils.get_class(cfg._target_)
    workspace = WorkspaceCls(cfg)

    payload = torch.load(
        open(ckpt_path, "rb"), pickle_module=dill,
        map_location="cpu", weights_only=False,
    )
    workspace.load_payload(payload, exclude_keys=["optimizer"])

    policy = workspace.model
    if cfg.training.use_ema and getattr(workspace, "ema_model", None) is not None:
        policy = workspace.ema_model
        print("using EMA weights")
    policy = policy.to(device).eval()
    return policy, cfg


def _check_shape_meta_compat(task_cfg, ckpt_cfg) -> list[str]:
    """Return a list of WARNINGS about shape_meta drift between the
    checkpoint and the current task cfg. Empty list = clean match."""
    warnings = []
    ckpt_obs = ckpt_cfg.task.shape_meta.get("obs", {}) or {}
    env_obs = task_cfg.shape_meta.get("obs", {}) or {}
    if set(ckpt_obs.keys()) != set(env_obs.keys()):
        warnings.append(
            f"obs keys differ — ckpt={sorted(ckpt_obs)}  env={sorted(env_obs)}")
    for k in ckpt_obs.keys() & env_obs.keys():
        c_shape = list(ckpt_obs[k].get("shape", []))
        e_shape = list(env_obs[k].get("shape", []))
        if c_shape != e_shape:
            warnings.append(
                f"obs.{k} shape differs — ckpt={c_shape} env={e_shape}")
    c_act = list((ckpt_cfg.task.shape_meta.get("action", {}) or {}).get("shape", []))
    e_act = list((task_cfg.shape_meta.get("action", {}) or {}).get("shape", []))
    if c_act != e_act:
        warnings.append(f"action shape differs — ckpt={c_act} env={e_act}")
    return warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True,
                    help="Task yaml name under config/task/. Drives the LIVE env.")
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="data/outputs/<date>/<time>_<workspace>_<task>/")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Specific checkpoint (default: best top-k or latest.ckpt).")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--n-obs-steps", type=int, default=2)
    ap.add_argument("--inference-fps", type=float, default=None,
                    help="Default: task.hardware.control_fps.")
    ap.add_argument("--out", type=Path, default=Path("/tmp/policy_dry_run"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-inference-steps", type=int, default=None,
                    help="Override DDIM denoise steps.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    run_dir = args.run_dir.resolve()
    ckpt_path = (args.ckpt or _find_best_ckpt(run_dir)).resolve()
    if not ckpt_path.exists():
        print(f"[fatal] ckpt not found: {ckpt_path}", file=sys.stderr)
        return 2
    print(f"[info] run_dir : {run_dir}")
    print(f"[info] ckpt    : {ckpt_path}")
    print(f"[info] out_dir : {args.out}")

    # Build policy.
    policy, ckpt_cfg = _build_policy(run_dir, ckpt_path, args.device)
    if args.num_inference_steps is not None:
        policy.num_inference_steps = args.num_inference_steps
        print(f"[info] num_inference_steps -> {args.num_inference_steps}")

    # Build env from the CURRENT task cfg (live hardware path).
    task_cfg = _load_task(args.task)
    warnings = _check_shape_meta_compat(task_cfg, ckpt_cfg)
    if warnings:
        print("[fatal] shape_meta mismatch between checkpoint and live env:")
        for w in warnings:
            print(f"  {w}")
        return 2

    inference_fps = float(args.inference_fps
                          if args.inference_fps is not None
                          else task_cfg.hardware.control_fps)
    interval = 1.0 / inference_fps
    print(f"[info] inference_fps = {inference_fps}  interval = {interval*1000:.1f}ms")

    env = Ur3BirEnv(
        task_cfg=task_cfg,
        n_obs_steps=args.n_obs_steps,
        obs_temporal_downsample_ratio=1,
    )
    env.start()
    try:
        # Wait for buffer warmup.
        warm_t = time.monotonic()
        while len(env._buffer) < args.n_obs_steps:
            if time.monotonic() - warm_t > 5.0:
                print("[fatal] buffer warmup timed out; producer errors:")
                for e in env.producer_errors[-5:]:
                    print(f"  {e}")
                return 2
            time.sleep(0.02)
        print(f"[info] buffer warm; starting inference loop")

        log_fp = open(args.out / "ticks.jsonl", "w", buffering=1)
        latencies = []
        action_cfg = task_cfg.shape_meta.action

        t_start = time.monotonic()
        next_t = t_start
        n_ticks = 0
        while time.monotonic() - t_start < args.duration:
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(next_t - now, 0.005))
                continue
            next_t += interval

            t0 = time.monotonic()
            try:
                obs_np = env.get_obs()
            except Exception as e:
                print(f"[get_obs] {e}")
                continue
            obs_t = {
                k: torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(0).to(args.device)
                for k, v in obs_np.items()
            }
            try:
                with torch.no_grad():
                    result = policy.predict_action(obs_t)
            except Exception as e:
                print(f"[predict_action] {e}")
                continue
            action_horizon = result["action"][0].detach().to("cpu").numpy()
            lat_ms = (time.monotonic() - t0) * 1000.0
            latencies.append(lat_ms)
            n_ticks += 1

            # Decode the first action of the predicted horizon (the one
            # the runner's control loop would pop next).
            try:
                decoded = decode_action(action_horizon[0], action_cfg)
            except Exception as e:
                print(f"[decode_action] {e}")
                decoded = {}

            row: dict = {
                "t": time.time(),
                "latency_ms": lat_ms,
                "action_pred_first": action_horizon[0].tolist(),
                "decoded": {k: v.tolist() for k, v in decoded.items()},
            }
            if "eef_state" in obs_np:
                row["eef_state_last"] = obs_np["eef_state"][-1].tolist()
            log_fp.write(json.dumps(row) + "\n")

        log_fp.close()
    finally:
        env.stop()

    print()
    print("=== summary ===")
    if not latencies:
        print("  no ticks recorded")
        return 2
    arr = np.array(latencies)
    print(f"  ticks={len(arr)}  mean={arr.mean():.1f}ms  "
          f"p50={np.percentile(arr,50):.1f}ms  p99={np.percentile(arr,99):.1f}ms  "
          f"max={arr.max():.1f}ms")
    budget_ms = interval * 1000
    on_budget = float((arr < budget_ms).sum()) / len(arr)
    print(f"  within {budget_ms:.0f}ms inference period: {on_budget*100:.2f}%")
    if env.producer_errors:
        print(f"  producer errors (last 5): {env.producer_errors[-5:]}")
    print(f"  log: {args.out/'ticks.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
