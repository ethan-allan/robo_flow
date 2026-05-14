"""Quick sanity check for the early-stopping block in
trainflow/workspace/train_diffusion_unet_image_workspace.py (mirrored in the
transformer workspace). Runs the same init + per-checkpoint logic against
synthetic step_log streams and asserts when the break fires.

Run: python trainflow/scripts/test_early_stopping.py
"""
from types import SimpleNamespace


def run_es(es_cfg_dict, monitor_values):
    """Replicates the workspace's inline ES logic verbatim.

    Returns (stop_epoch_or_None, es_best, es_stale, per_epoch_log).
    `monitor_values[i]` is the value the monitor_key takes at checkpoint event i
    (i.e. epoch i, assuming checkpoint_every=1). A value of None simulates the
    key being missing from step_log.
    """
    es_cfg = SimpleNamespace(**es_cfg_dict) if es_cfg_dict is not None else None
    es_enabled = bool(es_cfg.enabled) if es_cfg is not None else False
    es_mode = es_cfg.mode if es_enabled else 'min'
    es_best = float('inf') if es_mode == 'min' else float('-inf')
    es_stale = 0
    per_epoch = []

    for epoch, v in enumerate(monitor_values):
        if not es_enabled:
            per_epoch.append({'epoch': epoch, 'v': v})
            continue

        if v is None:
            raise KeyError(
                f"early_stopping: monitor_key {es_cfg.monitor_key!r} missing from "
                f"step_log at epoch {epoch}."
            )

        improved = (
            (es_best - v) > es_cfg.min_delta if es_mode == 'min'
            else (v - es_best) > es_cfg.min_delta
        )
        if improved:
            es_best = v
            es_stale = 0
        else:
            es_stale += 1

        per_epoch.append({
            'epoch': epoch, 'v': v, 'es_best': es_best, 'es_stale': es_stale,
            'improved': improved,
        })

        if es_stale >= es_cfg.patience:
            return epoch, es_best, es_stale, per_epoch

    return None, es_best, es_stale, per_epoch


def case(name, cfg, values, expect_stop_at, expect_best=None):
    try:
        stop, best, stale, log = run_es(cfg, values)
    except KeyError as e:
        if expect_stop_at == 'KeyError':
            print(f"  PASS  {name}: raised KeyError as expected ({e})")
            return True
        print(f"  FAIL  {name}: unexpected KeyError ({e})")
        return False

    if expect_stop_at == 'KeyError':
        print(f"  FAIL  {name}: expected KeyError but ran to end")
        return False

    ok = stop == expect_stop_at
    if expect_best is not None:
        ok = ok and abs(best - expect_best) < 1e-12
    tag = "PASS" if ok else "FAIL"
    print(f"  {tag}  {name}: stop_epoch={stop} (expected {expect_stop_at}), "
          f"best={best}, stale={stale}")
    if not ok:
        for row in log:
            print(f"        {row}")
    return ok


def main():
    print("Scenario A: mode=min, monotonic decrease > min_delta -> never stops")
    a = case(
        "monotonic_decrease",
        {'enabled': True, 'monitor_key': 'val_action_mse_error',
         'mode': 'min', 'patience': 3, 'min_delta': 1e-4},
        [0.9, 0.8, 0.7, 0.6, 0.5],
        expect_stop_at=None,
        expect_best=0.5,
    )

    print("Scenario B: mode=min, plateau after epoch 0 -> stops at patience")
    b = case(
        "plateau_min_patience3",
        {'enabled': True, 'monitor_key': 'val_action_mse_error',
         'mode': 'min', 'patience': 3, 'min_delta': 1e-4},
        # epoch0 sets best=0.5 (improvement from inf). epochs 1,2,3 don't improve.
        # stale increments to 3 at epoch 3 -> break.
        [0.5, 0.6, 0.55, 0.51, 0.49, 0.48],
        expect_stop_at=3,
        expect_best=0.5,
    )

    print("Scenario C: mode=min, improvement smaller than min_delta is NOT improvement")
    c = case(
        "min_delta_threshold",
        {'enabled': True, 'monitor_key': 'val_action_mse_error',
         'mode': 'min', 'patience': 2, 'min_delta': 0.1},
        # epoch0 best=1.0. epoch1 v=0.95 (delta 0.05 < 0.1) -> not improved.
        # epoch2 v=0.94 -> not improved. stale=2 -> break.
        [1.0, 0.95, 0.94, 0.5, 0.4],
        expect_stop_at=2,
        expect_best=1.0,
    )

    print("Scenario D: mode=max, monotonic increase -> never stops")
    d = case(
        "monotonic_increase_max",
        {'enabled': True, 'monitor_key': 'val_reward',
         'mode': 'max', 'patience': 2, 'min_delta': 0.0},
        [0.1, 0.2, 0.3, 0.4],
        expect_stop_at=None,
        expect_best=0.4,
    )

    print("Scenario E: mode=max, regression triggers patience")
    e = case(
        "regression_max",
        {'enabled': True, 'monitor_key': 'val_reward',
         'mode': 'max', 'patience': 2, 'min_delta': 0.0},
        # epoch0 best=0.5. epoch1 v=0.4 not improved (stale=1).
        # epoch2 v=0.45 not improved (stale=2) -> break.
        [0.5, 0.4, 0.45, 0.99],
        expect_stop_at=2,
        expect_best=0.5,
    )

    print("Scenario F: missing monitor_key raises KeyError")
    f = case(
        "missing_monitor_key",
        {'enabled': True, 'monitor_key': 'val_action_mse_error',
         'mode': 'min', 'patience': 3, 'min_delta': 1e-4},
        [0.5, None],
        expect_stop_at='KeyError',
    )

    print("Scenario G: disabled -> never stops, never errors on missing key")
    g = case(
        "disabled",
        {'enabled': False, 'monitor_key': 'val_action_mse_error',
         'mode': 'min', 'patience': 1, 'min_delta': 1e-4},
        [None, None, None],
        expect_stop_at=None,
    )

    print("Scenario H: patience=1 fires on first non-improvement")
    h = case(
        "patience_1",
        {'enabled': True, 'monitor_key': 'val_action_mse_error',
         'mode': 'min', 'patience': 1, 'min_delta': 1e-4},
        # epoch0 best=0.5. epoch1 v=0.6 stale=1 -> break.
        [0.5, 0.6, 0.4],
        expect_stop_at=1,
        expect_best=0.5,
    )

    all_ok = all([a, b, c, d, e, f, g, h])
    print()
    print("ALL PASSED" if all_ok else "SOME FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
