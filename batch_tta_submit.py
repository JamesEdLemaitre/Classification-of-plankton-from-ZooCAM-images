#!/usr/bin/env python3
"""
Batch inference (8-TTA) + Kaggle submit for all models in src/logs/.

For each TimmModel_* folder found:
  1. Load config.yaml from the logdir
  2. Inject model_path = logdir/best_model.pt
  3. Run inference with 8 TTA transformations:
       Original, FlipH, FlipV, Rot90, Rot180, Rot270, FlipH+V, Rot90+FlipH
  4. Submit the CSV to Kaggle

Usage:
    python batch_tta_submit.py
"""
import os
import sys
import time
import glob
import copy
import yaml
import subprocess
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(PROJECT_ROOT, "src", "logs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
PYTHON = "/home/lmalerba/.conda/envs/hub07_py311/bin/python"
KAGGLE = "/home/lmalerba/.conda/envs/hub07_py311/bin/kaggle"
COMPETITION = "3-md-4040-2026-challenge"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Discover all model folders
    model_dirs = sorted(glob.glob(os.path.join(LOGS_DIR, "TimmModel_*")))
    if not model_dirs:
        print(f"No TimmModel_* folders found in {LOGS_DIR}")
        sys.exit(1)

    timestamp_run = time.strftime("%Y%m%d-%H%M%S")
    print(f"{'='*70}")
    print(f"  BATCH TTA-8 INFERENCE + KAGGLE SUBMIT")
    print(f"  {len(model_dirs)} models found — {timestamp_run}")
    print(f"{'='*70}\n")

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        config_path = os.path.join(model_dir, "config.yaml")
        model_path = os.path.join(model_dir, "best_model.pt")

        if not os.path.isfile(config_path):
            print(f"  SKIP {model_name}: no config.yaml")
            continue
        if not os.path.isfile(model_path):
            print(f"  SKIP {model_name}: no best_model.pt")
            continue

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        backbone = cfg.get("model", {}).get("backbone", "?")
        loss_cls = cfg.get("loss", {})
        if isinstance(loss_cls, dict):
            loss_cls = loss_cls.get("class", "?")

        print(f"  [{time.strftime('%H:%M:%S')}] {model_name}  "
              f"backbone={backbone}  loss={loss_cls}")

        # Build temp config with model_path + generate_csv_tta8 command
        infer_cfg = copy.deepcopy(cfg)
        infer_cfg["model"]["model_path"] = model_path
        infer_cfg["tta_n_transforms"] = 8

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".yaml", dir=PROJECT_ROOT)
        with os.fdopen(tmp_fd, "w") as tmp:
            yaml.dump(infer_cfg, tmp)

        t0 = time.time()
        result = subprocess.run(
            [PYTHON, "-m", "torchtmpl.main", tmp_path, "generate_csv_tta8"],
            cwd=SRC_DIR, capture_output=True, text=True,
        )
        os.unlink(tmp_path)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"    FAILED ({elapsed:.0f}s)")
            # Show last lines of stderr
            err_lines = result.stderr.strip().split("\n")
            for line in err_lines[-5:]:
                print(f"    | {line}")
            continue

        # Find the submission CSV produced
        submission_csv = _find_submission_after(t0 - 5)
        if not submission_csv:
            print(f"    ERROR: no submission CSV found after inference")
            continue

        print(f"    CSV: {os.path.basename(submission_csv)} ({elapsed:.0f}s)")

        # Submit to Kaggle
        msg = f"tta8 {model_name} {backbone} {loss_cls} {timestamp_run}"
        rc = subprocess.run(
            [KAGGLE, "competitions", "submit",
             "-c", COMPETITION, "-f", submission_csv, "-m", msg],
            capture_output=True, text=True,
        )
        if rc.returncode != 0:
            print(f"    SUBMIT FAILED: {rc.stderr.strip()[:200]}")
        else:
            print(f"    SUBMITTED ✓")

        time.sleep(2)

    print(f"\n{'='*70}")
    print(f"  DONE — all models processed")
    print(f"{'='*70}")


def _find_submission_after(after_time):
    """Find the most recent submission_tta8_*.csv created after after_time."""
    for search_dir in [RESULTS_DIR, os.path.join(SRC_DIR, "results")]:
        pattern = os.path.join(search_dir, "submission_tta8_*.csv")
        files = [f for f in glob.glob(pattern) if os.path.getmtime(f) > after_time]
        if files:
            files.sort(key=os.path.getmtime)
            return files[-1]
    return None


if __name__ == "__main__":
    main()
