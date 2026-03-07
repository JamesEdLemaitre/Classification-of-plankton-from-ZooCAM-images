#!/usr/bin/env python3
"""
Re-run inference (TTA) + Kaggle submit for 6 already-trained models.
Fixes the bug where all 3 batch-1 inferences used the same logdir.
"""
import os
import sys
import time
import glob
import yaml
import subprocess
import copy
import tempfile
import re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "src", "logs")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
PYTHON = "/home/lmalerba/.conda/envs/hub07_py311/bin/python"
KAGGLE = "/home/lmalerba/.conda/envs/hub07_py311/bin/kaggle"
COMPETITION = "3-md-4040-2026-challenge"

# Mapping: config_file → logdir (verified from config.yaml in each logdir)
JOBS = [
    ("config_test_heavy_cutout_asymmetric.yaml",  "TimmModel_5"),
    ("config_test_ultra_heavy_polyfocal.yaml",     "TimmModel_6"),
    ("config_test_heavy_color_dicece.yaml",        "TimmModel_7"),
    ("config_test_heavy_spatial_ghmc.yaml",        "TimmModel_8"),
    ("config_test_heavy_cutout_polyfocal.yaml",    "TimmModel_9"),
    ("config_test_ultra_heavy_dicece.yaml",        "TimmModel_10"),
]


def find_submission_after(results_dir, after_time):
    pattern = os.path.join(results_dir, "submission_tta_*.csv")
    files = [f for f in glob.glob(pattern) if os.path.getmtime(f) > after_time]
    files.sort(key=os.path.getmtime)
    return files[-1] if files else None


def extract_best_f1(logdir):
    best_f1 = "N/A"
    try:
        for lf in glob.glob(os.path.join(logdir, "*.txt")):
            with open(lf, 'r') as f:
                matches = re.findall(r"'f1': ([\d.]+).*BEST", f.read())
                if matches:
                    best_f1 = max(matches, key=float)
    except Exception:
        pass
    return best_f1


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp_run = time.strftime("%Y%m%d-%H%M%S")

    print(f"=== RE-SUBMIT RUN {timestamp_run} ===")
    print(f"    {len(JOBS)} models to infer + submit\n")

    for config_name, logdir_name in JOBS:
        config_path = os.path.join(PROJECT_ROOT, config_name)
        logdir = os.path.join(LOGS_DIR, logdir_name)
        model_path = os.path.join(logdir, "best_model.pt")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        aug = cfg["data"]["sources"]["original_dataset"]["augmentations_version"]
        loss_cls = cfg["loss"]["class"] if isinstance(cfg["loss"], dict) else cfg["loss"]
        tag = f"{aug}__{loss_cls}"

        val_f1 = extract_best_f1(logdir)
        print(f"  [{time.strftime('%H:%M:%S')}] INFER: {config_name} (logdir={logdir_name}, val_f1={val_f1})")

        if not os.path.exists(model_path):
            print(f"    ERROR: {model_path} not found, skipping")
            continue

        # Build inference config with model_path
        infer_cfg = copy.deepcopy(cfg)
        infer_cfg["model"]["model_path"] = model_path
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.yaml', dir=PROJECT_ROOT)
        with os.fdopen(tmp_fd, 'w') as tmp:
            yaml.dump(infer_cfg, tmp)

        t0 = time.time()
        result = subprocess.run(
            [PYTHON, "-m", "torchtmpl.main", tmp_path, "generate_csv"],
            cwd=SRC_DIR, capture_output=True, text=True,
        )
        os.unlink(tmp_path)

        if result.returncode != 0:
            print(f"    INFER FAILED: {result.stderr[-300:]}")
            continue

        submission_csv = find_submission_after(RESULTS_DIR, t0 - 5)
        if not submission_csv:
            submission_csv = find_submission_after(os.path.join(SRC_DIR, "results"), t0 - 5)
        if not submission_csv:
            print(f"    ERROR: No submission CSV found")
            continue

        print(f"    SUBMIT: {os.path.basename(submission_csv)}")
        rc = subprocess.run(
            [KAGGLE, "competitions", "submit", "-c", COMPETITION, "-f", submission_csv,
             "-m", f"resubmit {tag} {logdir_name} {timestamp_run}"],
            capture_output=True, text=True,
        )

        if rc.returncode != 0:
            print(f"    SUBMIT FAILED: {rc.stderr[-200:]}")
        else:
            print(f"    OK: submitted")

        # Small pause between submissions
        time.sleep(2)

    print(f"\n=== DONE ===")


if __name__ == "__main__":
    main()
