#!/usr/bin/env python3
"""
Batch cache TTA logits for all trained models.

For each model config, runs cache_tta_logits to save raw logits
(N_test, 8, 86) to results/logits_cache/.

Usage:
    cd kaggle_challenge_sdi/src
    /home/lmalerba/.conda/envs/hub07_py311/bin/python ../batch_cache_logits.py
"""
import os
import sys
import time
import yaml
import copy
import tempfile
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
PYTHON = "/home/lmalerba/.conda/envs/hub07_py311/bin/python"

# (config_file, command, model_tag)
# command is the training command used — determines which model builder to use
# model_tag is a human-readable name for the cached logits
JOBS = [
    # Phase 2 models (trained with valid_ratio=0.2)
    ("config_phase2_convnext_ar.yaml",   "cache_tta_logits", "convnext_ar_p2"),
    ("config_phase2_dinov3.yaml",        "cache_tta_logits", "dinov3_p2"),
    ("config_phase2_efficientnet.yaml",  "cache_tta_logits", "efficientnet_p2"),
    ("config_phase2_eva02clip.yaml",     "cache_tta_logits", "eva02clip_p2"),
    ("config_phase2_resnext50_ar.yaml",  "cache_tta_logits", "resnext50_ar_p2"),
    # Phase 3 models (trained with valid_ratio=0, full data)
    ("config_phase3_convnext_ar.yaml",   "cache_tta_logits", "convnext_ar_p3"),
    ("config_phase3_dinov3.yaml",        "cache_tta_logits", "dinov3_p3"),
    # config_phase3_eva02clip.yaml — skipped: weights not saved yet (DINOv3Model_10)
]


def main():
    print(f"{'='*70}")
    print(f"  BATCH CACHE TTA LOGITS — {len(JOBS)} models")
    print(f"{'='*70}\n")

    for config_name, command, model_tag in JOBS:
        config_path = os.path.join(PROJECT_ROOT, config_name)
        if not os.path.isfile(config_path):
            print(f"  SKIP {config_name}: config file not found")
            continue

        print(f"  [{time.strftime('%H:%M:%S')}] {model_tag:25s} ({config_name})")

        # Build temp config with model_tag injected + absolute model_path
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg["model_tag"] = model_tag

        # Resolve relative model_path to absolute (relative to PROJECT_ROOT)
        mp = cfg.get("model", {}).get("model_path", "")
        if mp and not os.path.isabs(mp):
            cfg["model"]["model_path"] = os.path.join(PROJECT_ROOT, mp)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".yaml", dir=PROJECT_ROOT)
        with os.fdopen(tmp_fd, "w") as tmp:
            yaml.dump(cfg, tmp)

        t0 = time.time()
        result = subprocess.run(
            [PYTHON, "-m", "torchtmpl.main", tmp_path, command],
            cwd=SRC_DIR,
            capture_output=True, text=True,
        )
        os.unlink(tmp_path)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"    FAILED ({elapsed:.0f}s)")
            err_lines = result.stderr.strip().split("\n")
            for line in err_lines[-8:]:
                print(f"    | {line}")
        else:
            # Find the "Saved:" line in output
            for line in result.stderr.split("\n"):
                if "logits :" in line or "Saved:" in line:
                    print(f"    {line.strip()}")
            print(f"    OK ({elapsed:.0f}s)")

        print()

    print(f"{'='*70}")
    print(f"  DONE — logits cached in src/results/logits_cache/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
