#!/usr/bin/env python3
"""Train the 2 remaining configs, infer, and submit to Kaggle."""
import os
import sys
import time
import glob
import yaml
import subprocess
import copy
import tempfile
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "src", "logs")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
PYTHON = "/home/lmalerba/.conda/envs/hub07_py311/bin/python"
KAGGLE = "/home/lmalerba/.conda/envs/hub07_py311/bin/kaggle"
COMPETITION = "3-md-4040-2026-challenge"

CONFIGS = [
    "config_test_heavy_color_asymmetric.yaml",
    "config_test_heavy_spatial_polyfocal.yaml",
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


def train_one(config_name):
    config_path = os.path.join(PROJECT_ROOT, config_name)
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START training: {config_name}", flush=True)

    logfile = os.path.join(RESULTS_DIR, f"train_{config_name}.log")
    with open(logfile, "w") as lf:
        result = subprocess.run(
            [PYTHON, "-m", "torchtmpl.main", config_path, "train"],
            cwd=SRC_DIR, stdout=lf, stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )

    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[{time.strftime('%H:%M:%S')}] FAILED: {config_name} (exit {result.returncode}, {elapsed:.0f}s)", flush=True)
        return config_name, None, False

    logdir = None
    with open(logfile, 'r') as lf:
        for line in lf:
            m = re.search(r"Will be logging into (.+)", line)
            if m:
                logdir = m.group(1).strip()
                break
    if logdir and not os.path.isabs(logdir):
        logdir = os.path.join(SRC_DIR, logdir)

    print(f"[{time.strftime('%H:%M:%S')}] DONE: {config_name} ({elapsed:.0f}s) → {logdir}", flush=True)
    return config_name, logdir, True


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"=== BATCH 3 — {timestamp} ===")
    print(f"    Training {len(CONFIGS)} configs in parallel\n")

    # Parallel training
    trained = {}
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(train_one, c): c for c in CONFIGS}
        for future in as_completed(futures):
            name, logdir, ok = future.result()
            if ok and logdir:
                trained[name] = logdir

    print(f"\nTraining done: {len(trained)}/{len(CONFIGS)} succeeded.\n")

    # Sequential inference + submit
    for config_name, logdir in trained.items():
        config_path = os.path.join(PROJECT_ROOT, config_name)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        aug = cfg["data"]["sources"]["original_dataset"]["augmentations_version"]
        loss_cls = cfg["loss"]["class"] if isinstance(cfg["loss"], dict) else cfg["loss"]
        tag = f"{aug}__{loss_cls}"
        val_f1 = extract_best_f1(logdir)

        model_path = os.path.join(logdir, "best_model.pt")
        if not os.path.exists(model_path):
            print(f"  No model in {logdir}, skipping")
            continue

        print(f"  [{time.strftime('%H:%M:%S')}] INFER: {config_name} (val_f1={val_f1})")

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

        csv_path = find_submission_after(RESULTS_DIR, t0 - 5)
        if not csv_path:
            csv_path = find_submission_after(os.path.join(SRC_DIR, "results"), t0 - 5)
        if not csv_path:
            print(f"    No CSV found")
            continue

        print(f"    SUBMIT: {os.path.basename(csv_path)}")
        rc = subprocess.run(
            [KAGGLE, "competitions", "submit", "-c", COMPETITION, "-f", csv_path,
             "-m", f"batch3 {tag} {timestamp}"],
            capture_output=True, text=True,
        )
        if rc.returncode != 0:
            print(f"    SUBMIT FAILED: {rc.stderr[-200:]}")
        else:
            print(f"    OK: submitted")
        time.sleep(2)

    print(f"\n=== DONE ===")


if __name__ == "__main__":
    main()
