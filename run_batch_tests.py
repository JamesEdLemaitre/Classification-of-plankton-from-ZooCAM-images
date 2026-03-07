#!/usr/bin/env python3
"""
Parallel batch runner: train 3 models at once → infer (TTA) → submit to Kaggle.
Shares a single GPU across parallel trainings (ResNet34 @ 128px is lightweight).

Usage:
    conda activate hub07_py311
    nohup python -u kaggle_challenge_sdi/run_batch_tests.py > kaggle_challenge_sdi/results/batch_run.log 2>&1 &
"""
import os
import sys
import time
import glob
import yaml
import subprocess
import csv
import copy
import tempfile
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
COMPETITION = "3-md-4040-2026-challenge"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "src", "logs")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

PYTHON = os.environ.get(
    "BATCH_PYTHON",
    "/home/lmalerba/.conda/envs/hub07_py311/bin/python"
)
KAGGLE = "/home/lmalerba/.conda/envs/hub07_py311/bin/kaggle"

PARALLEL_JOBS = 3

CONFIG_FILES = [
    "config_test_ultra_heavy_polyfocal.yaml",
    "config_test_heavy_cutout_asymmetric.yaml",
    "config_test_heavy_color_dicece.yaml",
    "config_test_heavy_spatial_ghmc.yaml",
    "config_test_ultra_heavy_dicece.yaml",
    "config_test_heavy_cutout_polyfocal.yaml",
    "config_test_heavy_color_asymmetric.yaml",
    "config_test_heavy_spatial_polyfocal.yaml",
]

SUMMARY_FILE = os.path.join(RESULTS_DIR, "batch_test_summary.csv")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def find_logdir_after(logs_base, model_class, after_time):
    """Find the most recent logdir created after `after_time`."""
    pattern = os.path.join(logs_base, f"{model_class}_*")
    dirs = [d for d in glob.glob(pattern) if os.path.getmtime(d) > after_time]
    dirs.sort(key=os.path.getmtime)
    return dirs[-1] if dirs else None


def find_best_model(logdir):
    for name in ("best_model.pt", "final_model.pt"):
        p = os.path.join(logdir, name)
        if os.path.exists(p):
            return p
    return None


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


# ---------------------------------------------------------------------------
# TRAIN ONE CONFIG (called in parallel subprocess)
# ---------------------------------------------------------------------------
def train_one(config_name):
    """Train a single config. Returns (config_name, logdir_path, success).
    
    Parses the logdir directly from the training output ('Will be logging into <path>')
    to avoid collisions when multiple trainings run in parallel.
    """
    config_path = os.path.join(PROJECT_ROOT, config_name)
    t0 = time.time()

    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}]   >> START training: {config_name}", flush=True)

    logfile = os.path.join(RESULTS_DIR, f"train_{config_name}.log")
    with open(logfile, "w") as lf:
        result = subprocess.run(
            [PYTHON, "-m", "torchtmpl.main", config_path, "train"],
            cwd=SRC_DIR,
            stdout=lf, stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )

    elapsed = time.time() - t0
    ts = time.strftime("%H:%M:%S")
    if result.returncode != 0:
        print(f"[{ts}]   >> FAILED training: {config_name} (exit {result.returncode}, {elapsed:.0f}s)", flush=True)
        return config_name, None, False

    # Parse the actual logdir from training output (avoids parallel collision)
    logdir = None
    with open(logfile, 'r') as lf:
        for line in lf:
            m = re.search(r"Will be logging into (.+)", line)
            if m:
                logdir = m.group(1).strip()
                break

    if logdir and not os.path.isabs(logdir):
        logdir = os.path.join(SRC_DIR, logdir)

    print(f"[{ts}]   >> DONE training: {config_name} ({elapsed:.0f}s) → {logdir}", flush=True)
    return config_name, logdir, True


# ---------------------------------------------------------------------------
# INFER + SUBMIT (sequential — one at a time for GPU memory)
# ---------------------------------------------------------------------------
def infer_and_submit(config_name, logdir, timestamp_run):
    config_path = os.path.join(PROJECT_ROOT, config_name)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    aug = cfg["data"]["sources"]["original_dataset"]["augmentations_version"]
    loss_cls = cfg["loss"]["class"] if isinstance(cfg["loss"], dict) else cfg["loss"]
    tag = f"{aug}__{loss_cls}"

    model_path = find_best_model(logdir)
    if not model_path:
        log(f"  No model in {logdir}")
        return {"config": config_name, "aug": aug, "loss": loss_cls,
                "status": "NO_MODEL", "val_f1": "N/A", "kaggle_score": "N/A", "submission": "N/A"}

    val_f1 = extract_best_f1(logdir)
    log(f"  INFER: {config_name} | val_f1={val_f1}")

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
        log(f"  INFER FAILED: {config_name} — {result.stderr[-300:]}")
        return {"config": config_name, "aug": aug, "loss": loss_cls,
                "status": "INFER_FAIL", "val_f1": val_f1, "kaggle_score": "N/A", "submission": "N/A"}

    submission_csv = find_submission_after(RESULTS_DIR, t0 - 5)
    if not submission_csv:
        submission_csv = find_submission_after(os.path.join(SRC_DIR, "results"), t0 - 5)
    if not submission_csv:
        log(f"  No CSV for {config_name}")
        return {"config": config_name, "aug": aug, "loss": loss_cls,
                "status": "NO_CSV", "val_f1": val_f1, "kaggle_score": "N/A", "submission": "N/A"}

    log(f"  SUBMIT: {os.path.basename(submission_csv)}")
    rc = subprocess.run(
        [KAGGLE, "competitions", "submit", "-c", COMPETITION, "-f", submission_csv,
         "-m", f"auto-test {tag} {timestamp_run}"],
        capture_output=True, text=True,
    )

    if rc.returncode != 0:
        log(f"  SUBMIT FAILED: {rc.stderr[-200:]}")
        return {"config": config_name, "aug": aug, "loss": loss_cls,
                "status": "SUBMIT_FAIL", "val_f1": val_f1, "kaggle_score": "N/A",
                "submission": os.path.basename(submission_csv)}

    log(f"  OK: {config_name} submitted | val_f1={val_f1}")
    return {"config": config_name, "aug": aug, "loss": loss_cls,
            "status": "SUBMITTED", "val_f1": val_f1, "kaggle_score": "pending",
            "submission": os.path.basename(submission_csv)}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp_run = time.strftime("%Y%m%d-%H%M%S")
    results = []

    log(f"=== BATCH RUN {timestamp_run} ===")
    log(f"    {len(CONFIG_FILES)} configs, {PARALLEL_JOBS} parallel training jobs")
    log(f"    Competition: {COMPETITION}")

    # Process configs in batches of PARALLEL_JOBS
    for batch_start in range(0, len(CONFIG_FILES), PARALLEL_JOBS):
        batch = CONFIG_FILES[batch_start:batch_start + PARALLEL_JOBS]
        batch_num = batch_start // PARALLEL_JOBS + 1
        log(f"\n{'='*60}")
        log(f"BATCH {batch_num}: Training {len(batch)} models IN PARALLEL")
        for c in batch:
            log(f"  - {c}")
        log(f"{'='*60}")

        # ── Parallel training ──
        trained = {}  # config_name → logdir
        with ProcessPoolExecutor(max_workers=PARALLEL_JOBS) as executor:
            futures = {executor.submit(train_one, cfg): cfg for cfg in batch}
            for future in as_completed(futures):
                config_name, logdir, success = future.result()
                if success and logdir:
                    trained[config_name] = logdir
                else:
                    with open(os.path.join(PROJECT_ROOT, config_name)) as f:
                        cfg = yaml.safe_load(f)
                    results.append({
                        "config": config_name,
                        "aug": cfg["data"]["sources"]["original_dataset"]["augmentations_version"],
                        "loss": cfg["loss"]["class"] if isinstance(cfg["loss"], dict) else cfg["loss"],
                        "status": "TRAIN_FAIL", "val_f1": "N/A",
                        "kaggle_score": "N/A", "submission": "N/A"
                    })

        log(f"Training batch {batch_num} done: {len(trained)}/{len(batch)} succeeded.")

        # ── Sequential inference + submit (one GPU at a time) ──
        for config_name, logdir in trained.items():
            r = infer_and_submit(config_name, logdir, timestamp_run)
            results.append(r)

    # ── FINAL SUMMARY ──
    log(f"\n\n{'='*80}")
    log(f"  BATCH TEST SUMMARY — {timestamp_run}")
    log(f"{'='*80}")
    header = f"{'Config':<45} {'Aug':<16} {'Loss':<18} {'Val F1':<8} {'Kaggle':<10} {'Status'}"
    log(header)
    log("-" * len(header))
    for r in results:
        log(f"{r['config']:<45} {r.get('aug','?'):<16} {r.get('loss','?'):<18} "
            f"{r.get('val_f1','N/A'):<8} {r.get('kaggle_score','N/A'):<10} {r.get('status','?')}")

    with open(SUMMARY_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["config", "aug", "loss", "status", "val_f1", "kaggle_score", "submission"])
        writer.writeheader()
        writer.writerows(results)
    log(f"\nSummary saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
