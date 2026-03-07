#!/usr/bin/env python3
"""
Ensemble from cached TTA logits and submit to Kaggle.

For each combination:
  1. Average softmax probabilities across models (softmax per TTA view, then
     average across 5 views, then average across models)
  2. Argmax → predictions
  3. Save CSV + submit to Kaggle

Usage:
    cd kaggle_challenge_sdi
    /home/lmalerba/.conda/envs/hub07_py311/bin/python ensemble_logits_submit.py
"""
import os
import sys
import glob
import time
import itertools
import numpy as np
import pandas as pd
import yaml
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOGITS_DIR = os.path.join(PROJECT_ROOT, "src", "results", "logits_cache")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "ensemble_logits")
KAGGLE = "/home/lmalerba/.conda/envs/hub07_py311/bin/kaggle"
COMPETITION = "3-md-4040-2026-challenge"
MIN_MODELS = 6
MAX_MODELS = 6

# Fixed ensemble: set to None for combinatorial mode, or a list of tags for single ensemble
FIXED_ENSEMBLE = ["dinov3_p2", "eva02clip_p2", "convnext_ar_p2", "efficientnet_p2"]


def load_all_models():
    """Load all cached logits and return dict of {tag: softmax_probs}."""
    meta_files = sorted(glob.glob(os.path.join(LOGITS_DIR, "*_meta.yaml")))
    models = {}
    ref_fnames = None

    for meta_path in meta_files:
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        tag = meta["model_tag"]
        logits_path = os.path.join(LOGITS_DIR, f"{tag}_logits.npy")
        fnames_path = os.path.join(LOGITS_DIR, f"{tag}_filenames.npy")

        if not os.path.isfile(logits_path):
            print(f"  SKIP {tag}: logits file missing")
            continue

        logits = np.load(logits_path)  # (N, n_tta, C)
        fnames = np.load(fnames_path, allow_pickle=True)

        # Softmax per TTA view, then average (matches original generate_csv)
        # Original 5-TTA: original(0), flip_h(1), flip_v(2), rot90(3), rot270(5)
        tta5_idx = [0, 1, 2, 3, 5]
        logits_5 = logits[:, tta5_idx, :]  # (N, 5, C)
        # Numerically stable softmax per view
        shifted = logits_5 - logits_5.max(axis=2, keepdims=True)
        exp_l = np.exp(shifted)
        per_view_probs = exp_l / exp_l.sum(axis=2, keepdims=True)  # (N, 5, C)
        probs = per_view_probs.mean(axis=1)  # (N, C)

        models[tag] = probs
        print(f"  Loaded {tag}: {meta['backbone']:40s} probs={probs.shape} (5-TTA softmax avg)")

        if ref_fnames is None:
            ref_fnames = fnames
        else:
            assert len(fnames) == len(ref_fnames), \
                f"Filename count mismatch: {tag} has {len(fnames)} vs {len(ref_fnames)}"

    return models, ref_fnames


def ensemble_predict(model_probs_list):
    """Average softmax probs across models → argmax."""
    avg = np.mean(model_probs_list, axis=0)  # (N, C)
    return avg.argmax(axis=1)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    print(f"{'='*70}")
    print(f"  ENSEMBLE LOGITS → KAGGLE SUBMIT")
    print(f"  Loading models from {LOGITS_DIR}")
    print(f"{'='*70}\n")

    models, filenames = load_all_models()
    tags = sorted(models.keys())
    n_models = len(tags)

    # Fixed ensemble mode or combinatorial mode
    if FIXED_ENSEMBLE is not None:
        missing = [t for t in FIXED_ENSEMBLE if t not in models]
        if missing:
            print(f"  ERROR: missing models: {missing}")
            print(f"  Available: {tags}")
            sys.exit(1)
        combos = [tuple(FIXED_ENSEMBLE)]
        total = 1
        print(f"\n  Fixed ensemble: {' + '.join(FIXED_ENSEMBLE)}\n")
    else:
        if n_models < MIN_MODELS:
            print(f"Only {n_models} models found, need at least {MIN_MODELS}. Aborting.")
            sys.exit(1)
        combos = list(itertools.chain.from_iterable(
            itertools.combinations(tags, k)
            for k in range(MIN_MODELS, MAX_MODELS + 1)
        ))
        total = len(combos)
        print(f"\n  {n_models} models → {total} combinations (size {MIN_MODELS}..{MAX_MODELS})\n")
    print(f"{'='*70}")

    submissions = []
    idx = 0

    for combo in combos:
        idx += 1
        k = len(combo)
        combo_name = "+".join(combo)
        short_name = f"ens{k}_{'+'.join(c.split('_')[0] for c in combo)}"

        probs_list = [models[tag] for tag in combo]
        preds = ensemble_predict(probs_list)

        # Save CSV
        csv_name = f"ens_{idx:03d}_{short_name}_{timestamp}.csv"
        csv_path = os.path.join(RESULTS_DIR, csv_name)
        df = pd.DataFrame({
            "imgname": filenames,
            "label": preds.astype(int),
        })
        df.to_csv(csv_path, index=False)

        # Submit to Kaggle
        msg = f"ensemble {k}/{n_models}: {combo_name}"
        rc = subprocess.run(
            [KAGGLE, "competitions", "submit",
             "-c", COMPETITION, "-f", csv_path, "-m", msg],
            capture_output=True, text=True,
        )

        status = "OK" if rc.returncode == 0 else "FAIL"
        print(f"  [{idx:3d}/{total}] k={k} {short_name:50s} → {status}")

        if rc.returncode != 0:
            print(f"           {rc.stderr.strip()[:150]}")

        submissions.append({
            "idx": idx, "k": k, "combo": combo_name,
            "csv": csv_name, "status": status,
        })

        # Kaggle rate limit: small pause between submissions
        time.sleep(3)

    # Summary
    print(f"\n{'='*70}")
    print(f"  DONE — {len(submissions)} submissions")
    print(f"  CSVs in: {RESULTS_DIR}")
    ok = sum(1 for s in submissions if s["status"] == "OK")
    print(f"  Submitted: {ok}/{len(submissions)}")
    print(f"{'='*70}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, f"summary_{timestamp}.yaml")
    with open(summary_path, "w") as f:
        yaml.dump({"models": tags, "n_combinations": len(submissions),
                    "submissions": submissions}, f)
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
