"""
Run ML classifiers on pre-computed Derm Foundation embeddings for ISIC 2024.

Usage:
    python scripts/run_derm_foundation_classifiers.py \
        --embeddings /path/to/isic2024_derm_foundation_embeddings.npz \
        --metadata data/isic2024/final_metadata_all.csv \
        --split_seed 42 \
        --output_dir .
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Add src/ to path so we can import from MONET
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from MONET.models.ml_classifiers import (
    _get_classifiers,
    _print_results_table,
    compute_binary_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Classify ISIC 2024 using Derm Foundation embeddings")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="/home/mecha109/PanDerm/Evaluation_datasets/ISIC2024/isic2024_derm_foundation_embeddings.npz",
        help="Path to the .npz file with 'embeddings' and 'filenames' arrays.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "isic2024" / "final_metadata_all.csv"),
        help="Path to final_metadata_all.csv.",
    )
    parser.add_argument("--split_seed", type=int, default=42, help="Random seed for patient-aware split.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results JSON.")
    args = parser.parse_args()

    # ── Load embeddings ──
    log.info(f"Loading embeddings from {args.embeddings}")
    data = np.load(args.embeddings)
    embeddings = data["embeddings"]
    filenames = data["filenames"]
    log.info(f"  Loaded {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

    # Map filename → isic_id (strip .jpg extension)
    isic_ids = np.array([Path(f).stem for f in filenames])

    # ── Load metadata ──
    log.info(f"Loading metadata from {args.metadata}")
    meta_df = pd.read_csv(args.metadata, low_memory=False)
    meta_df = meta_df.set_index("isic_id")

    # ── Align embeddings with metadata ──
    # Build a DataFrame to join embeddings with labels/patient_id
    embed_df = pd.DataFrame({"isic_id": isic_ids})
    embed_df = embed_df.set_index("isic_id")

    # Keep only samples present in both embeddings and metadata
    common_ids = embed_df.index.intersection(meta_df.index)
    log.info(f"  Samples in embeddings: {len(isic_ids)}, in metadata: {len(meta_df)}, matched: {len(common_ids)}")

    meta_df = meta_df.loc[common_ids]

    # Build aligned arrays
    id_to_idx = {iid: i for i, iid in enumerate(isic_ids)}
    ordered_indices = [id_to_idx[iid] for iid in common_ids]
    X = embeddings[ordered_indices]
    y = meta_df["target"].values.astype(int)
    patient_ids = meta_df["patient_id"].values

    log.info(f"  Final dataset: {X.shape[0]} samples, {y.sum()} positives ({100*y.mean():.2f}%)")

    # ── Patient-aware train/val split (same as setup_isic2024) ──
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.split_seed)
    train_mask, val_mask = next(gss.split(X, groups=patient_ids))

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    # test = val (same as the original setup_isic2024)
    X_test, y_test = X_val, y_val

    # Verify no patient overlap
    train_patients = set(patient_ids[train_mask])
    val_patients = set(patient_ids[val_mask])
    assert train_patients.isdisjoint(val_patients), "Patient leakage detected!"

    log.info(
        f"  Split: {len(X_train)} train ({len(train_patients)} patients), "
        f"{len(X_val)} val ({len(val_patients)} patients). No patient overlap."
    )
    log.info(f"  Train positives: {y_train.sum()}/{len(y_train)}")
    log.info(f"  Val/Test positives: {y_val.sum()}/{len(y_val)}")

    # ── Class weight ──
    n_pos = max(y_train.sum(), 1)
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / n_pos
    log.info(f"  Class imbalance — neg:pos = {n_neg}:{n_pos} (pos_weight={pos_weight:.1f})")

    # ── Run classifiers ──
    classifiers = _get_classifiers(pos_weight)
    all_results = {}

    for name, clf in classifiers:
        log.info(f"Training {name}...")
        try:
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
            all_results[name] = compute_binary_metrics(y_test, y_pred, y_prob)

            y_val_pred = clf.predict(X_val)
            y_val_prob = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_val)
            all_results[f"{name}_val"] = compute_binary_metrics(y_val, y_val_pred, y_val_prob)

        except Exception as e:
            log.error(f"Error training {name}: {e}")
            continue

    _print_results_table(all_results)

    # ── Save results ──
    if args.output_dir:
        results_path = Path(args.output_dir) / "derm_foundation_ml_results.json"
        serializable = {k: {mk: float(mv) for mk, mv in v.items()} for k, v in all_results.items()}
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        log.info(f"Results saved to {results_path}")

    return all_results


if __name__ == "__main__":
    main()
