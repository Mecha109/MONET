"""
Apply per-patient 10-90 percentile normalization to pre-computed embeddings.

For each patient, and for each feature dimension, values are clipped to
that patient's [p10, p90] range and linearly scaled to [0, 1].
Patients with a single image are left at 0 (no range to normalize).

Usage:
    python scripts/normalize_embeddings.py \
        --input /path/to/isic2024_derm_foundation_embeddings.npz \
        --metadata data/isic2024/final_metadata_all.csv \
        --output /path/to/isic2024_derm_foundation_embeddings_normalized.npz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def percentile_normalize(embeddings: np.ndarray, low: float = 10, high: float = 90) -> np.ndarray:
    """Normalize each feature to [0, 1] using the low-high percentile range.

    Values below/above the percentiles are clipped to 0/1.
    """
    p_low = np.percentile(embeddings, low, axis=0)
    p_high = np.percentile(embeddings, high, axis=0)

    denom = p_high - p_low
    # Avoid division by zero for constant features
    denom[denom == 0] = 1.0

    normalized = (embeddings - p_low) / denom
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


def per_patient_normalize(embeddings: np.ndarray, patient_ids: np.ndarray,
                          low: float = 10, high: float = 90) -> np.ndarray:
    """Normalize embeddings per patient using percentile normalization.

    For each patient group, compute the low/high percentiles across that
    patient's samples and normalize independently.  Patients with fewer
    than 2 images get all-zero embeddings (no meaningful range).
    """
    normalized = np.zeros_like(embeddings)
    unique_patients = np.unique(patient_ids)

    for pid in unique_patients:
        mask = patient_ids == pid
        group = embeddings[mask]

        if group.shape[0] < 2:
            # Single image — no spread to normalize; leave as zeros
            continue

        normalized[mask] = percentile_normalize(group, low=low, high=high)

    return normalized


def main():
    parser = argparse.ArgumentParser(description="Per-patient 10-90 percentile normalization of embeddings")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/mecha109/PanDerm/Evaluation_datasets/ISIC2024/isic2024_derm_foundation_embeddings.npz",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "isic2024" / "final_metadata_all.csv"),
        help="Path to final_metadata_all.csv (for patient_id mapping).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path. Defaults to <input_stem>_normalized.npz alongside the input file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_normalized.npz"
    else:
        output_path = Path(args.output)

    # Load embeddings
    data = np.load(input_path)
    embeddings = data["embeddings"]
    filenames = data["filenames"]
    print(f"Loaded {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

    # Load metadata to get patient_id per sample
    meta_df = pd.read_csv(args.metadata, low_memory=False, usecols=["isic_id", "patient_id"])
    meta_df = meta_df.set_index("isic_id")

    # Map each embedding to its patient_id
    isic_ids = np.array([Path(f).stem for f in filenames])
    id_series = pd.Series(isic_ids, name="isic_id")
    patient_ids = id_series.map(meta_df["patient_id"]).values

    n_missing = pd.isna(patient_ids).sum()
    if n_missing > 0:
        print(f"Warning: {n_missing} samples have no patient_id in metadata — they will be left unnormalized.")
        patient_ids = np.where(pd.isna(patient_ids), "__UNKNOWN__", patient_ids).astype(str)

    unique_patients = np.unique(patient_ids)
    print(f"Found {len(unique_patients)} unique patients")

    normalized = per_patient_normalize(embeddings, patient_ids, low=10, high=90)
    print(f"Applied per-patient 10-90 percentile normalization")

    np.savez(output_path, embeddings=normalized, filenames=filenames)
    print(f"Saved normalized embeddings to {output_path}")


if __name__ == "__main__":
    main()
