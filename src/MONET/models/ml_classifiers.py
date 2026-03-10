"""
Traditional ML classifiers (Random Forest, LightGBM, XGBoost, CatBoost)
evaluated on frozen backbone features alongside the linear probe.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
)

log = logging.getLogger(__name__)


def extract_features(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from the frozen backbone for all samples in the dataloader.

    Returns:
        features: np.ndarray of shape (N, D)
        labels: np.ndarray of shape (N,)
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"]

            # Extract features from backbone
            backbone = model.backbone
            if backbone.__class__.__name__ == "CLIP":
                features = backbone.encode_image(images).float()
            elif backbone.__class__.__name__ == "ResNet":
                features = backbone(images)
            else:
                raise NotImplementedError(
                    f"Feature extraction not supported for {backbone.__class__.__name__}"
                )

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, max_fpr: float = 0.2) -> Dict[str, float]:
    """Compute binary classification metrics."""
    metrics = {}
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["cohenkappa"] = cohen_kappa_score(y_true, y_pred)

    # AUROC & pAUC — need at least 2 classes in y_true
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
        pauc_normalized = roc_auc_score(y_true, y_prob, max_fpr=max_fpr)
        metrics["pauc"] = pauc_normalized * max_fpr
    else:
        metrics["auroc"] = float("nan")
        metrics["pauc"] = float("nan")

    return metrics


def _get_classifiers(pos_weight: float) -> List[Tuple[str, object]]:
    """Instantiate all ML classifiers. pos_weight = n_neg / n_pos."""
    classifiers = []

    # 1) Random Forest
    classifiers.append((
        "RandomForest",
        RandomForestClassifier(
            n_estimators=500,
            class_weight={0: 1.0, 1: pos_weight},
            n_jobs=-1,
            random_state=42,
        ),
    ))

    # 2) LightGBM
    try:
        import lightgbm as lgb

        classifiers.append((
            "LightGBM",
            lgb.LGBMClassifier(
                n_estimators=500,
                scale_pos_weight=pos_weight,
                learning_rate=0.05,
                num_leaves=31,
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            ),
        ))
    except ImportError:
        log.warning("LightGBM not installed — skipping. Install with: pip install lightgbm")

    # 3) XGBoost
    try:
        import xgboost as xgb

        classifiers.append((
            "XGBoost",
            xgb.XGBClassifier(
                n_estimators=500,
                scale_pos_weight=pos_weight,
                learning_rate=0.05,
                max_depth=6,
                n_jobs=-1,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False,
            ),
        ))
    except ImportError:
        log.warning("XGBoost not installed — skipping. Install with: pip install xgboost")

    # 4) CatBoost
    try:
        from catboost import CatBoostClassifier

        classifiers.append((
            "CatBoost",
            CatBoostClassifier(
                iterations=500,
                class_weights={0: 1.0, 1: pos_weight},
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
            ),
        ))
    except ImportError:
        log.warning("CatBoost not installed — skipping. Install with: pip install catboost")

    return classifiers


def run_ml_classifiers(
    model,
    datamodule,
    device,
    target_type: str = "binary",
    output_dir: str = None,
) -> Dict[str, Dict[str, float]]:
    """Extract features and run all ML classifiers.

    Args:
        model: The trained ClassifierLitModule (backbone + linear head).
        datamodule: The LightningDataModule with train/val/test dataloaders.
        device: torch device.
        target_type: 'binary' supported for now.
        output_dir: optional directory to save results.

    Returns:
        Dict mapping classifier name → dict of metrics.
    """
    if target_type != "binary":
        log.warning(f"ML classifiers only support binary classification, got '{target_type}'. Skipping.")
        return {}

    model = model.to(device)

    log.info("Extracting train features...")
    X_train, y_train = extract_features(model, datamodule.train_dataloader(), device)
    log.info(f"  Train features: {X_train.shape}, positives: {y_train.sum()}/{len(y_train)}")

    log.info("Extracting val features...")
    X_val, y_val = extract_features(model, datamodule.val_dataloader(), device)
    log.info(f"  Val features: {X_val.shape}, positives: {y_val.sum()}/{len(y_val)}")

    log.info("Extracting test features...")
    X_test, y_test = extract_features(model, datamodule.test_dataloader(), device)
    log.info(f"  Test features: {X_test.shape}, positives: {y_test.sum()}/{len(y_test)}")

    # Compute class weight
    n_pos = max(y_train.sum(), 1)
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / n_pos
    log.info(f"Class imbalance — neg:pos = {n_neg}:{n_pos} (pos_weight={pos_weight:.1f})")

    classifiers = _get_classifiers(pos_weight)
    all_results = {}

    for name, clf in classifiers:
        log.info(f"Training {name}...")
        try:
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test)[:, 1]
            else:
                y_prob = clf.decision_function(X_test)

            test_metrics = compute_binary_metrics(y_test, y_pred, y_prob)
            all_results[name] = test_metrics

            # Also compute val metrics
            y_val_pred = clf.predict(X_val)
            if hasattr(clf, "predict_proba"):
                y_val_prob = clf.predict_proba(X_val)[:, 1]
            else:
                y_val_prob = clf.decision_function(X_val)
            val_metrics = compute_binary_metrics(y_val, y_val_pred, y_val_prob)
            all_results[f"{name}_val"] = val_metrics

        except Exception as e:
            log.error(f"Error training {name}: {e}")
            continue

    # Print results table
    _print_results_table(all_results)

    # Save results if output_dir provided
    if output_dir:
        import json
        from pathlib import Path

        results_path = Path(output_dir) / "ml_classifiers_results.json"
        # Convert to serializable format
        serializable = {k: {mk: float(mv) for mk, mv in v.items()} for k, v in all_results.items()}
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        log.info(f"ML classifier results saved to {results_path}")

    return all_results


def _print_results_table(results: Dict[str, Dict[str, float]]):
    """Print a formatted comparison table of all classifiers."""
    if not results:
        return

    # Filter to test results only (no _val suffix)
    test_results = {k: v for k, v in results.items() if not k.endswith("_val")}
    val_results = {k.replace("_val", ""): v for k, v in results.items() if k.endswith("_val")}

    metrics = ["balanced_accuracy", "auroc", "pauc", "precision", "recall", "f1", "cohenkappa"]

    header = f"{'Classifier':<16}"
    for m in metrics:
        header += f" {m:>12}"
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print("ML CLASSIFIERS — TEST SET RESULTS")
    print("=" * len(header))
    print(header)
    print(separator)
    for name, mdict in test_results.items():
        row = f"{name:<16}"
        for m in metrics:
            val = mdict.get(m, float("nan"))
            row += f" {val:>12.6f}"
        print(row)
    print(separator)

    if val_results:
        print(f"\n{'ML CLASSIFIERS — VALIDATION SET RESULTS'}")
        print("=" * len(header))
        print(header)
        print(separator)
        for name, mdict in val_results.items():
            row = f"{name:<16}"
            for m in metrics:
                val = mdict.get(m, float("nan"))
                row += f" {val:>12.6f}"
            print(row)
        print(separator)
    print()
