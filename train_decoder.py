"""
Cross-subject (zero-shot) decoder: train on many people, test on a new person.

Leave-One-Subject-Out (LOSO): each fold trains on 21 subjects and tests on the
held-out subject who was never seen during training. Uses a Transformer model
that learns spatial and temporal patterns from EEG. Reports 10-class and
3-class accuracy, confusion matrices, and an attention topomap for the best fold.

Run: python train_decoder.py [--quick] [--fast] [--ea]
  --quick  Few folds and epochs (pipeline check).
  --fast   Smaller model for CPU.
  --ea     Apply Euclidean Alignment per subject (recommended).
Requires GPU or long runtime on CPU; for full run use colab_pipeline.ipynb.
"""

from pathlib import Path
import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import mne
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_decoder_model import EEGTransformer, get_spatial_weights_for_topomap

# -----------------------------------------------------------------------------
# Paths and constants (overridden by --fast / --quick in main)
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS = SCRIPT_DIR / "outputs"
PREPROCESSED = OUTPUTS / "preprocessed"
MODELS_DIR = OUTPUTS / "models"
FIGURES = OUTPUTS / "figures"
LOGS = OUTPUTS / "logs"

SUBJECT_IDS = [f"{i:02d}" for i in range(1, 23)]  # 01..22
SESSIONS = ("01", "02")
TASKS = ("AVI", "FVI", "OVI")
TASK_TO_SUPER = {"AVI": "LIVING", "FVI": "GEOMETRIC", "OVI": "OBJECT"}
TASK_VALUE_TO_LABEL10 = {
    ("AVI", 1): 0, ("AVI", 2): 1, ("AVI", 3): 2,
    ("FVI", 1): 5, ("FVI", 2): 4, ("FVI", 3): 3,
    ("OVI", 1): 6, ("OVI", 2): 7, ("OVI", 3): 8, ("OVI", 4): 9,
}
CLASS_NAMES_10 = ["dog", "bird", "fish", "circle", "square", "pentagram", "scissor", "watch", "cup", "chair"]
SUBJECTS_SESSION_1_ONLY = ("09", "10")
SKIP_RUNS = {("08", "02", "AVI")}
FLAG_SUBJECT = "19"

CROP_TMIN, CROP_TMAX = 0.0, 3.0
# Defaults (full model); overridden when --fast
TARGET_SFREQ = 250
EXPECTED_N_TIMES = 750
N_CHANNELS = 32
MAX_EPOCHS = 50
BATCH_SIZE = 32
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 10
VAL_FRAC_SUBJECTS = 0.1
N_CLASSES_10 = 10
N_CLASSES_3 = 3
CHANCE_10 = 10.0
CHANCE_3 = 100.0 / 3.0

# Progress log: write to file and console (set in main)
_progress_file = None


def progress_log(msg: str) -> None:
    """Print to console and append to outputs/logs/phase4_progress.txt (real time)."""
    print(msg, flush=True)
    if _progress_file is not None:
        _progress_file.write(msg + "\n")
        _progress_file.flush()


def load_all_subjects(skipped_log: list) -> dict[str, mne.Epochs]:
    """
    Iterate all 22 subjects, both sessions, all 3 tasks.
    Skip missing files (log to skipped_log).
    Per subject: concatenate ses-01 + ses-02 epochs across tasks.
    Add metadata: supercategory, label_10, label_3.
    Returns dict {subject_id: epochs_object}.
    """
    out = {}
    for sub_id in SUBJECT_IDS:
        list_ep = []
        for ses in SESSIONS:
            if sub_id in SUBJECTS_SESSION_1_ONLY and ses == "02":
                continue
            for task in TASKS:
                if (sub_id, ses, task) in SKIP_RUNS:
                    skipped_log.append(f"sub-{sub_id} ses-{ses} {task} (known missing)")
                    continue
                fname = f"sub-{sub_id}_ses-{ses}_{task}_clean-epo.fif"
                fpath = PREPROCESSED / fname
                if not fpath.exists():
                    skipped_log.append(f"sub-{sub_id} ses-{ses} {task}: file not found")
                    continue
                ep = mne.read_epochs(fpath, verbose=False)
                n = len(ep)
                if n == 0:
                    skipped_log.append(f"sub-{sub_id} ses-{ses} {task}: no epochs")
                    continue
                # Map event value to label_10 and label_3
                events = ep.events
                label_10 = np.array([TASK_VALUE_TO_LABEL10.get((task, int(ev[2])), -1) for ev in events])
                if np.any(label_10 < 0):
                    skipped_log.append(f"sub-{sub_id} ses-{ses} {task}: unknown event value")
                    continue
                label_3 = (label_10 // 3).clip(0, 2)  # 0,1,2->0; 3,4,5->1; 6,7,8,9->2
                supercat = TASK_TO_SUPER[task]
                ep.metadata = pd.DataFrame({
                    "supercategory": [supercat] * n,
                    "label_10": label_10,
                    "label_3": label_3,
                    "task": [task] * n,
                }, index=range(n))
                list_ep.append(ep)
        if not list_ep:
            skipped_log.append(f"sub-{sub_id}: no data at all")
            continue
        out[sub_id] = mne.concatenate_epochs(list_ep, on_mismatch="ignore", verbose=False)
    return out


def epochs_to_arrays(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop 0-3s, downsample to TARGET_SFREQ, return (X, y10, y3).
    X shape: (n_epochs, 32, EXPECTED_N_TIMES). Uses module-level TARGET_SFREQ, EXPECTED_N_TIMES.
    """
    ep = epochs.copy()
    ep.crop(tmin=CROP_TMIN, tmax=CROP_TMAX)
    ep.resample(TARGET_SFREQ, verbose=False)
    X = ep.get_data(picks="eeg")
    n_times = X.shape[2]
    if n_times != EXPECTED_N_TIMES:
        from scipy.signal import resample
        X = resample(X, EXPECTED_N_TIMES, axis=2)
    y10 = epochs.metadata["label_10"].values.astype(np.int64)
    y3 = epochs.metadata["label_3"].values.astype(np.int64)
    return X, y10, y3


def zscore_fit_transform(X_train: np.ndarray, X_apply: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit mean/std per channel over trials and time (shape 1,C,1), apply to X_train and X_apply."""
    mean = X_train.mean(axis=(0, 2), keepdims=True)   # (1, 32, 1)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std[std < 1e-8] = 1.0
    X_train_n = (X_train - mean) / std
    X_apply_n = (X_apply - mean) / std
    return X_train_n, X_apply_n


def get_subject_arrays(
    subject_epochs: dict[str, mne.Epochs],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per subject: (X, y10, y3) after crop and downsample (no z-score yet)."""
    out = {}
    for sub_id, ep in subject_epochs.items():
        X, y10, y3 = epochs_to_arrays(ep)
        out[sub_id] = (X, y10, y3)
    return out


def main():
    global TARGET_SFREQ, EXPECTED_N_TIMES, MAX_EPOCHS, BATCH_SIZE, TRANSFORMER_LAYERS, TRANSFORMER_HEADS, _progress_file

    parser = argparse.ArgumentParser(description="Phase 4: LOSO Transformer decoder")
    parser.add_argument("--quick", action="store_true", help="Only 3 folds, 10 epochs max (pipeline check)")
    parser.add_argument("--fast", action="store_true", help="Smaller model for CPU: 128Hz/375pts, 1 layer, 2 heads, batch 16")
    parser.add_argument("--ea", action="store_true", help="Apply Euclidean Alignment per subject before LOSO")
    args = parser.parse_args()

    if args.fast:
        TARGET_SFREQ = 128
        EXPECTED_N_TIMES = 384  # 3s * 128 Hz
        BATCH_SIZE = 16
        TRANSFORMER_LAYERS = 1
        TRANSFORMER_HEADS = 2
    else:
        TARGET_SFREQ = 250
        EXPECTED_N_TIMES = 750
        BATCH_SIZE = 32
        TRANSFORMER_LAYERS = 2
        TRANSFORMER_HEADS = 4

    if args.quick:
        MAX_EPOCHS = 10

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    _progress_file = open(LOGS / "phase4_progress.txt", "w", encoding="utf-8")
    try:
        progress_log(f"phase4_progress.txt opened (real-time log). --quick={args.quick} --fast={args.fast}")
        progress_log(f"Config: TARGET_SFREQ={TARGET_SFREQ} EXPECTED_N_TIMES={EXPECTED_N_TIMES} BATCH_SIZE={BATCH_SIZE} MAX_EPOCHS={MAX_EPOCHS} transformer_layers={TRANSFORMER_LAYERS} heads={TRANSFORMER_HEADS} EA={getattr(args, 'ea', False)}")

        skipped = []
        progress_log("--- STEP 1: Data Loading ---")
        subject_epochs = load_all_subjects(skipped)
        if skipped:
            (LOGS / "phase4_skipped.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")
            progress_log(f"Logged {len(skipped)} skipped runs to outputs/logs/phase4_skipped.txt")
        progress_log(f"Loaded {len(subject_epochs)} subjects")

        progress_log(f"--- STEP 2: Feature Extraction (crop 0-3s, downsample {TARGET_SFREQ}Hz) ---")
        subject_arrays = get_subject_arrays(subject_epochs)
        if getattr(args, "ea", False):
            from align_subject import euclidean_align
            for sub_id in list(subject_arrays.keys()):
                X, y10, y3 = subject_arrays[sub_id]
                subject_arrays[sub_id] = (euclidean_align(X), y10, y3)
            progress_log("Euclidean Alignment applied per subject (EA on).")
        # Get channel info from first subject for topomap later
        first_epochs = next(iter(subject_epochs.values()))
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            first_epochs.set_montage(montage, on_missing="ignore", verbose=False)
        except Exception:
            pass
        ch_names = first_epochs.ch_names[:32] if len(first_epochs.ch_names) >= 32 else first_epochs.ch_names
        info_for_topomap = first_epochs.info

        subject_ids = sorted(subject_arrays.keys())
        if args.quick:
            subject_ids = subject_ids[:3]
            progress_log(f"--quick: using only {len(subject_ids)} folds (subjects {subject_ids})")
        n_folds = len(subject_ids)

        all_y10_true, all_y10_pred = [], []
        all_y3_true, all_y3_pred = [], []
        fold_results = []
        best_fold_sub_id = None
        best_fold_acc_3 = -1.0
        best_fold_state = None
        best_fold_model = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        progress_log(f"Using device: {device} (AMD GPU needs ROCm/DirectML; PyTorch default is CUDA so CPU is used on Windows+AMD)")

        for fold_idx, test_sub_id in enumerate(subject_ids):
            progress_log(f"  ---- Fold {fold_idx + 1}/{n_folds}: test subject = sub-{test_sub_id} ----")
            train_sub_ids = [s for s in subject_ids if s != test_sub_id]
            n_val = max(1, int(len(train_sub_ids) * VAL_FRAC_SUBJECTS))
            val_sub_ids = train_sub_ids[-n_val:]
            train_sub_ids = train_sub_ids[:-n_val]

            # Stack train and val data
            X_train_list, y10_train_list, y3_train_list = [], [], []
            for s in train_sub_ids:
                X, y10, y3 = subject_arrays[s]
                X_train_list.append(X)
                y10_train_list.append(y10)
                y3_train_list.append(y3)
            X_train = np.concatenate(X_train_list, axis=0)
            y10_train = np.concatenate(y10_train_list, axis=0)
            y3_train = np.concatenate(y3_train_list, axis=0)

            X_val_list, y10_val_list, y3_val_list = [], [], []
            for s in val_sub_ids:
                X, y10, y3 = subject_arrays[s]
                X_val_list.append(X)
                y10_val_list.append(y10)
                y3_val_list.append(y3)
            X_val = np.concatenate(X_val_list, axis=0)
            y10_val = np.concatenate(y10_val_list, axis=0)
            y3_val = np.concatenate(y3_val_list, axis=0)

            X_test, y10_test, y3_test = subject_arrays[test_sub_id]

            # Z-score: fit on train only, apply to train, val, test
            X_train, _ = zscore_fit_transform(X_train, X_train)
            _, X_val = zscore_fit_transform(X_train, X_val)
            _, X_test_n = zscore_fit_transform(X_train, X_test)
            X_test = X_test_n

            # To Tensor
            X_train_t = torch.from_numpy(X_train).float().unsqueeze(1)
            y_train_t = torch.from_numpy(y10_train).long()
            X_val_t = torch.from_numpy(X_val).float().unsqueeze(1)
            y_val_t = torch.from_numpy(y10_val).long()
            X_test_t = torch.from_numpy(X_test).float().unsqueeze(1)
            y10_test_t = torch.from_numpy(y10_test).long()
            y3_test_t = torch.from_numpy(y3_test).long()

            # WeightedRandomSampler for class balance
            classes, counts = torch.unique(y_train_t, return_counts=True)
            class_weight = 1.0 / counts.float()
            sample_weight = class_weight[y_train_t]
            sampler = WeightedRandomSampler(sample_weight, len(sample_weight))

            train_ds = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
            val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

            model = EEGTransformer(
                n_classes=N_CLASSES_10,
                n_channels=N_CHANNELS,
                n_times=EXPECTED_N_TIMES,
                num_layers=TRANSFORMER_LAYERS,
                nhead=TRANSFORMER_HEADS,
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
            criterion = nn.CrossEntropyLoss()

            best_val_loss = float("inf")
            best_epoch = 0
            patience_counter = 0

            for epoch in range(MAX_EPOCHS):
                model.train()
                train_loss = 0.0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    opt.step()
                    train_loss += loss.item() * xb.size(0)
                train_loss /= len(X_train_t)
                sched.step()

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        val_loss += criterion(logits, yb).item() * xb.size(0)
                val_loss /= len(X_val_t)

                if fold_idx == 0:
                    progress_log(f"    Epoch {epoch + 1}/{MAX_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
                else:
                    progress_log(f"    Epoch {epoch + 1}/{MAX_EPOCHS} (fold sub-{test_sub_id})")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    progress_log(f"    Early stop at epoch {epoch + 1} (patience {EARLY_STOP_PATIENCE})")
                    break

            progress_log(f"    Best val loss at epoch {best_epoch + 1}")
            model.load_state_dict(best_state)
            model.to(device)
            model.eval()
            with torch.no_grad():
                logits = model(X_test_t.to(device))
                pred_10 = logits.argmax(dim=1).cpu().numpy()
                pred_3 = (pred_10 // 3).clip(0, 2)

            acc_10 = (pred_10 == y10_test).mean() * 100
            acc_3 = (pred_3 == y3_test).mean() * 100
            fold_results.append((test_sub_id, acc_10, acc_3))
            all_y10_true.append(y10_test)
            all_y10_pred.append(pred_10)
            all_y3_true.append(y3_test)
            all_y3_pred.append(pred_3)

            if acc_3 > best_fold_acc_3:
                best_fold_acc_3 = acc_3
                best_fold_sub_id = test_sub_id
                best_fold_state = best_state
                best_fold_model = EEGTransformer(
                    n_classes=N_CLASSES_10,
                    n_channels=N_CHANNELS,
                    n_times=EXPECTED_N_TIMES,
                    num_layers=TRANSFORMER_LAYERS,
                    nhead=TRANSFORMER_HEADS,
                )
                best_fold_model.load_state_dict(best_state)

            torch.save(best_state, MODELS_DIR / f"fold_sub{test_sub_id}_best.pt")
            progress_log(f"  Fold sub-{test_sub_id} (test): 10-class acc={acc_10:.1f}%  3-class acc={acc_3:.1f}%")

        # STEP 5: Results and outputs
        all_y10_true = np.concatenate(all_y10_true)
        all_y10_pred = np.concatenate(all_y10_pred)
        all_y3_true = np.concatenate(all_y3_true)
        all_y3_pred = np.concatenate(all_y3_pred)

        # Confusion matrices
        cm10 = confusion_matrix(all_y10_true, all_y10_pred, labels=list(range(10)))
        cm3 = confusion_matrix(all_y3_true, all_y3_pred, labels=[0, 1, 2])
        super_names = ["LIVING", "GEOMETRIC", "OBJECT"]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm10, cmap="Blues")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CLASS_NAMES_10, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES_10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.colorbar(im, ax=ax, label="Count")
        for i in range(10):
            for j in range(10):
                ax.text(j, i, int(cm10[i, j]), ha="center", va="center", fontsize=8)
        plt.title("LOSO Combined Confusion Matrix (10-class)")
        plt.tight_layout()
        plt.savefig(FIGURES / "confusion_matrix_10class.png", dpi=150, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm3, cmap="Blues")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(super_names)
        ax.set_yticklabels(super_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.colorbar(im, ax=ax, label="Count")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, int(cm3[i, j]), ha="center", va="center", fontsize=10)
        plt.title("LOSO Combined Confusion Matrix (3-class)")
        plt.tight_layout()
        plt.savefig(FIGURES / "confusion_matrix_3class.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Per-subject accuracy bar chart
        subs = [f"sub-{s}" for s in subject_ids]
        accs_10 = [r[1] for r in fold_results]
        accs_3 = [r[2] for r in fold_results]
        colors = ["C1" if s == f"sub-{FLAG_SUBJECT}" else "C0" for s in subs]
        fig, ax = plt.subplots(figsize=(12, 4))
        x = np.arange(len(subs))
        ax.bar(x - 0.2, accs_10, 0.4, label="10-class", color=colors, alpha=0.8)
        ax.bar(x + 0.2, accs_3, 0.4, label="3-class", color=colors, alpha=0.5)
        ax.axhline(CHANCE_10, color="gray", linestyle="--", label=f"Chance 10-class ({CHANCE_10}%)")
        ax.axhline(CHANCE_3, color="gray", linestyle=":", label=f"Chance 3-class ({CHANCE_3:.1f}%)")
        ax.set_xticks(x)
        ax.set_xticklabels(subs, rotation=45, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
        ax.set_title("Per-subject LOSO accuracy (sub-19 in orange)")
        plt.tight_layout()
        plt.savefig(FIGURES / "per_subject_accuracy.png", dpi=150, bbox_inches="tight")
        plt.close()

        # loso_results.txt
        rows = []
        for sub_id, a10, a3 in fold_results:
            rows.append({"Subject": f"sub-{sub_id}", "10-class Acc": f"{a10:.2f}", "3-class Acc": f"{a3:.2f}", "Chance(10)": f"{CHANCE_10}", "Chance(3)": f"{CHANCE_3:.1f}"})
        df = pd.DataFrame(rows)
        df.to_csv(LOGS / "loso_results.txt", sep="\t", index=False)
        progress_log(f"Saved {LOGS / 'loso_results.txt'}")

        # Per-class accuracy (10-class) for best/worst category
        recall_10 = np.diag(cm10) / (cm10.sum(axis=1) + 1e-9)
        best_class_idx = int(np.argmax(recall_10))
        worst_class_idx = int(np.argmin(recall_10))
        best_class_name = CLASS_NAMES_10[best_class_idx]
        worst_class_name = CLASS_NAMES_10[worst_class_idx]
        best_class_acc = recall_10[best_class_idx] * 100
        worst_class_acc = recall_10[worst_class_idx] * 100

        best_sub_idx = np.argmax([r[2] for r in fold_results])
        worst_sub_idx = np.argmin([r[2] for r in fold_results])
        best_sub_id = fold_results[best_sub_idx][0]
        worst_sub_id = fold_results[worst_sub_idx][0]
        best_sub_acc = fold_results[best_sub_idx][2]
        worst_sub_acc = fold_results[worst_sub_idx][2]

        mean_10 = np.mean([r[1] for r in fold_results])
        std_10 = np.std([r[1] for r in fold_results])
        mean_3 = np.mean([r[2] for r in fold_results])
        std_3 = np.std([r[2] for r in fold_results])
        beats_10 = mean_10 - CHANCE_10
        beats_3 = mean_3 - CHANCE_3

        if mean_3 > 50 and mean_10 > 20:
            verdict = "STRONG"
        elif mean_3 > 40 or mean_10 > 15:
            verdict = "MODERATE"
        elif mean_3 > 33:
            verdict = "WEAK"
        else:
            verdict = "NULL"

        summary = f"""
=== PHASE 4 RESULTS ===
Mean 10-class LOSO accuracy: {mean_10:.2f}% +/- {std_10:.2f}% (chance: 10%)
Mean 3-class LOSO accuracy: {mean_3:.2f}% +/- {std_3:.2f}% (chance: 33%)
Best decoded category: {best_class_name} at {best_class_acc:.1f}%
Worst decoded category: {worst_class_name} at {worst_class_acc:.1f}%
Best subject (easiest to decode): sub-{best_sub_id} at {best_sub_acc:.1f}%
Worst subject (hardest to decode): sub-{worst_sub_id} at {worst_sub_acc:.1f}%
Beats chance by: {beats_10:.1f} percentage points (10-class)
Beats chance by: {beats_3:.1f} percentage points (3-class)
DISCOVERY VERDICT: {verdict}
"""
        progress_log(summary)
        (LOGS / "phase4_summary.txt").write_text(summary.strip() + "\n", encoding="utf-8")

        # STEP 6: Attention topomap for best fold (spatial conv weights -> which input channels matter)
        if best_fold_model is not None and best_fold_sub_id is not None and best_fold_state is not None:
            best_fold_model.load_state_dict(best_fold_state)
            weights = get_spatial_weights_for_topomap(best_fold_model)
            if weights is not None and len(weights) == len(ch_names):
                from mne import create_info
                info_32 = create_info(ch_names[:32], 250, "eeg")
                try:
                    montage = mne.channels.make_standard_montage("standard_1020")
                    info_32.set_montage(montage, on_missing="ignore", verbose=False)
                except Exception:
                    pass
                fig, ax = plt.subplots(figsize=(5, 4))
                mne.viz.plot_topomap(weights, info_32, axes=ax, show=False)
                ax.set_title(f"Channel importance (best fold: sub-{best_fold_sub_id})")
                plt.tight_layout()
                plt.savefig(FIGURES / "attention_topomap.png", dpi=150, bbox_inches="tight")
                plt.close()
                progress_log(f"Saved {FIGURES / 'attention_topomap.png'}")
    finally:
        if _progress_file is not None:
            _progress_file.close()
            _progress_file = None


if __name__ == "__main__":
    main()
