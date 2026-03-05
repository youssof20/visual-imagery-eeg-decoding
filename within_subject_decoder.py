"""
Within-subject decoding: train on session 1, test on session 2.

Trains an AI model on one session of a subject's data and tests it on their
second session. This shows the upper bound of what is decodable per person
(same person, different day). Euclidean Alignment and z-score are fit on
train (ses-01) only and applied to test (ses-02) to avoid leakage.

Run: python within_subject_decoder.py [--fast]
  --fast = smaller/faster model (128 Hz, 1 layer, 2 heads).
Output: outputs/logs/within_subject_results.txt
"""

from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import mne

from train_decoder import (
    PREPROCESSED,
    SUBJECT_IDS,
    SESSIONS,
    TASKS,
    TASK_TO_SUPER,
    TASK_VALUE_TO_LABEL10,
    SUBJECTS_SESSION_1_ONLY,
    SKIP_RUNS,
    CROP_TMIN,
    CROP_TMAX,
    N_CHANNELS,
    N_CLASSES_10,
    LR,
    WEIGHT_DECAY,
    VAL_FRAC_SUBJECTS,
    CLASS_NAMES_10,
)
from train_decoder_model import EEGTransformer
from align_subject import euclidean_align_fit_apply

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS = SCRIPT_DIR / "outputs"
LOGS = OUTPUTS / "logs"

# Full model defaults; overridden by --fast
TARGET_SFREQ = 250
EXPECTED_N_TIMES = 750
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4
MAX_EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 10


def load_subject_session_epochs(sub_id: str, ses_id: str) -> mne.Epochs | None:
    """Load one subject, one session, all three tasks; build metadata; concatenate. Returns None if any required file missing."""
    list_ep = []
    for task in TASKS:
        if (sub_id, ses_id, task) in SKIP_RUNS:
            continue
        fname = f"sub-{sub_id}_ses-{ses_id}_{task}_clean-epo.fif"
        fpath = PREPROCESSED / fname
        if not fpath.exists():
            return None
        ep = mne.read_epochs(fpath, verbose=False)
        n = len(ep)
        if n == 0:
            return None
        events = ep.events
        label_10 = np.array([TASK_VALUE_TO_LABEL10.get((task, int(ev[2])), -1) for ev in events])
        if np.any(label_10 < 0):
            return None
        label_3 = (label_10 // 3).clip(0, 2)
        supercat = TASK_TO_SUPER[task]
        ep.metadata = pd.DataFrame({
            "supercategory": [supercat] * n,
            "label_10": label_10,
            "label_3": label_3,
            "task": [task] * n,
        }, index=range(n))
        list_ep.append(ep)
    if not list_ep:
        return None
    return mne.concatenate_epochs(list_ep, on_mismatch="ignore", verbose=False)


def epochs_to_arrays(epochs: mne.Epochs, target_sfreq: int, expected_n_times: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ep = epochs.copy()
    ep.crop(tmin=CROP_TMIN, tmax=CROP_TMAX)
    ep.resample(target_sfreq, verbose=False)
    X = ep.get_data(picks="eeg")
    n_times = X.shape[2]
    if n_times != expected_n_times:
        from scipy.signal import resample
        X = resample(X, expected_n_times, axis=2)
    y10 = epochs.metadata["label_10"].values.astype(np.int64)
    y3 = epochs.metadata["label_3"].values.astype(np.int64)
    return X, y10, y3


def zscore_fit_transform(X_train: np.ndarray, X_apply: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std[std < 1e-8] = 1.0
    return (X_train - mean) / std, (X_apply - mean) / std


def main():
    global TARGET_SFREQ, EXPECTED_N_TIMES, TRANSFORMER_LAYERS, TRANSFORMER_HEADS, BATCH_SIZE

    parser = argparse.ArgumentParser(description="Within-subject: train ses-01, test ses-02")
    parser.add_argument("--fast", action="store_true", help="Reduced model: 128Hz, 384 pts, 1 layer, 2 heads, batch 16")
    args = parser.parse_args()

    if args.fast:
        TARGET_SFREQ = 128
        EXPECTED_N_TIMES = 384
        TRANSFORMER_LAYERS = 1
        TRANSFORMER_HEADS = 2
        BATCH_SIZE = 16

    LOGS.mkdir(parents=True, exist_ok=True)

    # Subjects that have both ses-01 and ses-02
    subject_ids = [s for s in SUBJECT_IDS if s not in SUBJECTS_SESSION_1_ONLY]
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Within-subject decoding: train ses-01, test ses-02. Device: {device}")
    print(f"Model: {TARGET_SFREQ}Hz, {EXPECTED_N_TIMES} pts, {TRANSFORMER_LAYERS} layers, {TRANSFORMER_HEADS} heads\n")

    for sub_id in subject_ids:
        train_ep = load_subject_session_epochs(sub_id, "01")
        test_ep = load_subject_session_epochs(sub_id, "02")
        if train_ep is None or test_ep is None:
            print(f"  sub-{sub_id}: skip (missing ses-01 or ses-02)")
            continue

        X_train, y10_train, y3_train = epochs_to_arrays(train_ep, TARGET_SFREQ, EXPECTED_N_TIMES)
        X_test, y10_test, y3_test = epochs_to_arrays(test_ep, TARGET_SFREQ, EXPECTED_N_TIMES)

        # EA: fit on train, apply to train and test
        X_train, X_test = euclidean_align_fit_apply(X_train, X_test)

        # Z-score: fit on train, apply to both
        X_train_n, X_test_n = zscore_fit_transform(X_train, X_test)

        # Validation split from train (10%) — fixed seed for reproducibility
        n_val = max(1, int(len(X_train_n) * VAL_FRAC_SUBJECTS))
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(X_train_n))
        val_idx = indices[:n_val]
        tr_idx = indices[n_val:]
        X_tr = X_train_n[tr_idx]
        y_tr = y10_train[tr_idx]
        X_val = X_train_n[val_idx]
        y_val = y10_train[val_idx]

        X_tr_t = torch.from_numpy(X_tr).float().unsqueeze(1)
        y_tr_t = torch.from_numpy(y_tr).long()
        X_val_t = torch.from_numpy(X_val).float().unsqueeze(1)
        y_val_t = torch.from_numpy(y_val).long()
        X_test_t = torch.from_numpy(X_test_n).float().unsqueeze(1)

        # Weight by inverse frequency (all 10 classes; missing classes get small weight)
        classes, counts = torch.unique(y_tr_t, return_counts=True)
        class_weight = torch.ones(N_CLASSES_10, dtype=torch.float) * 1e-6
        for c, cnt in zip(classes.tolist(), counts.tolist()):
            class_weight[c] = 1.0 / float(cnt)
        sample_weight = class_weight[y_tr_t]
        sampler = WeightedRandomSampler(sample_weight, len(sample_weight))

        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=BATCH_SIZE, sampler=sampler, num_workers=0,
        )
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
        best_state = None
        patience_counter = 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                opt.step()
            sched.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(model(xb), yb).item() * xb.size(0)
            val_loss /= len(X_val_t)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                break

        model.load_state_dict(best_state)
        model.to(device)
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t.to(device))
            pred_10 = logits.argmax(dim=1).cpu().numpy()
            pred_3 = (pred_10 // 3).clip(0, 2)

        acc_10 = (pred_10 == y10_test).mean() * 100
        acc_3 = (pred_3 == y3_test).mean() * 100
        chance_10 = 10.0
        chance_3 = 100.0 / 3.0
        results.append((sub_id, acc_10, acc_3, chance_10, chance_3))
        print(f"  sub-{sub_id}  10-class: {acc_10:.1f}%  3-class: {acc_3:.1f}%")

    if not results:
        print("No subjects with both sessions. Exiting.")
        return

    # Summary table
    print("\n" + "=" * 60)
    print("Within-subject results (train ses-01, test ses-02)")
    print("=" * 60)
    print(f"{'Subject':<12} {'10-class Acc':<14} {'3-class Acc':<14} {'Chance(10)':<12} {'Chance(3)':<12}")
    print("-" * 60)
    for sub_id, a10, a3, c10, c3 in results:
        print(f"sub-{sub_id:<8} {a10:>10.2f}%    {a3:>10.2f}%    {c10:>8.1f}%    {c3:>8.1f}%")

    mean_10 = np.mean([r[1] for r in results])
    mean_3 = np.mean([r[2] for r in results])
    print("-" * 60)
    print(f"Mean (n={len(results)})   {mean_10:.2f}%         {mean_3:.2f}%")
    print(f"\nMean 10-class accuracy: {mean_10:.2f}%")
    print(f"Mean 3-class accuracy:  {mean_3:.2f}%")

    out_path = LOGS / "within_subject_results.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Subject\t10-class Acc\t3-class Acc\tChance(10)\tChance(3)\n")
        for sub_id, a10, a3, c10, c3 in results:
            f.write(f"sub-{sub_id}\t{a10:.2f}\t{a3:.2f}\t{c10}\t{c3:.1f}\n")
        f.write(f"\nMean (n={len(results)})\t{mean_10:.2f}\t{mean_3:.2f}\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
