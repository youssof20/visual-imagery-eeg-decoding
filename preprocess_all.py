"""
Preprocessing pipeline for VI-BCI EEG data.

Takes raw brain signal files (.bdf) and cleans them for downstream analysis:
  - Removes eye-blink artifacts using ICA (Fp1/Fp2 as proxy for eye activity)
  - Filters out electrical noise (band-pass 1–40 Hz, notch at 50 Hz)
  - Cuts data into 4.5-second windows around each trial and baseline-corrects
  - Drops bad trials (peak-to-peak amplitude too high)

Run: python preprocess_all.py [full]
  'full' = all 22 subjects; otherwise processes sub-01 only for testing.
Output: outputs/preprocessed/sub-XX_ses-YY_TASK_clean-epo.fif per run.
"""

from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA

# -----------------------------------------------------------------------------
# Constants (from Phase 1 audit)
# -----------------------------------------------------------------------------
TASKS = ["AVI", "FVI", "OVI"]
SUBJECTS_SESSION_1_ONLY = ("09", "10")  # sub-09, sub-10 have no ses-02
REJECT_PEAK_TO_PEAK_UV = 150  # first pass
REJECT_PEAK_TO_PEAK_UV_RETRY = 200  # if >20% dropped for any category
MAX_DROP_FRACTION = 0.20  # flag if any category loses >20% after retry
ICA_N_COMPONENTS = 20
EOG_CHANNELS = ["Fp1", "Fp2"]
TMIN, TMAX = -0.5, 4.0
BASELINE = (-0.2, 0.0)


def get_paths(subject_id: str, session_id: str, task: str, dataset_root: Path) -> tuple[Path, Path]:
    """Return (bdf_path, tsv_path) for given sub/ses/task."""
    subjects_root = dataset_root / "subjects"
    eeg_dir = subjects_root / f"sub-{subject_id}" / f"sub-{subject_id}" / f"ses-{session_id}" / "eeg"
    stem = f"sub-{subject_id}_ses-{session_id}_task-{task}"
    return eeg_dir / f"{stem}_eeg.bdf", eeg_dir / f"{stem}_events.tsv"


def preprocess_subject(
    subject_id: str,
    session_id: str,
    task: str,
    dataset_root: Path,
    output_root: Path,
    log_missing: list,
    log_ica: list,
    log_quality: list,
    summary_rows: list,
) -> bool:
    """
    Run full preprocessing for one subject/session/task.
    Returns True if output was written, False if skipped (e.g. missing file).
    Appends to log_missing, log_ica, log_quality, summary_rows.
    """
    bdf_path, tsv_path = get_paths(subject_id, session_id, task, dataset_root)
    out_dir = output_root / "preprocessed"
    logs_dir = output_root / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not bdf_path.exists():
        log_missing.append(f"{datetime.now().isoformat()} BDF not found: {bdf_path}")
        return False
    if not tsv_path.exists():
        log_missing.append(f"{datetime.now().isoformat()} Events TSV not found: {tsv_path}")
        return False

    # ----- 1. Load BDF, pick only EEG (exclude Status) -----
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    raw.pick_types(eeg=True, exclude=[])  # drops Status

    # ----- 2. Re-reference to average -----
    raw.set_eeg_reference(ref_channels="average", projection=False)

    # ----- 3. Band-pass 1-40 Hz -----
    raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)

    # ----- 4. Notch 50 Hz -----
    raw.notch_filter(freqs=50.0, verbose=False)

    # ----- 5. ICA: 20 components, Fp1/Fp2 as EOG proxy -----
    # Use Fp1 (and Fp2 if accepted) as EOG proxy; no dedicated EOG channel in dataset
    ica = ICA(n_components=ICA_N_COMPONENTS, random_state=97, verbose=False)
    ica.fit(raw, verbose=False)

    # Find and exclude eye-blink components using Fp1 (and Fp2) as proxy
    try:
        eog_inds, _ = ica.find_bads_eog(raw, ch_name=EOG_CHANNELS)
    except Exception:
        eog_inds, _ = ica.find_bads_eog(raw, ch_name=EOG_CHANNELS[0])
    ica.exclude = eog_inds
    n_removed = len(ica.exclude)
    raw_clean = raw.copy()
    ica.apply(raw_clean)

    # Log ICA
    log_ica.append(f"sub-{subject_id} ses-{session_id} {task} | ICA components removed: {n_removed} (excluded indices: {ica.exclude})")

    # ----- 6. Load events from TSV (latency = sample index, value = trigger code) -----
    events_df = pd.read_csv(tsv_path, sep="\t")
    events = np.column_stack([
        events_df["latency"].values.astype(int),
        np.zeros(len(events_df), dtype=int),
        events_df["value"].values.astype(int),
    ])
    event_id = {str(v): v for v in sorted(events_df["value"].unique())}

    # ----- 7. Epoch: -0.5 to 4.0 s -----
    epochs = mne.Epochs(
        raw_clean,
        events,
        event_id=event_id,
        tmin=TMIN,
        tmax=TMAX,
        baseline=BASELINE,
        preload=True,
        verbose=False,
    )

    # ----- 8. Baseline already set in Epochs -----
    # (baseline=(-0.2, 0) applied above)

    # ----- 9. Auto-reject bad epochs -----
    reject_v = REJECT_PEAK_TO_PEAK_UV * 1e-6  # MNE uses Volts
    epochs.drop_bad(reject={"eeg": reject_v})

    def count_per_event_id(evs, eid_to_code):
        return {eid: (evs[:, 2] == eid_to_code[eid]).sum() for eid in eid_to_code}

    n_before = count_per_event_id(events, event_id)
    n_after = count_per_event_id(epochs.events, event_id)
    drop_frac = {eid: 1.0 - (n_after[eid] / n_before[eid]) if n_before[eid] else 0 for eid in event_id}
    any_over_20 = any(drop_frac[eid] > MAX_DROP_FRACTION for eid in event_id)

    if any_over_20:
        # Retry with 200 µV
        epochs = mne.Epochs(
            raw_clean,
            events,
            event_id=event_id,
            tmin=TMIN,
            tmax=TMAX,
            baseline=BASELINE,
            preload=True,
            verbose=False,
        )
        reject_v_retry = REJECT_PEAK_TO_PEAK_UV_RETRY * 1e-6
        epochs.drop_bad(reject={"eeg": reject_v_retry})
        n_after = count_per_event_id(epochs.events, event_id)
        drop_frac = {eid: 1.0 - (n_after[eid] / n_before[eid]) if n_before[eid] else 0 for eid in event_id}
        if any(drop_frac[eid] > MAX_DROP_FRACTION for eid in event_id):
            log_quality.append(
                f"sub-{subject_id} ses-{session_id} {task} | >20% epochs dropped for at least one category after 200uV threshold | drop_frac={drop_frac}"
            )
        reject_used = REJECT_PEAK_TO_PEAK_UV_RETRY
    else:
        reject_used = REJECT_PEAK_TO_PEAK_UV

    n_kept = len(epochs)
    n_dropped = len(events) - n_kept
    drop_pct = 100.0 * n_dropped / len(events) if len(events) else 0
    flag = ">20% drop" if any(drop_frac.get(eid, 0) > MAX_DROP_FRACTION for eid in event_id) else ""

    # Build summary row (one per run that produced output)
    summary_rows.append({
        "Subject": f"sub-{subject_id}",
        "Session": f"ses-{session_id}",
        "Task": task,
        "Kept": n_kept,
        "Dropped": n_dropped,
        "Drop%": f"{drop_pct:.1f}",
        "ICA_removed": n_removed,
        "Flag": flag,
    })

    # ----- 10. Save -----
    out_fname = f"sub-{subject_id}_ses-{session_id}_{task}_clean-epo.fif"
    out_path = out_dir / out_fname
    epochs.save(out_path, overwrite=True, verbose=False)
    return True


def get_all_runs(dataset_root: Path) -> list[tuple[str, str, str]]:
    """Return list of (subject_id, session_id, task). Skip ses-02 for sub-09, sub-10."""
    subjects_root = dataset_root / "subjects"
    runs = []
    for sub_dir in sorted(subjects_root.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        subject_id = sub_dir.name.replace("sub-", "")
        for inner in sub_dir.iterdir():
            if not inner.is_dir() or inner.name != sub_dir.name:
                continue
            for ses_dir in sorted(inner.iterdir()):
                if not ses_dir.is_dir() or not ses_dir.name.startswith("ses-"):
                    continue
                session_id = ses_dir.name.replace("ses-", "")
                if subject_id in SUBJECTS_SESSION_1_ONLY and session_id == "02":
                    continue
                for task in TASKS:
                    runs.append((subject_id, session_id, task))
            break
        else:
            continue
    return runs


def main():
    mne.set_log_level("WARNING")  # reduce filter/reject messages when running many subjects
    script_dir = Path(__file__).resolve().parent
    dataset_root = script_dir / "dataset"
    output_root = script_dir / "outputs"

    log_missing: list[str] = []
    log_ica: list[str] = []
    log_quality: list[str] = []
    summary_rows: list[dict] = []

    # Parse args: if "sub01_only" or no arg run sub-01 ses-01 only first; if "full" run all
    run_mode = "sub01_only" if len(sys.argv) <= 1 else sys.argv[1].strip().lower()

    if run_mode == "sub01_only" or run_mode not in ("full", "all"):
        runs = [("01", "01", t) for t in TASKS]
        print("Running sub-01 ses-01, all three tasks (confirm .fif exist before full batch).")
    else:
        runs = get_all_runs(dataset_root)
        print(f"Running full batch: {len(runs)} runs.")

    for subject_id, session_id, task in runs:
        print(f"  Processing sub-{subject_id} ses-{session_id} {task} ...")
        preprocess_subject(
            subject_id, session_id, task,
            dataset_root, output_root,
            log_missing, log_ica, log_quality, summary_rows,
        )

    # Write logs
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if log_missing:
        (logs_dir / "missing_files.txt").write_text("\n".join(log_missing) + "\n", encoding="utf-8")
        print(f"Logged {len(log_missing)} missing file(s) to outputs/logs/missing_files.txt")
    if log_ica:
        (logs_dir / "ica_log.txt").write_text("\n".join(log_ica) + "\n", encoding="utf-8")
        print(f"Logged ICA details to outputs/logs/ica_log.txt")
    if log_quality:
        (logs_dir / "quality_flags.txt").write_text("\n".join(log_quality) + "\n", encoding="utf-8")
        print(f"Logged {len(log_quality)} quality flag(s) to outputs/logs/quality_flags.txt")

    # Summary table
    df = pd.DataFrame(summary_rows)
    summary_path = logs_dir / "preprocessing_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False) + "\n")
    print("\nPreprocessing summary:")
    print(df.to_string(index=False))
    print(f"\nSaved to {summary_path}")

    if run_mode in ("sub01_only", "") or run_mode not in ("full", "all"):
        print("\nConfirm outputs/preprocessed/sub-01_ses-01_*_clean-epo.fif exist, then run: python preprocess_all.py full")


if __name__ == "__main__":
    main()
