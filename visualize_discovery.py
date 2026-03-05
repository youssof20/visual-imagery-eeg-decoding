"""
Discovery visualizations: does the brain respond differently to different imagery types?

Creates brain maps and graphs to show whether EEG differs by category (Living vs
Geometric vs Object). Uses one subject (sub-01) for illustration.
  - ERP comparison: average brain response over time; tests Living vs Geometric on Oz
  - Topomaps: scalp maps at 100, 200, 500, 1000 ms
  - Time–frequency (alpha 8–12 Hz, beta 13–30 Hz) for Living vs Geometric
  - Channel importance: which scalp regions carry the signal

Run: python visualize_discovery.py
Requires: outputs/preprocessed/ from preprocess_all.py.
Outputs: outputs/figures/*.png, outputs/logs/phase3_findings.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_morlet
from scipy import stats

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS = SCRIPT_DIR / "outputs"
PREPROCESSED = OUTPUTS / "preprocessed"
FIGURES = OUTPUTS / "figures"
LOGS = OUTPUTS / "logs"

# Sub-01 only for this phase
SUBJECT_ID = "01"
SESSIONS = ("01", "02")
TASKS = ("AVI", "FVI", "OVI")
TASK_TO_SUPER = {"AVI": "LIVING", "FVI": "GEOMETRIC", "OVI": "OBJECT"}
SUPER_COLORS = {"LIVING": "green", "GEOMETRIC": "blue", "OBJECT": "orange"}
CHANNELS_ERP = ["O1", "Oz", "O2", "T7", "T8"]
CHANNELS_OCCIPITAL = ["O1", "Oz", "O2"]
TOPOMAP_TIMES_MS = [100, 200, 500, 1000]
TFR_FREQS = np.logspace(np.log10(4), np.log10(40), 20)
ALPHA_BAND = (8, 12)
BETA_BAND = (13, 30)


def load_sub01_epochs():
    """Load all available .fif for sub-01 (ses-01 and ses-02, all 3 tasks); add supercategory; concatenate."""
    list_epochs = []
    for ses in SESSIONS:
        for task in TASKS:
            fname = f"sub-{SUBJECT_ID}_ses-{ses}_{task}_clean-epo.fif"
            fpath = PREPROCESSED / fname
            if not fpath.exists():
                print(f"  Skip (missing): {fpath.name}")
                continue
            ep = mne.read_epochs(fpath, verbose=False)
            n = len(ep)
            ep.metadata = pd.DataFrame({
                "supercategory": [TASK_TO_SUPER[task]] * n,
                "task": [task] * n,
                "session": [f"ses-{ses}"] * n,
            }, index=range(n))
            list_epochs.append(ep)
    if not list_epochs:
        raise FileNotFoundError(f"No preprocessed epochs found for sub-{SUBJECT_ID} under {PREPROCESSED}")
    all_epochs = mne.concatenate_epochs(list_epochs, on_mismatch="ignore", verbose=False)
    # Set standard 10-20 montage for topomap plotting (preprocessed FIF may lack digitization)
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        all_epochs.set_montage(montage, on_missing="ignore", verbose=False)
    except Exception:
        pass
    return all_epochs


def step1_load_and_label():
    """Load sub-01, add supercategory, print trial counts."""
    print("--- STEP 1: Load and Label ---")
    epochs = load_sub01_epochs()
    counts = epochs.metadata["supercategory"].value_counts().sort_index()
    print("Trial counts per supercategory:")
    for sc, n in counts.items():
        print(f"  {sc}: {n}")
    return epochs


def step2_erp_comparison(epochs, out_path):
    """Grand average per supercategory; plot O1, Oz, O2, T7, T8; on Oz pointwise t-test LIVING vs GEOMETRIC."""
    fig, axes = plt.subplots(len(CHANNELS_ERP), 1, figsize=(10, 10), sharex=True)
    evokeds = {}
    for sc in ("LIVING", "GEOMETRIC", "OBJECT"):
        idx = np.where(epochs.metadata["supercategory"].values == sc)[0]
        evokeds[sc] = epochs[idx].average()
    times = evokeds["LIVING"].times * 1000  # ms
    picks = [evokeds["LIVING"].ch_names.index(ch) for ch in CHANNELS_ERP]
    oz_idx = evokeds["LIVING"].ch_names.index("Oz")

    for ax, ch_name in zip(axes, CHANNELS_ERP):
        for sc in ("LIVING", "GEOMETRIC", "OBJECT"):
            ev = evokeds[sc]
            ch_idx = ev.ch_names.index(ch_name)
            ax.plot(times, ev.data[ch_idx] * 1e6, color=SUPER_COLORS[sc], label=sc, linewidth=1.5)
        ax.set_ylabel(f"{ch_name}\n(µV)")
        ax.legend(loc="upper right", fontsize=8)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.grid(True, alpha=0.3)

    # On Oz: pointwise t-test LIVING vs GEOMETRIC, shade p < 0.05
    idx_liv = np.where(epochs.metadata["supercategory"].values == "LIVING")[0]
    idx_geo = np.where(epochs.metadata["supercategory"].values == "GEOMETRIC")[0]
    liv = epochs[idx_liv].get_data(picks=[oz_idx])  # (n, 1, n_times)
    geo = epochs[idx_geo].get_data(picks=[oz_idx])
    liv_flat = liv[:, 0, :]  # (n_liv, n_times)
    geo_flat = geo[:, 0, :]
    p_vals = np.array([stats.ttest_ind(liv_flat[:, t], geo_flat[:, t])[1] for t in range(liv_flat.shape[1])])
    sig = p_vals < 0.05
    oz_ax = axes[CHANNELS_ERP.index("Oz")]
    for i in range(len(times) - 1):
        if sig[i] or (i > 0 and sig[i - 1]):
            oz_ax.axvspan(times[i], times[i + 1], color="red", alpha=0.2)
    oz_ax.set_title("Oz (red shade: p<0.05 LIVING vs GEOMETRIC)")

    # First significant window (post-stimulus only: t >= 0)
    post_stim = times >= 0
    sig_post = sig & post_stim
    first_sig_idx = np.where(sig_post)[0]
    if len(first_sig_idx) > 0:
        start = first_sig_idx[0]
        end = start
        while end < len(sig) and sig[end]:
            end += 1
        first_window_ms = (float(times[start]), float(times[min(end, len(times) - 1)]))
        print(f"ERP: First significant LIVING vs GEOMETRIC window on Oz: {first_window_ms[0]:.0f}-{first_window_ms[1]:.0f} ms")
    else:
        first_window_ms = (None, None)
        print("ERP: No significant LIVING vs GEOMETRIC window on Oz (p<0.05) post-stimulus")

    # Peak difference magnitude on Oz
    mean_liv = liv_flat.mean(axis=0) * 1e6
    mean_geo = geo_flat.mean(axis=0) * 1e6
    diff_oz = np.abs(mean_liv - mean_geo)
    peak_idx = np.argmax(diff_oz)
    peak_uv = diff_oz[peak_idx]
    peak_ms = times[peak_idx]
    print(f"ERP: Peak difference magnitude: {peak_uv:.2f} µV at Oz {peak_ms:.0f} ms")

    plt.xlabel("Time (ms)")
    plt.suptitle("ERP comparison: LIVING (green) vs GEOMETRIC (blue) vs OBJECT (orange)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return evokeds, first_window_ms, peak_uv, peak_ms


def _ensure_montage(inst):
    """Set standard 10-20 montage if no channel positions (for topomap plotting)."""
    if inst.info.get("dig") is None or len(inst.info.get("dig", [])) == 0:
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            inst.set_montage(montage, on_missing="ignore", verbose=False)
        except Exception:
            pass


def step3_topomaps(evokeds, out_path):
    """3 rows (LIVING, GEOMETRIC, OBJECT) x 4 cols (100, 200, 500, 1000 ms); same vmin/vmax."""
    times_s = np.array(TOPOMAP_TIMES_MS) / 1000.0
    ev = evokeds["LIVING"].copy()
    _ensure_montage(ev)
    all_data = []
    for sc in ("LIVING", "GEOMETRIC", "OBJECT"):
        for t in times_s:
            idx = np.argmin(np.abs(ev.times - t))
            all_data.append(evokeds[sc].data[:, idx])
    all_data = np.array(all_data)
    vmin, vmax = all_data.min(), all_data.max()
    vmax = max(abs(vmin), abs(vmax))
    vmin = -vmax

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for row, sc in enumerate(("LIVING", "GEOMETRIC", "OBJECT")):
        ev_sc = evokeds[sc].copy()
        _ensure_montage(ev_sc)
        for col, t_ms in enumerate(TOPOMAP_TIMES_MS):
            t_s = t_ms / 1000.0
            idx = np.argmin(np.abs(ev_sc.times - t_s))
            mne.viz.plot_topomap(
                ev_sc.data[:, idx],
                ev_sc.info,
                axes=axes[row, col],
                show=False,
                vlim=(vmin, vmax),
            )
            if row == 0:
                axes[row, col].set_title(f"{t_ms} ms")
            if col == 0:
                axes[row, col].set_ylabel(sc)
    plt.suptitle("Topomaps by supercategory (same scale)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    # Visible spatial difference: mean topomap value per supercategory (3 rows x 4 times -> 3 means)
    all_data = np.array(all_data)  # (12, n_ch)
    row_means = all_data.reshape(3, 4, -1).mean(axis=(1, 2))  # (3,)
    visible_diff = "YES" if np.any(np.abs(np.diff(row_means)) > 0.05 * (vmax - vmin)) else "NO"
    return visible_diff


def step4_tfr_alpha_beta(epochs, out_path):
    """TFR Morlet 4–40 Hz; LIVING and GEOMETRIC; average O1, Oz, O2; side-by-side + difference; mark alpha/beta."""
    picks_occ = [epochs.ch_names.index(ch) for ch in CHANNELS_OCCIPITAL]
    n_cycles = TFR_FREQS / 2.0
    idx_liv = np.where(epochs.metadata["supercategory"].values == "LIVING")[0]
    idx_geo = np.where(epochs.metadata["supercategory"].values == "GEOMETRIC")[0]
    ev_liv = epochs[idx_liv]
    ev_geo = epochs[idx_geo]
    out_liv = tfr_morlet(ev_liv, TFR_FREQS, n_cycles=n_cycles, picks=picks_occ, average=True, verbose=False)
    out_geo = tfr_morlet(ev_geo, TFR_FREQS, n_cycles=n_cycles, picks=picks_occ, average=True, verbose=False)
    tfr_liv = out_liv[0] if isinstance(out_liv, tuple) else out_liv
    tfr_geo = out_geo[0] if isinstance(out_geo, tuple) else out_geo
    # average=True -> (n_ch, n_freqs, n_times); mean over occipital channels
    power_liv = tfr_liv.data.mean(axis=0)  # (n_freqs, n_times)
    power_geo = tfr_geo.data.mean(axis=0)
    diff = power_liv - power_geo
    times = tfr_liv.times * 1000
    freqs = tfr_liv.freqs

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    extent = [times[0], times[-1], freqs[0], freqs[-1]]
    for ax, data, title in zip(
        axes,
        [power_liv, power_geo, diff],
        ["LIVING", "GEOMETRIC", "LIVING − GEOMETRIC"],
    ):
        im = ax.imshow(data, aspect="auto", origin="lower", extent=extent, cmap="RdBu_r")
        ax.axhline(ALPHA_BAND[0], color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(ALPHA_BAND[1], color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(BETA_BAND[0], color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(BETA_BAND[1], color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Power (a.u.)")
    plt.suptitle("Time–frequency (occipital O1, Oz, O2); alpha 8–12 Hz, beta 13–30 Hz")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Alpha/beta difference: mean power in band LIVING vs GEOMETRIC
    alpha_idx = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    beta_idx = (freqs >= BETA_BAND[0]) & (freqs <= BETA_BAND[1])
    alpha_diff = diff[alpha_idx, :].mean()
    beta_diff = diff[beta_idx, :].mean()
    alpha_yes_no = "YES" if np.abs(alpha_diff) > 0.1 * (np.abs(diff).max()) else "NO"
    beta_yes_no = "YES" if np.abs(beta_diff) > 0.1 * (np.abs(diff).max()) else "NO"
    return alpha_yes_no, beta_yes_no


def step5_channel_importance(epochs, out_path):
    """Mean absolute amplitude 0–2 s per channel per supercategory; 3 topomaps, same scale."""
    tmin, tmax = 0.0, 2.0
    times = epochs.times
    idx_02 = (times >= tmin) & (times <= tmax)
    ev = epochs.average()
    _ensure_montage(ev)  # epochs already have montage from step1; ev inherits
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    all_vals = []
    for ax, sc in zip(axes, ("LIVING", "GEOMETRIC", "OBJECT")):
        idx = np.where(epochs.metadata["supercategory"].values == sc)[0]
        data = epochs[idx].get_data()  # (n_epochs, n_ch, n_times)
        mean_abs = np.abs(data[:, :, idx_02]).mean(axis=0).mean(axis=1)  # (n_ch,)
        all_vals.append(mean_abs)
        mne.viz.plot_topomap(mean_abs, ev.info, axes=ax, show=False)
        ax.set_title(sc)
    all_vals = np.array(all_vals)
    vmin, vmax = all_vals.min(), all_vals.max()
    for ax in axes:
        for coll in ax.collections:
            coll.set_clim(vmin, vmax)
    plt.suptitle("Channel importance (mean |amplitude| 0–2 s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    # Strongest discriminating channel: max variance across supercategories
    var_per_ch = np.array(all_vals).var(axis=0)
    strongest_ch = epochs.ch_names[np.argmax(var_per_ch)]
    return strongest_ch


def step6_findings_report(
    first_window_ms,
    peak_uv,
    peak_ms,
    topomap_visible,
    alpha_yes,
    beta_yes,
    strongest_ch,
    best_figure,
    verdict,
    out_path,
):
    """Write and print Phase 3 findings."""
    first_sig_str = (
        f"ERP: First significant LIVING vs GEOMETRIC difference at {first_window_ms[0]:.0f}ms on Oz"
        if first_window_ms[0] is not None
        else "ERP: First significant LIVING vs GEOMETRIC difference: N/A (no p<0.05)"
    )
    window_str = (
        f"Most discriminating timewindow: {first_window_ms[0]:.0f} to {first_window_ms[1]:.0f}ms"
        if first_window_ms[0] is not None
        else "Most discriminating timewindow: N/A"
    )
    lines = [
        "=== PHASE 3 FINDINGS: sub-01 ===",
        first_sig_str,
        f"ERP: Peak difference magnitude: {peak_uv:.2f}µV at Oz {peak_ms:.0f}ms",
        f"Alpha (8-12Hz): LIVING vs GEOMETRIC difference: {alpha_yes}",
        f"Beta (13-30Hz): LIVING vs GEOMETRIC difference: {beta_yes}",
        f"Strongest discriminating channel: {strongest_ch}",
        window_str,
        f"Topomap: Visible spatial difference between categories: {topomap_visible}",
        f"Best figure for professor pitch: {best_figure}",
        f"Verdict: {verdict}",
    ]
    text = "\n".join(lines)
    LOGS.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print("\n--- Phase 3 findings (saved and below) ---")
    print(text)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    epochs = step1_load_and_label()

    evokeds, first_window_ms, peak_uv, peak_ms = step2_erp_comparison(
        epochs, FIGURES / "erp_comparison.png"
    )
    topomap_visible = step3_topomaps(evokeds, FIGURES / "topomaps_supercategories.png")
    alpha_yes, beta_yes = step4_tfr_alpha_beta(epochs, FIGURES / "tfr_alpha_beta.png")
    strongest_ch = step5_channel_importance(epochs, FIGURES / "channel_importance.png")

    # Most discriminating timewindow: use first significant window
    if first_window_ms[0] is not None:
        best_window = f"{first_window_ms[0]:.0f} to {first_window_ms[1]:.0f}ms"
    else:
        best_window = "N/A"

    best_figure = "erp_comparison.png"  # default; can be overridden by inspection
    verdict = "PROCEED to Phase 4" if first_window_ms[0] is not None or topomap_visible == "YES" else "INVESTIGATE FURTHER"

    step6_findings_report(
        first_window_ms,
        peak_uv,
        peak_ms,
        topomap_visible,
        alpha_yes,
        beta_yes,
        strongest_ch,
        best_figure,
        verdict,
        LOGS / "phase3_findings.txt",
    )


if __name__ == "__main__":
    main()
