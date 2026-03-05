"""
VVIQ vs decoding accuracy: does vividness of mental imagery predict decodability?

Tests whether people who report more vivid mental imagery (VVIQ questionnaire
score) also have more decodable brain signals. Loads participants.tsv and
within_subject_results.txt, joins on subject ID, computes Pearson correlation,
median-split (High vs Low VVIQ), and a scatter plot.

Run: python vviq_correlation.py
Requires: dataset/participants.tsv, outputs/logs/within_subject_results.txt
Outputs: outputs/figures/vviq_scatter.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET = SCRIPT_DIR / "dataset"
OUTPUTS = SCRIPT_DIR / "outputs"
LOGS = OUTPUTS / "logs"
FIGURES = OUTPUTS / "figures"

PARTICIPANTS_TSV = DATASET / "participants.tsv"
WITHIN_RESULTS_TXT = LOGS / "within_subject_results.txt"
CHANCE_3 = 100.0 / 3.0  # 33.33%


def main():
    participants = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    # within_subject_results: Subject, 10-class Acc, 3-class Acc, Chance(10), Chance(3)
    results = pd.read_csv(WITHIN_RESULTS_TXT, sep="\t")
    results = results[results["Subject"].str.startswith("sub-")].copy()  # drop Mean row

    # Join on subject ID
    df = results.merge(
        participants,
        left_on="Subject",
        right_on="participant_id",
        how="inner",
    )

    df["VVIQ"] = df["VVIQ score"].astype(float)
    df["acc_3"] = df["3-class Acc"].astype(float)
    df["Group_notes"] = ""
    df.loc[df["Subject"] == "sub-08", "Group_notes"] = "outlier (3-class=4%)"
    df.loc[df["Subject"] == "sub-19", "Group_notes"] = "experienced"

    # Full table
    print("=" * 70)
    print("Full table: Subject | VVIQ | 3-class Acc | Group")
    print("=" * 70)
    for _, row in df.iterrows():
        notes = f" [{row['Group_notes']}]" if row["Group_notes"] else ""
        print(f"  {row['Subject']:<8}  VVIQ={row['VVIQ']:.0f}  3-class={row['acc_3']:.2f}%  {row['group']}{notes}")
    print()

    # Pearson correlation (exclude sub-08 outlier for correlation?)
    # Instructions say "flag sub-08 as outlier" - we'll compute correlation both ways: all subjects and excluding sub-08
    r_all, p_all = stats.pearsonr(df["VVIQ"], df["acc_3"])
    df_no08 = df[df["Subject"] != "sub-08"]
    r_no08, p_no08 = stats.pearsonr(df_no08["VVIQ"], df_no08["acc_3"])

    print("Pearson correlation (VVIQ vs 3-class accuracy):")
    print(f"  All subjects (n={len(df)}):  r = {r_all:.4f},  p = {p_all:.4f}")
    print(f"  Excluding sub-08 (n={len(df_no08)}):  r = {r_no08:.4f},  p = {p_no08:.4f}")
    r, p = r_no08, p_no08  # use excluding outlier for interpretation
    if p < 0.05:
        interp = "significant correlation"
    else:
        interp = "no significant correlation"
    print(f"  Interpretation: {interp}")
    print()

    # Median split on VVIQ
    med = df_no08["VVIQ"].median()
    high = df_no08[df_no08["VVIQ"] > med]
    low = df_no08[df_no08["VVIQ"] <= med]
    mean_high = high["acc_3"].mean()
    mean_low = low["acc_3"].mean()
    t_stat, t_p = stats.ttest_ind(high["acc_3"], low["acc_3"])

    print("Median split on VVIQ (excluding sub-08):")
    print(f"  Median VVIQ = {med:.0f}")
    print(f"  High VVIQ (above median): n={len(high)}, mean 3-class acc = {mean_high:.2f}%")
    print(f"  Low VVIQ (at or below median): n={len(low)}, mean 3-class acc = {mean_low:.2f}%")
    print(f"  Independent t-test: p = {t_p:.4f}")
    print()

    # Scatter plot
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["green" if a > CHANCE_3 else "red" for a in df["acc_3"]]
    ax.scatter(df["VVIQ"], df["acc_3"], c=colors, s=80, alpha=0.8, edgecolors="black", linewidths=0.5)

    # Regression line (excluding sub-08)
    x = df_no08["VVIQ"].values
    y = df_no08["acc_3"].values
    slope, intercept, r_val, p_val, se = stats.linregress(x, y)
    x_line = np.array([df["VVIQ"].min(), df["VVIQ"].max()])
    ax.plot(x_line, slope * x_line + intercept, "b-", linewidth=2, label=f"Regression (r={r_no08:.3f})")

    for _, row in df.iterrows():
        ax.annotate(
            row["Subject"].replace("sub-", ""),
            (row["VVIQ"], row["acc_3"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            ha="left",
        )

    ax.axhline(CHANCE_3, color="gray", linestyle="--", linewidth=1.5, label=f"Chance ({CHANCE_3:.1f}%)")
    ax.set_xlabel("VVIQ Score")
    ax.set_ylabel("3-class Accuracy (%)")
    ax.set_title("VVIQ Score vs Visual Imagery Decoding Accuracy")
    ax.legend()
    ax.set_xlim(df["VVIQ"].min() - 2, df["VVIQ"].max() + 2)
    ax.set_ylim(-5, 70)
    plt.tight_layout()
    plt.savefig(FIGURES / "vviq_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES / 'vviq_scatter.png'}")
    print()

    # Final statement
    direction = "higher" if r_no08 > 0 and p_no08 < 0.05 else "no difference in"
    print("Final statement:")
    print(
        f'  Subjects with higher VVIQ scores showed {direction} decoding accuracy '
        f'(r={r_no08:.3f}, p={p_no08:.4f}). '
        f'High-VVIQ group: {mean_high:.1f}% vs Low-VVIQ group: {mean_low:.1f}% (t-test p={t_p:.4f}).'
    )


if __name__ == "__main__":
    main()
