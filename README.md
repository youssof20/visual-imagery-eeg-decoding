# VI-BCI: Can We Read What You're Imagining?

A machine learning pipeline that decodes **visual imagery categories from EEG brainwaves** — figuring out whether someone is imagining a living thing, a geometric shape, or an object, purely from their brain activity.

Built on the VI-BCI dataset (22 subjects, 32-channel EEG, 1000Hz).

---

## The Question

When you close your eyes and imagine a face vs. a triangle — does your brain actually produce different electrical signals? And if so, can a machine learn to tell them apart?

This project tries to answer that.

---

## What We Found

**1. The signal is real.**

There is a measurable difference in brainwave activity at the back of the head (occipital channels — the visual processing region) around 200–250ms after a stimulus, when subjects imagine Living things vs. Geometric shapes. This shows up consistently across subjects and matches what neuroscience already knows about visual processing timing.

**2. Decoding works — modestly.**

A transformer-based decoder trained on one session and tested on another achieved **37.28% mean accuracy** on 3-class classification (Living / Geometric / Object). Chance is 33.3%. **12 out of 19 subjects decoded above chance.** Small but real.

**3. Cross-subject generalization is hard.**

When we trained on all subjects and tested on a held-out person (leave-one-subject-out), accuracy dropped. Every person's brain is wired slightly differently — this is the fundamental problem in BCI research and exactly what needs to be solved before any of this reaches clinical use.

**4. Mental imagery ability doesn't predict decoding accuracy.**

We tested whether people who score higher on the VVIQ (a questionnaire measuring how vivid your mental imagery is) decode better. They don't — correlation r = 0.07, essentially zero. Vivid imagers don't produce more decodable EEG, at least not with this approach.

---

## Pipeline

```
preprocess_all.py       → Filter, ICA artifact removal, epoch extraction
visualize_discovery.py  → ERPs, topomaps, time-frequency analysis
within_subject_decoder.py → Train session 1, test session 2 (per subject)
train_decoder.py        → Cross-subject LOSO transformer decoder
vviq_correlation.py     → VVIQ score vs. decoding accuracy
```

## Dataset

VI-BCI dataset — 22 participants, 32 channels, 1000Hz, 10 visual imagery categories.
Download: [Figshare DOI 10.6084/m9.figshare.30227503](https://doi.org/10.6084/m9.figshare.30227503)

## Requirements

```
pip install mne torch torchvision scikit-learn matplotlib scipy tqdm pandas
```

---

## Key Results Summary

| Metric | Result |
|---|---|
| Within-subject accuracy (3-class) | 37.28% mean (chance: 33.3%) |
| Subjects above chance | 12 / 19 |
| Cross-subject LOSO accuracy | Lower — domain shift identified |
| VVIQ correlation | r = 0.07 (no relationship) |
| Key EEG finding | Living vs. Geometric difference at Oz, 208–257ms |

---

### Figure previews

| ERP comparison | Topomaps | VVIQ vs accuracy |
|----------------|----------|-------------------|
| ![ERP comparison](outputs/figures/erp_comparison.png) | ![Topomaps](outputs/figures/topomaps_supercategories.png) | ![VVIQ scatter](outputs/figures/vviq_scatter.png) |


---

## Status

Research prototype. Not a medical device. Code is provided for reproducibility.
