"""
Euclidean Alignment (EA) for EEG domain adaptation.

A mathematical step that adjusts each person's brain signals to remove
person-specific scaling and rotation, making signals more comparable across
people. Reference: He et al. 2020, "Align and Pool for EEG Headset Domain
Adaptation."

  euclidean_align(X)         — fit and transform one block (e.g. one subject)
  euclidean_align_fit_apply(X_fit, X_apply) — fit on X_fit, apply to both (e.g. train/test)
"""

import numpy as np
from scipy.linalg import fractional_matrix_power


def euclidean_align(X: np.ndarray, reg: float = 1e-5) -> np.ndarray:
    """
    Euclidean Alignment: whiten each subject's trials to reduce subject-specific
    covariance structure while preserving trial-wise signal.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_trials, n_channels, n_times). One subject's data.
    reg : float
        Regularization added to diagonal of R_bar for numerical stability.

    Returns
    -------
    np.ndarray
        Aligned data, same shape as X.
    """
    n_trials, n_channels, n_times = X.shape
    T = n_times

    # Per-trial covariance R_i = x_i @ x_i.T / T, then average
    R_list = []
    for i in range(n_trials):
        x_i = X[i]  # (n_channels, n_times)
        R_i = (x_i @ x_i.T) / T  # (n_channels, n_channels)
        R_list.append(R_i)
    R_bar = np.mean(R_list, axis=0)

    # Regularize for invertibility
    R_bar += reg * np.eye(n_channels)

    # R_bar^(-1/2)
    R_bar_inv_sqrt = fractional_matrix_power(R_bar, -0.5)

    # Align each trial: x_i_aligned = R_bar^(-1/2) @ x_i
    X_aligned = np.zeros_like(X)
    for i in range(n_trials):
        X_aligned[i] = R_bar_inv_sqrt @ X[i]

    return X_aligned


def euclidean_align_fit_apply(
    X_fit: np.ndarray, X_apply: np.ndarray, reg: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit R_bar on X_fit only, apply R_bar^(-1/2) to both X_fit and X_apply.
    Use for within-subject: fit on session 1, apply to session 1 and session 2.

    Parameters
    ----------
    X_fit : np.ndarray
        Shape (n_trials, n_channels, n_times). Data to fit R_bar (e.g. train/ses-01).
    X_apply : np.ndarray
        Same shape convention. Data to transform with the same R_bar (e.g. test/ses-02).
    reg : float
        Regularization for R_bar.

    Returns
    -------
    X_fit_aligned, X_apply_aligned : np.ndarray
        Same shapes as inputs.
    """
    n_fit, n_channels, n_times = X_fit.shape
    T = n_times
    R_list = []
    for i in range(n_fit):
        x_i = X_fit[i]
        R_i = (x_i @ x_i.T) / T
        R_list.append(R_i)
    R_bar = np.mean(R_list, axis=0) + reg * np.eye(n_channels)
    R_bar_inv_sqrt = fractional_matrix_power(R_bar, -0.5)

    def transform(X):
        out = np.zeros_like(X)
        for i in range(X.shape[0]):
            out[i] = R_bar_inv_sqrt @ X[i]
        return out

    return transform(X_fit), transform(X_apply)
