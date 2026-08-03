"""
covariance.py
--------------
Utilities to verify that an estimated matrix is a *valid* covariance
matrix: symmetric, positive semi-definite (PSD), and finite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    is_symmetric: bool
    is_psd: bool
    is_finite: bool
    min_eigenvalue: float
    max_asymmetry: float

    @property
    def is_valid(self) -> bool:
        return self.is_symmetric and self.is_psd and self.is_finite


def validate_covariance(
    Sigma: Union[pd.DataFrame, np.ndarray], tol: float = 1e-8
) -> ValidationResult:
    """
    Check that `Sigma` is a valid covariance matrix:
      1. all entries finite (no NaN/Inf),
      2. symmetric (Sigma == Sigma.T within numerical tolerance),
      3. positive semi-definite (smallest eigenvalue >= -tol).
    """
    S = Sigma.values if isinstance(Sigma, pd.DataFrame) else np.asarray(Sigma)

    is_finite = bool(np.isfinite(S).all())
    max_asym = float(np.max(np.abs(S - S.T))) if is_finite else float("inf")
    is_symmetric = max_asym < 1e-6

    if is_finite:
        eigvals = np.linalg.eigvalsh((S + S.T) / 2.0)
        min_eig = float(eigvals.min())
    else:
        min_eig = float("-inf")
    is_psd = min_eig >= -tol

    return ValidationResult(
        is_symmetric=is_symmetric,
        is_psd=is_psd,
        is_finite=is_finite,
        min_eigenvalue=min_eig,
        max_asymmetry=max_asym,
    )
