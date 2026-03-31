import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
import numpy as np
from scipy.stats import entropy

# -----------------------------
# 1️⃣ Signal-to-noise ratio (SNR)
# -----------------------------
def snr_per_protein(adata):
    """
    Compute SNR = mean / std for each protein across aggregated units.

    Parameters
    ----------
    adata : AnnData
        Aggregated expression matrix (cells x proteins)

    Returns
    -------
    pd.Series : SNR per protein
    """
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray() if hasattr(X, "toarray") else X.A
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    snr = means / (stds + 1e-8)
    return pd.Series(snr, index=adata.var_names)


# -----------------------------
# 2️⃣ Coefficient of variation (CV)
# -----------------------------
def cv_per_protein(adata):
    """
    Compute CV = std / mean per protein across aggregated units.

    Parameters
    ----------
    adata : AnnData

    Returns
    -------
    pd.Series : CV per protein
    """
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray() if hasattr(X, "toarray") else X.A
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    cv = stds / (means + 1e-8)
    return pd.Series(cv, index=adata.var_names)


# -----------------------------
# 3️⃣ Dynamic range per protein
# -----------------------------
def dynamic_range_per_protein(adata):
    """
    Compute dynamic range = max - min per protein across aggregated units.

    Parameters
    ----------
    adata : AnnData

    Returns
    -------
    pd.Series : dynamic range per protein
    """
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray() if hasattr(X, "toarray") else X.A
    rng = np.max(X, axis=0) - np.min(X, axis=0)
    return pd.Series(rng, index=adata.var_names)

# -----------------------------
# 4️⃣ Sparsity (fraction of zeros)
# -----------------------------
def sparsity_per_protein(adata):
    """
    Compute fraction of zeros per protein.

    Parameters
    ----------
    adata : AnnData

    Returns
    -------
    pd.Series : sparsity per protein
    """
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray() if hasattr(X, "toarray") else X.A
    frac_zeros = np.sum(X == 0, axis=0) / X.shape[0]
    return pd.Series(frac_zeros, index=adata.var_names)
