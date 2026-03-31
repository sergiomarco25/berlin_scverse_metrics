import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1️⃣ Variance explained by technical covariates
# -----------------------------
def variance_explained_by_covariates(adata, covariates=["total_counts", "pct_mito", "batch"]):
    """
    Fit a linear model per gene: gene expression ~ technical covariates.
    Return average R² across genes.

    Parameters
    ----------
    adata : AnnData
        Normalized expression matrix
    covariates : list
        Columns in adata.obs representing technical factors

    Returns
    -------
    float : mean R² across genes
    """
    X = pd.get_dummies(adata.obs[covariates], drop_first=True).values
    Y = adata.X
    if not isinstance(Y, np.ndarray):
        Y = Y.toarray() if hasattr(Y, "toarray") else Y.A  # handle sparse
    r2_list = []
    for i in range(Y.shape[1]):
        y = Y[:, i]
        model = LinearRegression().fit(X, y)
        r2_list.append(model.score(X, y))
    return np.mean(r2_list)


# -----------------------------
# 2️⃣ Mean-variance correlation
# -----------------------------
def mean_variance_correlation(adata):
    """
    Compute correlation between mean and variance per gene.

    Parameters
    ----------
    adata : AnnData
        Normalized expression matrix

    Returns
    -------
    float : Pearson correlation between mean and variance
    """
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray() if hasattr(X, "toarray") else X.A
    gene_means = np.mean(X, axis=0)
    gene_vars = np.var(X, axis=0)
    corr = np.corrcoef(gene_means, gene_vars)[0, 1]
    return corr


# -----------------------------
# 3️⃣ Library size dependence on PCs
# -----------------------------
def library_size_pc_correlation(adata, n_pcs=20, covariate="total_counts"):
    """
    Compute correlation between PCs and a technical covariate (e.g., total_counts)

    Parameters
    ----------
    adata : AnnData
        Normalized expression matrix
    n_pcs : int
        Number of PCs to compute
    covariate : str
        Column in adata.obs

    Returns
    -------
    float : mean absolute correlation across PCs
    """
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray() if hasattr(X, "toarray") else X.A
    pca = PCA(n_components=n_pcs)
    pcs = pca.fit_transform(X)
    cov_values = adata.obs[covariate].values
    corrs = [abs(np.corrcoef(pcs[:, i], cov_values)[0, 1]) for i in range(n_pcs)]
    return np.mean(corrs)

