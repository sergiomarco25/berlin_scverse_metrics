import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
import numpy as np
from scipy.stats import entropy
from itertools import combinations


def gene_variance_evenness(adata, layer=None):
    """
    Compute a variance evenness metric for an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Normalized data (log-normalized or similar).
    layer : str, optional
        If specified, use adata.layers[layer] instead of adata.X.

    Returns
    -------
    evenness : float
        Shannon entropy-based evenness of per-gene variance (0-1).
    """
    # Extract data
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    # Ensure dense array
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Compute per-gene variance
    gene_var = np.var(X, axis=0)

    # Avoid zero variance genes
    gene_var = gene_var[gene_var > 0]

    # Fraction of variance per gene
    gene_var_fraction = gene_var / gene_var.sum()

    # Shannon entropy normalized by maximum entropy
    evenness = entropy(gene_var_fraction) / np.log(len(gene_var_fraction))

    return evenness

def equivalent_proteins_correlation(adata, genes):
    """
    Compute a single similarity metric for a list of genes based on their
    expression patterns across cells.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the expression data.
    genes : list of str
        List of gene names to compare (must be in adata.var.index).

    Returns
    -------
    overall_similarity : float
        Average pairwise Pearson correlation between the genes across cells.
    """
    # Extract expression matrix for these genes (cells x genes)
    X = adata[:, genes].X
    if hasattr(X, "toarray"):  # Convert sparse to dense
        X = X.toarray()

    # Remove genes with zero variance
    var_per_gene = np.var(X, axis=0)
    keep = var_per_gene > 1e-8
    X = X[:, keep]
    genes_kept = [g for g, k in zip(genes, keep) if k]

    # If less than 2 genes remain, cannot compute correlation
    if X.shape[1] < 2:
        return np.nan

    # Compute pairwise correlations and average
    corrs = []
    for i, j in combinations(range(X.shape[1]), 2):
        corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
        corrs.append(corr)

    overall_similarity = np.mean(corrs)
    return overall_similarity

