import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import pandas as pd
import numpy as np
import pandas as pd

def gene_pair_cosine_similarity(adata1, adata2):
    """
    Compute cosine similarity for all pairs of genes between two AnnData objects.
    Each gene is represented by its vector of values across cells.

    Parameters
    ----------
    adata1, adata2 : AnnData
        Must contain 'cell_id' in obs to align cells.

    Returns
    -------
    pd.DataFrame
        DataFrame with genes from adata1 as rows, genes from adata2 as columns,
        values are cosine similarities.
    """
    # Align cells by cell_id
    common_cells = np.intersect1d(adata1.obs['cell_id'], adata2.obs['cell_id'])
    ad1 = adata1[adata1.obs['cell_id'].isin(common_cells), :]
    ad2 = adata2[adata2.obs['cell_id'].isin(common_cells), :]

    # Sort by cell_id to ensure order matches
    ad1 = ad1[np.argsort(ad1.obs['cell_id']), :]
    ad2 = ad2[np.argsort(ad2.obs['cell_id']), :]

    # Get dense matrices (cells x genes)
    X1 = ad1.X
    if hasattr(X1, "todense"):
        X1 = np.array(X1.todense())
    else:
        X1 = np.array(X1)

    X2 = ad2.X
    if hasattr(X2, "todense"):
        X2 = np.array(X2.todense())
    else:
        X2 = np.array(X2)

    # Compute cosine similarity: each row is a cell, columns are genes
    # So we transpose to have genes as rows
    cos_sim_matrix = cosine_similarity(X1.T, X2.T)  # shape: genes1 x genes2
    cosim_df=pd.DataFrame(cos_sim_matrix, index=ad1.var_names, columns=ad2.var_names)
    out=pd.DataFrame(index=cosim_df.columns,columns=['cosine_similarity'])
    for col in cosim_df.columns:
        cosim=cosim_df.loc[col,col]
        out.loc[col,'cosine_similarity']=cosim

    return out


def compute_all_method_cosine(agg_results):
    """
    Compute diagonal gene cosine similarity for all ordered pairs of aggregation methods,
    including self-comparisons and reverse pairs.

    Parameters
    ----------
    agg_results : dict
        Key = method name, value = AnnData object.

    Returns
    -------
    pd.DataFrame
        Columns: ['method1', 'method2', 'gene', 'cosine_similarity']
        One row per gene per pair of methods.
    """
    records = []
    methods = list(agg_results.keys())

    # Use product to get all ordered pairs including self-comparisons
    for method1, method2 in itertools.product(methods, repeat=2):
        ad1 = agg_results[method1]
        ad2 = agg_results[method2]

        cosim_df = gene_pair_cosine_similarity(ad1, ad2)  # diagonal version

        for gene, row in cosim_df.iterrows():
            records.append({
                'method1': method1,
                'method2': method2,
                'gene': gene,
                'cosine_similarity': row['cosine_similarity']
            })

    return pd.DataFrame(records)


import pandas as pd

def cosine_vs_mean(all_cosim, reference_method='mean'):
    """
    Compute, per gene, cosine similarity of all aggregation methods vs a reference method (default 'mean').

    Parameters
    ----------
    all_cosim : pd.DataFrame
        Must contain columns ['method1', 'method2', 'gene', 'cosine_similarity'].
    reference_method : str
        The reference method to compare against (e.g., 'mean').

    Returns
    -------
    pd.DataFrame
        Rows = genes, columns = aggregation methods (excluding reference),
        values = cosine similarity vs reference.
    """
    # Keep only rows where method1 or method2 is the reference
    df_ref = all_cosim[(all_cosim['method1'] == reference_method) | (all_cosim['method2'] == reference_method)].copy()

    # Ensure reference is always in method1
    df_ref.loc[df_ref['method2'] == reference_method, ['method1','method2']] = df_ref.loc[df_ref['method2'] == reference_method, ['method2','method1']].values

    # Drop duplicates in case of repeated entries
    df_ref = df_ref.drop_duplicates(subset=['gene','method2'])

    # Keep only non-reference methods
    df_ref = df_ref[df_ref['method2'] != reference_method]

    # Pivot to gene x method2
    gene_vs_method = df_ref.pivot(index='gene', columns='method2', values='cosine_similarity')

    return gene_vs_method


def compute_gene_percentile95(agg_results, percentile=95):
    """
    Compute a given percentile (default 95) per gene for each aggregation method.

    Parameters
    ----------
    agg_results : dict
        Key = method name, value = AnnData object (cells x genes)
    percentile : float
        Percentile to compute.

    Returns
    -------
    pd.DataFrame
        Rows = genes, columns = aggregation methods, values = percentile of all cells
    """
    records = []

    for method, adata in agg_results.items():
        # Get dense array
        X = adata.X
        if hasattr(X, "todense"):
            X = np.array(X.todense())
        else:
            X = np.array(X)

        # Compute percentile per gene
        gene_percentiles = np.percentile(X, percentile, axis=0)

        for gene, val in zip(adata.var_names, gene_percentiles):
            records.append({
                'gene': gene,
                'method': method,
                'percentile_value': val
            })

    df = pd.DataFrame(records)
    # Pivot to gene x method table
    gene_vs_method = df.pivot(index='gene', columns='method', values='percentile_value')

    return gene_vs_method


import numpy as np
import pandas as pd
from scipy.stats import entropy

def kl_compare_adata(adata1, adata2, layer=None, bins=50, pseudocount=1e-8):
    """
    Compare per-gene distributions from two AnnData objects using KL divergence.

    Parameters
    ----------
    adata1 : anndata.AnnData
        First AnnData object (aggregation method 1).
    adata2 : anndata.AnnData
        Second AnnData object (aggregation method 2).
    layer : str, optional
        Name of the layer to use instead of .X. If None, use .X.
    bins : int, default=50
        Number of bins to discretize expression for KL computation.
    pseudocount : float, default=1e-8
        Small value to avoid log(0) issues in KL divergence.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['gene', 'kl_divergence']
    """

    # Check genes match
    if not all(adata1.var_names == adata2.var_names):
        raise ValueError("Gene names in both AnnData objects must match and be in the same order.")

    X1 = adata1.layers[layer] if layer else adata1.X
    X2 = adata2.layers[layer] if layer else adata2.X

    # Convert sparse to dense if needed
    if hasattr(X1, "toarray"):
        X1 = X1.toarray()
    if hasattr(X2, "toarray"):
        X2 = X2.toarray()

    genes = adata1.var_names
    kl_list = []

    for i, gene in enumerate(genes):
        data1 = X1[:, i]
        data2 = X2[:, i]

        # Define common bins across both distributions
        combined = np.concatenate([data1, data2])
        hist1, bin_edges = np.histogram(data1, bins=bins, range=(combined.min(), combined.max()), density=True)
        hist2, _ = np.histogram(data2, bins=bin_edges, density=True)

        # Add pseudocount to avoid zero probabilities
        hist1 += pseudocount
        hist2 += pseudocount

        # Normalize to sum to 1
        hist1 /= hist1.sum()
        hist2 /= hist2.sum()

        # Compute KL divergence: KL(hist1 || hist2)
        kl_value = entropy(hist1, hist2)
        kl_list.append(kl_value)

    return pd.DataFrame({'gene': genes, 'kl_divergence': kl_list})
