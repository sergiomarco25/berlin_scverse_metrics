import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt
import seaborn as sns
import matplotlib.pyplot as plt


def plot_gene_histograms(adata, layer=None, ncols=4, figsize=(12, 8), bins=30, log=False):
    """
    Plot histogram distributions of expression values for each gene in an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing gene expression data.
    layer : str, optional
        Name of the layer to use instead of .X. If None, use adata.X.
    ncols : int, default=4
        Number of columns in the subplot grid.
    figsize : tuple, default=(12, 8)
        Figure size.
    bins : int, default=30
        Number of bins for histograms.
    log : bool, default=False
        Whether to plot log-scaled expression values.
    """

    # Select matrix to use
    X = adata.layers[layer] if layer else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    n_genes = adata.n_vars
    nrows = ceil(n_genes / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, gene in enumerate(adata.var_names):
        ax = axes[i]
        data = X[:, i]
        if log:
            data = np.log1p(data)
        ax.hist(data, bins=bins, color='steelblue', alpha=0.7)
        ax.set_title(gene, fontsize=9)
        ax.set_xlabel("Expression" + (" (log1p)" if log else ""))
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()



def plot_heatmap(all_cosim,metric='kl_divergence',figsize=(6,6),cmap='viridis'):
    """
    Plot a heatmap of mean cosine similarity between aggregation methods.

    Parameters
    ----------
    all_cosim : pd.DataFrame
        Must contain columns ['method1', 'method2', 'gene', 'cosine_similarity'].
    """
    # Compute mean cosine similarity per method pair
    mean_cosim = all_cosim.groupby(['method1', 'method2'])[metric].mean().reset_index()

    # Pivot for heatmap
    heatmap_data = mean_cosim.pivot(index='method2', columns='method1', values=metric)

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap=cmap)
    plt.title(metric)
    plt.xlabel("Method 1")
    plt.ylabel("Method 2")
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_aggregation_histograms(agg_results, genes, bins=30, figsize_per_plot=(4,3)):
    methods = list(agg_results.keys())
    n_genes = len(genes)
    n_methods = len(methods)

    fig, axes = plt.subplots(n_genes, n_methods, figsize=(figsize_per_plot[0]*n_methods,
                                                         figsize_per_plot[1]*n_genes),
                             squeeze=False)

    for i, gene in enumerate(genes):
        for j, method in enumerate(methods):
            adata = agg_results[method]
            if gene not in adata.var_names:
                axes[i, j].text(0.5, 0.5, f"{gene} not found", ha='center', va='center')
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                continue

            vals = adata[:, gene].X
            if hasattr(vals, "todense"):  # sparse
                vals = np.array(vals.todense()).ravel()
            else:
                vals = np.array(vals).ravel()

            axes[i, j].hist(vals, bins=bins, color='skyblue', edgecolor='k')
            if i == 0:
                axes[i, j].set_title(method, fontsize=12)
            if j == 0:
                axes[i, j].set_ylabel(gene, fontsize=12)

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_channels(img_array, all_channel_names, selected_channels=None, roi=None, cmap="Greys"):
    """
    Plot multiple channels from a numpy image array in subplots.

    Parameters
    ----------
    img_array : np.ndarray
        Shape (channels, y, x)
    all_channel_names : list of str
        Names of all channels corresponding to img_array
    selected_channels : list of str or None
        Names of channels to plot. If None, plot all channels.
    roi : tuple or None
        (x_min, x_max, y_min, y_max) in pixel coordinates. If None, plot full image.
    cmap : str
        Colormap for imshow
    """
    if selected_channels is None:
        selected_channels = all_channel_names

    # Map selected_channels to indices in img_array
    channel_indices = [all_channel_names.index(ch) for ch in selected_channels]

    n_channels = len(channel_indices)
    fig, axes = plt.subplots(1, n_channels, figsize=(4*n_channels, 4))

    if n_channels == 1:
        axes = [axes]  # ensure iterable

    for ax, idx in zip(axes, channel_indices):
        # Extract channel
        ch_img = img_array[idx, :, :]

        # Apply ROI
        if roi is not None:
            x_min, x_max, y_min, y_max = roi
            ch_img = ch_img[y_min:y_max, x_min:x_max]

        ax.imshow(ch_img, cmap=cmap)
        ax.set_title(all_channel_names[idx])
        ax.axis("off")

    plt.tight_layout()
    plt.show()
