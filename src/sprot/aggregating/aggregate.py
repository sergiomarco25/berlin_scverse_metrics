import numpy as np
import anndata as ad
from skimage.measure import regionprops
from tqdm import tqdm

def sdata_aggregate(
    sdata,
    values_name,
    by_name,
    mode="percentile",
    percentile=95
):
    """

    Aggregation of image values per shape in a SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing images and shapes.
    values_name : str
        Name of the image layer to aggregate.
    by_name : str
        Name of the shape layer (labels) to aggregate over.
    mode : str
        Aggregation mode: 'percentile', 'mean', 'sum', 'median'.
    percentile : float
        Percentile to compute (used only if mode='percentile').

    Returns
    -------
    AnnData
        Each row corresponds to a shape/cell, each column to a channel.
    """
    img = sdata[values_name].values
    labels = sdata[by_name].values

    # Handle channels
    if img.ndim == 4:  # assume (batch, C, H, W)
        img = img[0]
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
    C, H, W = img.shape
    if labels.shape != (H, W):
        raise ValueError(f"Shape mismatch: {img.shape} vs {labels.shape}")

    # Get cell IDs
    cell_ids = np.unique(labels)
    cell_ids = cell_ids[cell_ids != 0]
    n_cells = len(cell_ids)

    result_matrix = np.zeros((n_cells, C), dtype=float)

    # Use skimage.regionprops for faster per-label extraction
    for c in tqdm(range(C)):
        channel_img = img[c]
        props = regionprops(labels, intensity_image=channel_img)
        for i, prop in enumerate(props):
            if mode == "mean":
                result_matrix[i, c] = prop.mean_intensity
            elif mode == "sum":
                result_matrix[i, c] = prop.intensity_image.sum()
            elif mode == "median":
                result_matrix[i, c] = np.median(prop.intensity_image)
            elif mode == "percentile":
                result_matrix[i, c] = np.percentile(
                    prop.intensity_image, percentile
                )
            else:
                raise NotImplementedError(f"Mode {mode} not implemented")

    # Channel names
    if mode == "percentile":
        suffix = f"p{percentile}"
    else:
        suffix = mode
    channel_names = [f"{values_name}_ch{i}_{suffix}" for i in range(C)]

    # AnnData
    adata_out = ad.AnnData(X=result_matrix)
    adata_out.obs["cell_id"] = cell_ids
    adata_out.var_names = channel_names

    return adata_out
