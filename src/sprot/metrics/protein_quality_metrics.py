import anndata as ad
import spatialdata as sd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path
import scipy.stats as stats
import numpy as np
import pandas as pd


def metric_sparsity_gini(sdata, image_key, labels_key, protein):
    """
    Calculates the Gini coefficient to measure the sparsity of protein expression 
    across the cell population.
    
    A Gini coefficient of 0 indicates perfectly even distribution (all cells have 
    the same mean intensity), while a value approaching 1 indicates high sparsity 
    (most signal is concentrated in a very small number of cells).

    Args:
        sdata (SpatialData): The SpatialData object containing images and labels.
        image_key (str): Key for the image data in sdata.
        labels_key (str): Key for the segmentation masks (labels) in sdata.
        protein (str): The specific channel/protein name to analyze.

    Returns:
        float: Gini coefficient value between 0.0 and 1.0.
    """
    img_data, mask_data = get_protein_data(sdata, image_key, labels_key, protein)
    # Get mean intensity per cell
    cell_means = [np.mean(img_data[mask_data == label]) for label in np.unique(mask_data[mask_data > 0])]
    
    if not cell_means or np.sum(cell_means) == 0: return 0.0
    x = np.sort(cell_means)
    n = len(x)
    return (n + 1 - 2 * np.sum(np.cumsum(x)) / np.sum(x)) / n

def metric_rel_sni(sdata, image_key, labels_key, protein, thresh=0.0):
    """
    Calculates the Relative Signal-to-Noise Integral (Rel_SNI).
    
    This metric measures the area between the cell intensity distribution curve 
    and the background noise profile, normalized by the Otsu threshold. It 
    quantifies how much "real" signal exists above the expected background noise.

    Args:
        sdata (SpatialData): The SpatialData object containing images and labels.
        image_key (str): Key for the image data in sdata.
        labels_key (str): Key for the segmentation masks (labels) in sdata.
        protein (str): The specific channel/protein name to analyze.
        thresh (float, optional): The intensity threshold (e.g., Otsu). 
            Defaults to 0.0. If > 0, the result is normalized by this value.

    Returns:
        float: The normalized SNI value. Higher values indicate a stronger 
            signal-to-noise separation.
    """
    df_cells, df_bg = get_processed_distributions(sdata, image_key, labels_key, protein, thresh)
    
    y_cell = df_cells['mean_intensity'].values
    y_bg = np.interp(np.linspace(0, 100, len(df_cells)), df_bg['percentile'], df_bg['mean_intensity'])
    
    raw_area = np.trapz(y_cell - y_bg) / len(df_cells)
    return raw_area / thresh if thresh > 0 else raw_area

def metric_intracell_coverage(sdata, image_key, labels_key, protein, thresh=0.0):
    """
    Measures the Proportion of Positive Pixels (Coverage) within the brightest cells.
    
    This identifies the top 5% of cells (by mean intensity) and calculates what 
    fraction of their internal pixels exceed the threshold. This helps distinguish 
    between true cellular staining (high coverage) and punctate noise/artifacts 
    (low coverage).

    Args:
        sdata (SpatialData): The SpatialData object containing images and labels.
        image_key (str): Key for the image data in sdata.
        labels_key (str): Key for the segmentation masks (labels) in sdata.
        protein (str): The specific channel/protein name to analyze.
        thresh (float, optional): The intensity threshold to determine a 
            'positive' pixel. Defaults to 0.0.

    Returns:
        float: Mean proportion of positive pixels (0.0 to 1.0) in the top 
            5% brightest cells.
    """
    df_cells, _ = get_processed_distributions(sdata, image_key, labels_key, protein, thresh)
    top_5_pct = df_cells.tail(max(1, int(len(df_cells) * 0.05)))
    return top_5_pct['prop_positive'].mean()

def get_protein_data(sdata, image_key, labels_key, protein):
    """Internal helper to extract numpy arrays from sdata."""
    img_data = sdata[image_key].sel(c=protein).values
    mask_data = sdata[labels_key].values.squeeze()
    return img_data, mask_data

def get_processed_distributions(sdata, image_key, labels_key, protein, thresh=0.0):
    """Extracts the dataframes needed for SNI and plotting."""
    img_data, mask_data = get_protein_data(sdata, image_key, labels_key, protein)
    
    # Process Cells
    unique_labels = np.unique(mask_data[mask_data > 0])
    results = []
    for label in unique_labels:
        pixels = img_data[mask_data == label]
        results.append({
            'mean_intensity': np.mean(pixels),
            'prop_positive': np.mean(pixels > thresh) if thresh > 0 else 1.0
        })
    
    df_cells = pd.DataFrame(results).sort_values('mean_intensity').reset_index(drop=True)
    df_cells['cell_rank_pct'] = (df_cells.index / len(df_cells)) * 100
    
    # Process Background
    bg_pixels = img_data[mask_data == 0]
    avg_cell_size = int(np.sum(mask_data > 0) / len(df_cells)) if len(df_cells) > 0 else 100
    num_samples = min(len(df_cells), len(bg_pixels) // avg_cell_size)
    
    if num_samples > 10:
        bg_samples = np.random.choice(bg_pixels, (num_samples, avg_cell_size), replace=False)
        df_bg = pd.DataFrame({
            'mean_intensity': np.sort(np.mean(bg_samples, axis=1)),
            'prop_positive': np.sort(np.mean(bg_samples > thresh, axis=1)) if thresh > 0 else 0.0,
            'percentile': np.linspace(0, 100, num_samples)
        })
    else:
        df_bg = pd.DataFrame({'mean_intensity': [0.0], 'prop_positive': [0.0], 'percentile': [0.0]})
        
    return df_cells, df_bg
