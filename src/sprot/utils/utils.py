import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

def get_protein_data(sdata, image_key, labels_key, protein):
    """Internal helper to extract numpy arrays from sdata."""
    img_data = sdata[image_key].sel(c=protein).values
    mask_data = sdata[labels_key].values.squeeze()
    return img_data, mask_data

def calculate_otsu_threshold(sdata, image_key, labels_key, protein):
    """Calculates Otsu threshold based on pixels within masks."""
    img_data, mask_data = get_protein_data(sdata, image_key, labels_key, protein)
    cell_pixels = img_data[mask_data > 0]
    return float(threshold_otsu(cell_pixels)) if len(cell_pixels) > 0 else 0.0

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