## Plotting functions
import matplotlib.pyplot as plt
import seaborn as sns

def plot_diagnostic_curves(df_cells, df_bg, protein, thresh):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Intensity Profiles
    ax1.plot(df_cells['cell_rank_pct'], df_cells['mean_intensity'], color='orange', label='Cells')
    ax1.plot(df_bg['percentile'], df_bg['mean_intensity'], color='green', linestyle='--', label='BG Noise')
    ax1.axhline(thresh, color='red', linestyle=':', label=f'Otsu: {thresh:.2f}')
    #ax1.set_yscale('log')
    ax1.set_title(f'Intensity Profile: {protein}')
    ax1.legend()

    # Right: Detection Confidence
    ax2.scatter(df_cells['cell_rank_pct'], df_cells['prop_positive'], color='gray', alpha=0.1, s=2)
    ax2.plot(df_bg['percentile'], df_bg['prop_positive'], color='green', linestyle='--')
    ax2.set_title('Detection Confidence (% Pixels > Thresh)')
    plt.tight_layout()
    plt.show()

def plot_spatial_roi(sdata, image_key, labels_key, protein, x_range, y_range):
    img, mask = get_protein_data(sdata, image_key, labels_key, protein)
    roi = (slice(y_range[0], y_range[1]), slice(x_range[0], x_range[1]))
    
    img_roi = img[roi]
    mask_roi = mask[roi]
    
    # Rescale for visibility
    v_min, v_max = np.percentile(img_roi, (1, 99))
    rescaled = np.clip((img_roi - v_min) / (v_max - v_min), 0, 1)
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(rescaled, cmap='magma')
    ax[1].imshow(rescaled, cmap='gray')
    ax[1].imshow(np.ma.masked_where(mask_roi == 0, mask_roi), cmap='prism', alpha=0.4)
    for a in ax: a.axis('off')
    plt.show()


def plot_protein_scatter(df, x, y, hue, title, xlabel, ylabel, palette='viridis', ax=None):
    """
    A versatile scatter plotter for protein metric dataframes.
    
    Args:
        df (pd.DataFrame): The summary dataframe.
        x, y, hue (str): Column names for axes and color coding.
        title, xlabel, ylabel (str): Labels for the plot.
        palette (str): Seaborn color palette.
        ax (matplotlib.axes): Existing axis to plot on (optional).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 8))
    
    sns.set_context("talk")
    
    # 1. Create the scatter plot
    scatter = sns.scatterplot(
        data=df, x=x, y=y, hue=hue, 
        s=150, ax=ax, palette=palette, edgecolor='black', alpha=0.8
    )
    
    # 2. Add text labels for each protein
    for i in range(len(df)):
        row = df.iloc[i]
        ax.text(
            row[x], row[y], f"  {row['Protein']}", 
            fontsize=10, weight='semibold', va='center'
        )
    
    # 3. Styling
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    return ax

def get_protein_data(sdata, image_key, labels_key, protein):
    """Internal helper to extract numpy arrays from sdata."""
    img_data = sdata[image_key].sel(c=protein).values
    mask_data = sdata[labels_key].values.squeeze()
    return img_data, mask_data