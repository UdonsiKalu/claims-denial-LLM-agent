import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Load your passive maps CSV
df = pd.read_csv("passive_maps.csv")

variables = ['token_density', 'neighbor_overlap', 'heading_depth']

def scatter_density(x, y, **kwargs):
    # Calculate point density
    xy = np.vstack([x, y])
    try:
        z = gaussian_kde(xy)(xy)
    except np.linalg.LinAlgError:
        # Add tiny noise if KDE fails
        xy_jittered = xy + np.random.normal(scale=1e-4, size=xy.shape)
        try:
            z = gaussian_kde(xy_jittered)(xy_jittered)
        except np.linalg.LinAlgError:
            z = np.ones_like(x)
    
    idx = z.argsort()
    x, y, z = x.iloc[idx], y.iloc[idx], z[idx]

    plt.scatter(x, y, c=z, cmap='viridis', s=20, edgecolors='none', alpha=0.7)

def diag_dist(x, **kwargs):
    # Clear current axes and plot shaded histogram + KDE curve
    plt.cla()
    sns.histplot(x, bins=30, kde=True, color='skyblue', stat='density', edgecolor='black', linewidth=0.5)
    plt.xlabel(kwargs.get('label', ''))
    plt.ylabel('Density')

g = sns.PairGrid(df[variables])

# Diagonal with hist + KDE
g.map_diag(diag_dist)

# Off-diagonal with density-colored scatter
g.map_offdiag(scatter_density)

plt.suptitle("9-Panel Pairplot with Token Density Distribution and Density-Colored Scatter Dots", y=1.02)
plt.tight_layout()
plt.show()
