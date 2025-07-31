import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# Load data
df = pd.read_csv("passive_maps.csv")

# Normalize 'heading_depth' for coloring
norm = plt.Normalize(df['heading_depth'].min(), df['heading_depth'].max())
cmap = plt.cm.viridis

# Define the custom scatter function
def scatter_with_colors(x, y, **kwargs):
    kwargs.pop('color', None)
    kwargs.pop('c', None)
    indices = x.index if isinstance(x, pd.Series) else range(len(x))
    local_colors = cmap(norm(df.loc[indices, 'heading_depth']))
    plt.scatter(x, y, c=local_colors, **kwargs)

# Create a grid layout with room for colorbar
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])
main_ax = plt.subplot(gs[0])

# Create the seaborn PairGrid
pairgrid = sns.PairGrid(
    df[['token_density', 'neighbor_overlap', 'heading_depth']],
    corner=True,
    diag_sharey=False
)
pairgrid.map_lower(scatter_with_colors, s=20, edgecolors='w', alpha=0.7)
pairgrid.map_diag(sns.kdeplot, lw=2, warn_singular=False)

# Add a colorbar using a dedicated axis
cax = fig.add_subplot(gs[1])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cax, label='Heading Depth')

# Final adjustments
pairgrid.fig.suptitle("Pairplot with Points Colored by Heading Depth", y=1.02)
plt.tight_layout()
plt.show()
