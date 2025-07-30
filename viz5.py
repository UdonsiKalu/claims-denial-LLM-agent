import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load your passive maps CSV
df = pd.read_csv("passive_maps.csv")

# Select x, y data columns
x = df['token_density']
y = df['neighbor_overlap']

# Calculate the point density
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

# Sort the points by density so that dense points plot on top
idx = z.argsort()
x, y, z = x.iloc[idx], y.iloc[idx], z[idx]

# Plot scatter with color coding for density
plt.figure(figsize=(10, 7))
sc = plt.scatter(x, y, c=z, cmap='viridis', s=20, edgecolors='none', alpha=0.7)
plt.colorbar(sc, label='Density Concentration')
plt.xlabel('Token Density')
plt.ylabel('Neighbor Overlap')
plt.title('Scatter Plot Colored by Density Concentration')
plt.tight_layout()
plt.show()
