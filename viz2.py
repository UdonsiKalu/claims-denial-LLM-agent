import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Load your passive maps CSV
df = pd.read_csv("passive_maps.csv")

# ========== 3D Scatter Plot ==========
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter_3d = ax.scatter(
    df['token_density'],
    df['neighbor_overlap'],
    df['heading_depth'],
    c=df['heading_depth'],
    cmap='viridis',
    alpha=0.7
)

ax.set_xlabel('Token Density')
ax.set_ylabel('Neighbor Overlap')
ax.set_zlabel('Heading Depth')
plt.title('3D Scatter: Token Density, Neighbor Overlap, Heading Depth')
fig.colorbar(scatter_3d, ax=ax, label='Heading Depth')
plt.tight_layout()
plt.show()

# ========== 2D Scatter Plot with Size & Color ==========
plt.figure(figsize=(10, 7))
scatter_2d = plt.scatter(
    df['token_density'],
    df['neighbor_overlap'],
    c=df['heading_depth'],
    s=df['heading_depth'] * 20,  # scale size by heading depth
    cmap='plasma',
    alpha=0.6,
    edgecolors='w'
)

plt.colorbar(scatter_2d, label='Heading Depth')
plt.xlabel('Token Density')
plt.ylabel('Neighbor Overlap')
plt.title('Token Density vs Neighbor Overlap (Color & Size = Heading Depth)')
plt.tight_layout()
plt.show()

# ========== Pairplot Matrix ==========
sns.pairplot(df[['token_density', 'neighbor_overlap', 'heading_depth']], kind='scatter', diag_kind='kde')
plt.suptitle("Pairwise Scatterplots and Distributions", y=1.02)
plt.show()
