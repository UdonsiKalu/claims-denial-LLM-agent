import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your passive maps CSV
df = pd.read_csv("passive_maps.csv")

# Optional: Normalize or clip extreme outliers if needed
# df = df[(df['token_density'] < 1000) & (df['neighbor_overlap'] < 1.0)]

# Seaborn style
sns.set(style="white", font_scale=1.2)

# Use seaborn's pairplot with KDE & 2D density heatmap
pair = sns.pairplot(
    df[['token_density', 'neighbor_overlap', 'heading_depth']],
    kind='kde',           # Off-diagonal uses contour KDE
    diag_kind='kde',      # Diagonal uses KDE instead of hist
    corner=False,         # Set to True if you want lower triangle only
    plot_kws={
        'fill': True,     # Fill contours
        'cmap': 'mako',   # Color map
        'thresh': 0.05,   # Minimum density threshold
        'levels': 10      # Number of contour levels
    }
)

pair.fig.suptitle("Token Density × Overlap × Heading Depth — KDE Density Matrix", y=1.02)
plt.tight_layout()
plt.show()
