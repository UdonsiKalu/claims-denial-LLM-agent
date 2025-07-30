import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your passive maps CSV
df = pd.read_csv("passive_maps.csv")

# Heatmap using 2D hex bins (density)
plt.figure(figsize=(10, 8))
hb = plt.hexbin(
    df['token_density'], 
    df['neighbor_overlap'], 
    gridsize=40, 
    cmap='viridis', 
    bins='log'  # log color scale for count
)

plt.colorbar(hb, label='Log(Chunk Count)')
plt.xlabel('Token Density')
plt.ylabel('Neighbor Overlap')
plt.title('Heatmap: Token Density × Neighbor Overlap')
plt.tight_layout()
plt.show()
