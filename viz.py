import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("passive_maps.csv")

# Distribution of heading depths
plt.figure()
df['heading_depth'].value_counts().sort_index().plot(kind='bar')
plt.title("Heading Depth Distribution")
plt.xlabel("Heading Level")
plt.ylabel("Number of Chunks")
plt.savefig("heading_depth_distribution.png")
plt.close()

# Scatter token_density vs neighbor_overlap
plt.figure()
plt.scatter(df['token_density'], df['neighbor_overlap'], alpha=0.5)
plt.title("Token Density vs Neighbor Overlap")
plt.xlabel("Token Density")
plt.ylabel("Neighbor Overlap")
plt.savefig("token_density_vs_neighbor_overlap.png")
plt.close()
