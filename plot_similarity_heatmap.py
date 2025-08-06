import os
import json
import plotly.graph_objects as go
from faiss_gpu_react5 import CMSDenialAnalyzer  # Adjust path/module name if needed

# Step 1: Ensure visualization_data.json exists
if not os.path.exists("visualization_data.json"):
    print("🔄 visualization_data.json not found. Generating from test claim...")

    analyzer = CMSDenialAnalyzer()
    test_claim = {
        "cpt_code": "99214",
        "diagnosis": "Z79.899",
        "modifiers": [],
        "payer": "Medicare"
    }
    analyzer.save_visualization_data(test_claim)
    print("✅ visualization_data.json created.\n")

# Step 2: Load visualization data
with open("visualization_data.json", "r") as f:
    data = json.load(f)

# Step 3: Extract node IDs and similarity scores
x_labels = [node["id"] for node in data["nodes"] if node["type"] == "document"]
z_values = [edge["similarity"] for edge in data["edges"]]

# Step 4: Build Plotly heatmap
fig = go.Figure(data=go.Heatmap(
    z=[z_values],  # Single-row heatmap
    x=x_labels,
    y=["Claim Query"],
    colorscale="YlOrRd",
    colorbar=dict(title="Similarity Score"),
    hoverinfo="x+z"
))

fig.update_layout(
    title="CMS Document Similarity Heatmap",
    xaxis_title="Retrieved CMS Chunks",
    yaxis_title="Query",
    height=400
)

# Step 5: Save and display
fig.write_html("similarity_heatmap.html")
print("📊 Heatmap saved as similarity_heatmap.html")
fig.show()
