import pandas as pd, torch
from pathlib import Path
from src.graph.graph_builder import validate_graph

df = pd.read_csv("data/processed/graph_stats.csv")
bridged = df[df["bridged"] == True]   # may be "bridged" or "True"/"True" depending on dtype
print(f"{len(bridged)} bridged tiles in stats")

bad = []
for _, row in bridged.iterrows():
    p = Path("data/processed/graphs") / row["split"] / f"{row['stem']}.pt"
    if not p.exists():
        continue
    data = torch.load(str(p), map_location="cpu", weights_only=False)
    checks = validate_graph(data)
    if not checks["single_component"]:
        bad.append(row["stem"])

print("Disconnected after bridging:", bad if bad else "NONE — all bridged tiles are connected ✅")