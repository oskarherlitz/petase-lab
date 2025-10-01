import sys, pandas as pd
# Example: python scripts/rank_designs.py results/ddg_scans/out.csv 20
df = pd.read_csv(sys.argv[1])
k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
df = df.sort_values("ddg_reu")
print(df.head(k).to_string(index=False))
