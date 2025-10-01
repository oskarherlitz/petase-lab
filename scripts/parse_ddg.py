import json, sys, csv, glob, os
# Usage: python scripts/parse_ddg.py runs/2025-09-30_ddg_cart_*/outputs/*.json results/ddg_scans/out.csv
inputs = sorted([p for pat in sys.argv[1:-1] for p in glob.glob(pat)])
out = sys.argv[-1]
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(["run","mutation","ddg_reu"])
    for jpath in inputs:
        with open(jpath) as jf:
            j = json.load(jf)
        run = jpath.split("/")[1]
        for mut, info in j.get("mutations", {}).items():
            w.writerow([run, mut, info.get("ddg")])
