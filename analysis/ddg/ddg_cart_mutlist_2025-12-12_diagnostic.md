## Diagnosis: `runs/2025-12-12_ddg_cart_mutlist/outputs/` did not process mutations

### What we see in the outputs
- **`mutlist.json`**: 3 entries, keys are only `mutations` + `scores`.
  - Every entry has **`mutations: []`**.
- **`mutlist.ddg` / `mutlist.clean.ddg`**: only `COMPLEX: ... WT_` lines for Round1–Round3.
  - There are **no `MUT_...`** labels and no `MUTATION:` records.

### Conclusion
This run contains **WT scoring only** (3 rounds) and **no mutant calculations**, so it is **not possible** to compute per-mutation ΔΔG statistics (mean/std/tier) from this run.

### Most likely cause
The `mutlist.clean.mut` exists, but Rosetta did not accept any mutations (classic causes):
- **Residue numbering mismatch** (Rosetta cartesian_ddg expects *pose numbering* 1..N).
- **WT amino-acid mismatch** at the specified positions.
- A silent/older-script failure mode where Rosetta runs but effectively performs only WT rounds.

### What to do next (in this repo)
Re-run ddG using the current script (it has a pre-flight check that fails fast instead of producing empty outputs):

```bash
bash scripts/rosetta_ddg.sh runs/*relax*/outputs/*.pdb configs/rosetta/mutlist.mut
```

Then summarize results:

```bash
python scripts/parse_ddg.py 'runs/*ddg*/outputs/*.json' results/ddg_scans/phase1_raw.csv
python scripts/summarize_ddg.py results/ddg_scans/phase1_raw.csv --out analysis/ddg/phase1
```

(You already have a usable Phase-1 summary at `analysis/ddg/phase1.csv` / `analysis/ddg/phase1.md` based on runs that *did* produce mutation records.)
