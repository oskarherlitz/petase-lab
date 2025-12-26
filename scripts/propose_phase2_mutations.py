#!/usr/bin/env python3
"""
Propose a Phase-2 mutation list from a Phase-1 ddG summary table.

This repo already produces a Phase-1 summary via scripts/summarize_ddg.py:
  analysis/ddg/phase1.csv  (columns: mutation, mean_ddG, std_ddG, n_replicates, tier, notes)

Phase-2 goal (see docs/NEXT_STEPS_CURSOR.md):
- Focus on positions that look promising (Tier A/B)
- Propose 2–4 rational alternatives per position
- Keep catalytic residues intact (exclude by default)

Outputs a Rosetta cartesian_ddg mut_file (same style as configs/rosetta/mutlist.mut):
  total N
  1
  <WT> <POSE_RESNUM> <CHAIN> <MUT>

Example:
  python scripts/propose_phase2_mutations.py \
    --in analysis/ddg/phase1.csv \
    --out configs/rosetta/mutlist_phase2.mut
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


@dataclass(frozen=True)
class Phase1Row:
    wt: str
    pos: int  # Rosetta pose numbering
    mut: str
    mean_ddg: float
    tier: str

    @property
    def sig(self) -> str:
        return f"{self.wt}{self.pos}{self.mut}"


def _parse_phase1_csv(path: Path) -> List[Phase1Row]:
    rows: List[Phase1Row] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"Empty CSV (no header): {path}")
        for r in reader:
            mut_s = (r.get("mutation") or "").strip()
            mean_s = (r.get("mean_ddG") or "").strip()
            tier = (r.get("tier") or "").strip().upper()
            if not mut_s or not mean_s or not tier:
                continue
            m = _MUT_RE.match(mut_s)
            if not m:
                # Skip multi-muts like "A10B;C20D" or "WT"
                continue
            wt, pos_s, mut = m.group(1), m.group(2), m.group(3)
            try:
                pos = int(pos_s)
                mean = float(mean_s)
            except ValueError:
                continue
            rows.append(Phase1Row(wt=wt, pos=pos, mut=mut, mean_ddg=mean, tier=tier))
    return rows


def _aa_class(aa: str) -> str:
    hydrophobic = set("AVLIMFWY")
    polar = set("STNQ")
    charged_pos = set("KRH")
    charged_neg = set("DE")
    special = set("CGP")

    if aa in hydrophobic:
        return "hydrophobic"
    if aa in polar:
        return "polar"
    if aa in charged_pos:
        return "pos"
    if aa in charged_neg:
        return "neg"
    if aa in special:
        return "special"
    return "other"


def _unique_keep_order(xs: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def propose_alternatives(
    wt: str,
    pos: int,
    observed_best_mut: str,
    max_per_pos: int,
) -> List[str]:
    """
    Return a list of mutant amino acids to try at (wt,pos).

    We keep this conservative and human-readable:
    - Prefer "nearby" substitutions (same broad class) unless literature suggests otherwise.
    - Include literature-known substitutions at the SAME position (when applicable).
    """
    # Position-specific knowledge for IsPETase / 5XJH numbering (pose = PDB - 29 in this repo):
    # - PDB S238F is a canonical engineered mutation (Austin et al., 2018),
    #   which corresponds to pose 209 in this repo.
    # - PDB R224Q appears in FAST-PETase; corresponds to pose 195 here.
    literature_overrides: Dict[int, List[str]] = {
        209: ["F"],       # S238F (Austin et al., 2018)
        195: ["Q", "E"],  # R224Q / R224E are recurring engineered substitutions
    }

    muts: List[str] = []

    # Always include the best observed mutation as an optional "control" first.
    muts.append(observed_best_mut)

    # Add literature suggestions if we have them.
    muts.extend(literature_overrides.get(pos, []))

    # Generic class-based exploration.
    cls = _aa_class(observed_best_mut)  # explore around what worked best
    if cls == "hydrophobic":
        muts.extend(list("ILVFMYW"))  # modest set: aliphatic + aromatics
    elif cls == "polar":
        muts.extend(list("TSNQ"))
    elif cls == "pos":
        muts.extend(list("KRH"))
    elif cls == "neg":
        muts.extend(list("DE"))
    else:
        # fall back to a small diverse set
        muts.extend(list("AVLIMFWYSTNQKRHDE"))

    # Remove WT (no-op) and keep order.
    muts = [m for m in _unique_keep_order(muts) if m != wt]

    return muts[:max_per_pos]


def write_rosetta_mutfile(
    mutations: List[Tuple[str, int, str]],
    out_path: Path,
    chain: str,
    header_comment: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("# Phase-2 mutation list (auto-generated)\n")
        if header_comment:
            for line in header_comment.strip().splitlines():
                f.write(f"# {line.rstrip()}\n")
        f.write("# Format: total N; then for each set: '1' + '<WT> <POSE> <CHAIN> <MUT>'\n")
        f.write(f"total {len(mutations)}\n\n")
        for (wt, pos, mut) in mutations:
            f.write("1\n")
            f.write(f"{wt} {pos} {chain} {mut}\n\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="analysis/ddg/phase1.csv", help="Phase-1 ddG summary CSV")
    ap.add_argument("--out", default="configs/rosetta/mutlist_phase2.mut", help="Output .mut file path")
    ap.add_argument("--chain", default="A", help="Chain ID column to include (script will strip it when running ddG)")
    ap.add_argument("--tiers", default="A,B", help="Comma-separated tiers to include (e.g. A,B)")
    ap.add_argument(
        "--exclude-positions",
        default="131,177,208",
        help="Comma-separated POSE residue indices to exclude (default: catalytic triad for 5XJH in this repo)",
    )
    ap.add_argument("--max-per-position", type=int, default=4, help="Max mutants to propose per position")
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)
    tiers = {t.strip().upper() for t in args.tiers.split(",") if t.strip()}
    exclude = {int(x) for x in args.exclude_positions.split(",") if x.strip()}

    phase1 = _parse_phase1_csv(inp)
    phase1 = [r for r in phase1 if r.tier in tiers]

    # Group by position; choose the best-scoring (most negative mean_ddG) observed mutation per position.
    best_by_pos: Dict[int, Phase1Row] = {}
    for r in phase1:
        cur = best_by_pos.get(r.pos)
        if cur is None or r.mean_ddg < cur.mean_ddg:
            best_by_pos[r.pos] = r

    # Propose alternatives per selected position.
    proposed: List[Tuple[str, int, str]] = []
    skipped: List[str] = []

    for pos in sorted(best_by_pos.keys()):
        r = best_by_pos[pos]
        if pos in exclude:
            skipped.append(r.sig)
            continue
        muts = propose_alternatives(r.wt, r.pos, r.mut, max_per_pos=args.max_per_position)
        for m in muts:
            proposed.append((r.wt, r.pos, m))

    # De-dupe exact sets while preserving order.
    seen: Set[Tuple[str, int, str]] = set()
    deduped: List[Tuple[str, int, str]] = []
    for t in proposed:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)

    header = (
        f"Input: {inp}\n"
        f"Included tiers: {','.join(sorted(tiers))}\n"
        f"Excluded pose positions: {','.join(str(x) for x in sorted(exclude))}\n"
        f"Note: excluded defaults correspond to catalytic triad in this repo (pose 131/177/208)."
    )
    if skipped:
        header += "\nSkipped (excluded) best hits: " + ", ".join(skipped)

    write_rosetta_mutfile(deduped, outp, chain=args.chain, header_comment=header)
    print(f"Wrote {len(deduped)} mutation set(s) to {outp}")


if __name__ == "__main__":
    main()

