#!/usr/bin/env python3
"""Summarize Rosetta ddG results across replicates.

This produces the Phase-1 style table described in docs/NEXT_STEPS_CURSOR.md:
- mutation
- mean_ddG
- std_ddG
- n_replicates
- tier (A/B/C)
- notes (blank)

Inputs:
- Either one or more Rosetta cartesian_ddg JSON files (e.g. runs/*ddg*/outputs/*.json)
- Or a CSV produced by scripts/parse_ddg.py (columns: run, mutation, ddg_reu)

Usage examples (zsh users: quote globs):
  python scripts/summarize_ddg.py 'runs/*ddg*/outputs/*.json' --out analysis/ddg/phase1
  python scripts/summarize_ddg.py results/ddg_scans/out.csv --out analysis/ddg/phase1

Outputs:
- <out>.csv
- <out>.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _avg(xs: List[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _std(xs: List[float]) -> float | None:
    if len(xs) < 2:
        return 0.0 if len(xs) == 1 else None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _tier(mean_ddg: float, a_threshold: float, c_threshold: float) -> str:
    # Negative ddG = stabilizing
    if mean_ddg <= a_threshold:
        return "A"
    if mean_ddg >= c_threshold:
        return "C"
    return "B"


def _expand_inputs(inputs: List[str]) -> List[str]:
    expanded: List[str] = []
    for inp in inputs:
        matches = glob.glob(inp)
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(inp)
    # de-dupe while preserving order
    seen = set()
    out = []
    for p in expanded:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


_AA3_TO_AA1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def _mutation_signature_from_json_mutations(mutations) -> str:
    """Build a stable signature like S160A or multi: A10B;C20D.

    Treat empty/no-op as WT.
    """
    if not isinstance(mutations, list) or not mutations:
        return "WT"

    pieces = []
    for m in mutations:
        if not isinstance(m, dict):
            continue
        wt = m.get("wt")
        mut = m.get("mut")
        pos = m.get("pos")
        if wt and mut and pos is not None:
            if wt == mut:
                continue
            pieces.append(f"{wt}{pos}{mut}")

    return ";".join(pieces) if pieces else "WT"


def _ddg_from_cartesian_ddg_json(j) -> List[Tuple[str, float]]:
    """Compute ddG from totals if explicit ddg fields are absent.

    ddg = avg(mut_total) - avg(wt_total)
    """
    if not isinstance(j, list):
        return []

    totals_by_sig: Dict[str, List[float]] = {}
    for item in j:
        if not isinstance(item, dict):
            continue
        scores = item.get("scores") or {}
        if not isinstance(scores, dict):
            continue
        total = scores.get("total")
        if total is None:
            continue
        try:
            total_f = float(total)
        except (ValueError, TypeError):
            continue

        sig = _mutation_signature_from_json_mutations(item.get("mutations"))
        totals_by_sig.setdefault(sig, []).append(total_f)

    wt_avg = _avg(totals_by_sig.get("WT", []))
    if wt_avg is None:
        return []

    out: List[Tuple[str, float]] = []
    for sig, totals in totals_by_sig.items():
        if sig == "WT":
            continue
        mut_avg = _avg(totals)
        if mut_avg is None:
            continue
        out.append((sig, mut_avg - wt_avg))

    return out


def _records_from_json(path: str) -> List[Tuple[str, float]]:
    """Return [(mutation_sig, ddg)] for a JSON file."""
    with open(path, "r") as f:
        content = f.read().strip()

    # Guard: someone might pass a CSV by mistake.
    if content.startswith("run,mutation"):
        return []

    j = json.loads(content)

    recs: List[Tuple[str, float]] = []

    # Schema 1: explicit ddg per mutation dict(s)
    if isinstance(j, list):
        for item in j:
            if not isinstance(item, dict):
                continue
            muts = item.get("mutations")
            if isinstance(muts, list):
                for m in muts:
                    if not isinstance(m, dict):
                        continue
                    name = m.get("name") or m.get("mutation") or m.get("mut")
                    ddg = m.get("ddg") or m.get("ddg_reu") or m.get("delta_delta_g")
                    if name and ddg is not None:
                        try:
                            recs.append((str(name), float(ddg)))
                        except (ValueError, TypeError):
                            pass

    elif isinstance(j, dict):
        muts = j.get("mutations")
        if isinstance(muts, dict):
            for name, info in muts.items():
                ddg = info.get("ddg") if isinstance(info, dict) else info
                if ddg is None:
                    continue
                try:
                    recs.append((str(name), float(ddg)))
                except (ValueError, TypeError):
                    pass

    # Schema 2: totals-only -> compute ddg
    if not recs:
        recs = _ddg_from_cartesian_ddg_json(j)

    # Drop WT/no-op
    recs = [(m, d) for (m, d) in recs if m and m != "WT"]
    return recs


def _records_from_parse_ddg_csv(path: str) -> List[Tuple[str, float]]:
    """Return [(mutation, ddg_reu)] from parse_ddg CSV."""
    out: List[Tuple[str, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        if "mutation" not in reader.fieldnames or "ddg_reu" not in reader.fieldnames:
            return []
        for row in reader:
            mut = (row.get("mutation") or "").strip()
            ddg_s = (row.get("ddg_reu") or "").strip()
            if not mut or not ddg_s:
                continue
            try:
                out.append((mut, float(ddg_s)))
            except ValueError:
                continue
    # Drop WT/no-op
    out = [(m, d) for (m, d) in out if m != "WT"]
    return out


@dataclass(frozen=True)
class SummaryRow:
    mutation: str
    mean_ddg: float
    std_ddg: float
    n: int
    tier: str
    notes: str = ""


def _write_csv(rows: List[SummaryRow], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mutation", "mean_ddG", "std_ddG", "n_replicates", "tier", "notes"])
        for r in rows:
            w.writerow([r.mutation, f"{r.mean_ddg:.4f}", f"{r.std_ddg:.4f}", r.n, r.tier, r.notes])


def _write_md(rows: List[SummaryRow], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("| mutation | mean_ddG | std_ddG | n_replicates | tier | notes |\n")
        f.write("|---|---:|---:|---:|:---:|---|\n")
        for r in rows:
            f.write(
                f"| {r.mutation} | {r.mean_ddg:.4f} | {r.std_ddg:.4f} | {r.n} | {r.tier} | {r.notes} |\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="JSON files/globs OR a parse_ddg CSV")
    ap.add_argument("--out", default="analysis/ddg/phase1", help="Output path stem (no extension)")
    ap.add_argument("--tier-a", type=float, default=-1.0, help="Tier A threshold (mean_ddG <= this)")
    ap.add_argument("--tier-c", type=float, default=1.0, help="Tier C threshold (mean_ddG >= this)")
    args = ap.parse_args()

    inps = _expand_inputs(args.inputs)

    # Collect replicate ddGs by mutation
    by_mut: Dict[str, List[float]] = {}

    for inp in inps:
        if inp.endswith(".csv"):
            recs = _records_from_parse_ddg_csv(inp)
        elif inp.endswith(".json"):
            recs = _records_from_json(inp)
        else:
            # Try JSON first, then CSV
            recs = []
            try:
                recs = _records_from_json(inp)
            except Exception:
                pass
            if not recs:
                try:
                    recs = _records_from_parse_ddg_csv(inp)
                except Exception:
                    pass

        for mut, ddg in recs:
            by_mut.setdefault(mut, []).append(ddg)

    if not by_mut:
        raise SystemExit(
            "No ddG records found. Provide one or more JSON files from a successful ddG run, "
            "or a CSV produced by scripts/parse_ddg.py."
        )

    rows: List[SummaryRow] = []
    for mut, ddgs in by_mut.items():
        m = _avg(ddgs)
        s = _std(ddgs)
        if m is None or s is None:
            continue
        rows.append(
            SummaryRow(
                mutation=mut,
                mean_ddg=m,
                std_ddg=s,
                n=len(ddgs),
                tier=_tier(m, args.tier_a, args.tier_c),
            )
        )

    rows.sort(key=lambda r: r.mean_ddg)  # most negative first

    out_stem = args.out
    _write_csv(rows, out_stem + ".csv")
    _write_md(rows, out_stem + ".md")

    print(f"Wrote {len(rows)} mutations to {out_stem}.csv and {out_stem}.md")


if __name__ == "__main__":
    main()
