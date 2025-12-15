# Understanding Rosetta Output: What Do All These Numbers Mean?

## Quick Answer

The numbers show Rosetta's relaxation process in action:
- **Negative scores** = Energy (lower = better structure)
- **Small positive numbers** = RMSD (how much structure changed)
- **Decimals (0.022, 0.55, etc.)** = Weights/constraints (control relaxation)

---

## Breaking Down a Typical Line

```
protocols.relax.FastRelax: CMD: min  -882.218  0.814959  0.814959  0.55
```

### Parts:
1. **`CMD: min`** = Command: minimize (optimize structure)
2. **`-882.218`** = Total energy score (lower = better)
3. **`0.814959`** = RMSD from starting structure (how much it moved)
4. **`0.814959`** = RMSD from best structure so far
5. **`0.55`** = Constraint weight (how tightly to hold structure)

---

## The Numbers Explained

### 1. Energy Scores (Negative Numbers)

**Example:** `-882.218`, `-1338.02`, `-1079.1`

**What it is:**
- Rosetta's energy function score
- Measures how "good" the structure is
- **Lower (more negative) = Better structure**

**Why negative?**
- Energy is measured relative to a reference
- Negative = favorable interactions
- Positive = unfavorable (clashes, bad geometry)

**What's a good score?**
- Depends on protein size
- For PETase (~290 residues): -800 to -1000 is typical
- You want the **lowest** (most negative) score

---

### 2. RMSD (Root Mean Square Deviation)

**Example:** `0.814959`, `1.5339`, `0.022`

**What it is:**
- How much the structure has moved (in Angstroms)
- Measures distance atoms moved from starting position
- **Lower = Less movement**

**Two RMSD values:**
- First: RMSD from **original** structure
- Second: RMSD from **best** structure found so far

**What's normal?**
- 0.5-2.0 Å = Typical relaxation movement
- < 0.5 Å = Very small changes
- > 3.0 Å = Large changes (might indicate problems)

---

### 3. Constraint Weights

**Example:** `0.55`, `0.022`, `0.154`, `0.31955`

**What it is:**
- How tightly Rosetta holds the structure
- Controls how much the structure can change
- **Higher = More constrained** (less movement)

**Why they change:**
- FastRelax uses **ramped constraints**
- Starts loose (0.022) → gets tighter (0.55)
- Allows gradual optimization

**The ramping:**
```
0.022 → 0.02805 → 0.14575 → 0.154 → 0.30745 → 0.31955 → 0.55
(loose)                                                    (tight)
```

---

## Understanding the Commands

### `CMD: min` (Minimize)
- Energy minimization
- Optimizes atom positions
- Finds lowest energy geometry

### `CMD: repack` (Repack)
- Optimizes side chain positions
- Tries different rotamers (side chain conformations)
- Finds best side chain arrangements

### `CMD: scale:fa_rep` (Scale Repulsive)
- Adjusts repulsive energy term
- Controls how much atoms can overlap
- Part of ramped relaxation

### `CMD: coord_cst_weight` (Coordinate Constraint Weight)
- Sets how tightly to hold backbone
- Prevents structure from moving too much
- Gradually relaxed during optimization

### `CMD: accept_to_best`
- Accepts current structure as new best
- Happens when score improves

### `CMD: endrepeat`
- End of one relaxation cycle
- FastRelax does multiple cycles

### `MRP: 1` (Move Rejected/Passed)
- Shows which cycle number
- Tracks progress through relaxation

---

## Reading the Full Output

### Example Sequence:

```
protocols.relax.FastRelax: CMD: min  -882.218  0.814959  0.814959  0.55
```
**Translation:** Minimize structure. Energy is -882.218. Structure moved 0.81 Å from start and from best.

```
protocols.relax.FastRelax: MRP: 1  -882.218  -882.218  0.814959  0.814959
```
**Translation:** Cycle 1. Current energy -882.218. Best energy -882.218 (same). RMSD 0.81 Å.

```
protocols.relax.FastRelax: CMD: accept_to_best  -882.218  0.814959  0.814959  0.55
```
**Translation:** Accept this as the new best structure.

```
protocols.relax.FastRelax: CMD: scale:fa_rep  -1060.18  0.814959  0.814959  0.022
```
**Translation:** Adjust repulsive energy. New energy -1060.18 (better!). Constraint weight now 0.022 (looser).

```
protocols.relax.FastRelax: CMD: repack  -1079.1  0.814959  0.814959  0.022
```
**Translation:** Repack side chains. Energy improved to -1079.1.

---

## What to Look For

### ✅ Good Signs:
- **Energy decreasing** (becoming more negative): `-800 → -900 → -1000`
- **RMSD reasonable** (0.5-2.0 Å): Structure isn't moving too much
- **Energy stabilizing**: Scores converging to similar values
- **`accept_to_best` appearing**: Structure is improving

### ⚠️ Warning Signs:
- **Energy increasing** (becoming less negative): `-1000 → -800 → -600`
- **RMSD very high** (> 3.0 Å): Structure moving too much
- **Energy not converging**: Scores jumping around wildly

---

## The FastRelax Protocol

FastRelax does this sequence multiple times:

1. **Start loose** (constraint weight 0.022)
   - Allow structure to move
   - Find better conformations

2. **Gradually tighten** (0.028 → 0.15 → 0.31 → 0.55)
   - Refine the structure
   - Optimize details

3. **Repeat cycles**
   - Multiple rounds of optimization
   - Each cycle improves the structure

4. **Finish tight** (constraint weight 0.55)
   - Final refinement
   - Best structure

---

## Your Specific Output

Looking at your output:

```
CMD: min  -882.218  0.814959  0.814959  0.55
```
- Energy: **-882.218** (good for PETase)
- RMSD: **0.81 Å** (reasonable movement)
- Constraint: **0.55** (tight, final refinement)

```
CMD: scale:fa_rep  -1060.18  0.814959  0.814959  0.022
```
- Energy improved to **-1060.18** (better!)
- Constraint relaxed to **0.022** (starting new cycle)

```
CMD: repack  -1079.1  0.814959  0.814959  0.022
```
- Energy improved further to **-1079.1** (even better!)
- Side chains optimized

**This is all normal and good!** Your relaxation is working correctly.

---

## Key Takeaways

1. **Energy (negative numbers):**
   - Lower (more negative) = Better
   - Should decrease during relaxation

2. **RMSD (small positive numbers):**
   - How much structure moved
   - 0.5-2.0 Å is normal

3. **Constraint weights (0.022-0.55):**
   - Control how much structure can change
   - Ramp from loose to tight

4. **Commands:**
   - `min` = Minimize energy
   - `repack` = Optimize side chains
   - `accept_to_best` = Structure improved

---

## Final Score File

At the end, you get `score.sc` with the final scores:

```
SCORE: total_score cart_bonded dslf_fa13 fa_atr fa_dun fa_elec ...
SCORE: -888.500 150.752 -2.341 -1672.549 226.654 -511.450 ...
```

**Key columns:**
- `total_score`: Overall energy (-888.500)
- `fa_atr`: Attractive interactions (-1672.549)
- `fa_rep`: Repulsive interactions (182.383)
- `fa_sol`: Solvation energy (988.619)
- `hbond_*`: Hydrogen bonds (-93.072, etc.)

**Lower total_score = Better structure!**

---

## Summary

**The numbers show:**
- How good your structure is (energy)
- How much it changed (RMSD)
- How tightly it's constrained (weights)

**You want:**
- Low (negative) energy
- Reasonable RMSD (0.5-2.0 Å)
- Energy decreasing over time

**Your output looks good!** The relaxation is working correctly.

