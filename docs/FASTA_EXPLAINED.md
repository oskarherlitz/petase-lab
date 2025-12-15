# What is FASTA?

## Quick Answer

**FASTA** is a simple text file format used to store biological sequences (DNA, RNA, or protein). It's the standard way to represent amino acid sequences for structure prediction tools like ColabFold and AlphaFold.

---

## Format Structure

A FASTA file has two parts per sequence:

1. **Header line** - Starts with `>` followed by a description
2. **Sequence line(s)** - The actual amino acid sequence (one-letter codes)

### Example:

```
>PETase_wildtype
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGLAQAGVLEKHHDRVFFKDLDNEERAKAAQAS
```

### Breakdown:

- `>PETase_wildtype` = Header (description/name)
- `MKTAYIAKQRQ...` = Sequence (amino acids in one-letter code)

---

## Amino Acid One-Letter Codes

| Letter | Amino Acid | Full Name |
|--------|------------|-----------|
| A | Alanine | Ala |
| R | Arginine | Arg |
| N | Asparagine | Asn |
| D | Aspartic acid | Asp |
| C | Cysteine | Cys |
| Q | Glutamine | Gln |
| E | Glutamic acid | Glu |
| G | Glycine | Gly |
| H | Histidine | His |
| I | Isoleucine | Ile |
| L | Leucine | Leu |
| K | Lysine | Lys |
| M | Methionine | Met |
| F | Phenylalanine | Phe |
| P | Proline | Pro |
| S | Serine | Ser |
| T | Threonine | Thr |
| W | Tryptophan | Trp |
| Y | Tyrosine | Tyr |
| V | Valine | Val |

**Special codes:**
- `X` = Unknown amino acid
- `-` = Gap (in alignments)
- `*` = Stop codon

---

## Why FASTA is Used

### 1. **Simple and Universal**
- Plain text (readable by humans and computers)
- Works on any operating system
- No special software needed to view/edit

### 2. **Standard Format**
- Accepted by virtually all bioinformatics tools
- ColabFold, AlphaFold, BLAST, etc. all use FASTA
- Easy to convert between formats

### 3. **Sequence-Only**
- Just the amino acid sequence (no 3D coordinates)
- Much smaller than PDB files
- Perfect for structure prediction (predicts 3D from sequence)

---

## FASTA vs PDB

| Feature | FASTA | PDB |
|---------|-------|-----|
| Contains | Sequence only | 3D coordinates |
| Size | Small (~1 KB) | Large (~100 KB+) |
| Use | Structure prediction | Structure analysis |
| Format | Text | Text (structured) |
| Example | `MKTAYIAK...` | `ATOM 1 N MET A 1 ...` |

**Key difference:**
- **FASTA** = "What is the sequence?" (1D)
- **PDB** = "Where are the atoms?" (3D)

---

## Multiple Sequences in One File

You can have multiple sequences in one FASTA file:

```
>PETase_wildtype
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGLAQAGVLEKHHDRVFFKDLDNEERAKAAQAS

>PETase_mutant_S160A
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGLAQAGVLEKHHDRVFFKDLDNEERAKAAQAS
```

Notice: The second sequence has Serine (S) at position 160 changed to Alanine (A).

---

## In Your PETase Project

### When You Need FASTA:

1. **ColabFold/AlphaFold Prediction**
   - These tools predict structure from sequence
   - Input: FASTA file
   - Output: PDB file (3D structure)

2. **Sequence Analysis**
   - Compare sequences
   - Find mutations
   - Check sequence identity

3. **Database Searches**
   - BLAST (find similar proteins)
   - Pfam (find protein families)

### Converting PDB → FASTA:

Use the script I created:
```bash
python scripts/pdb_to_fasta.py design_001.pdb
```

This extracts the sequence from your PDB file and creates a FASTA file.

---

## Example Workflow

```
1. Rosetta Design → design_001.pdb (3D structure)
2. Convert to FASTA → design_001.fasta (sequence)
3. ColabFold Prediction → predicted_structure.pdb (3D from sequence)
4. Compare → design_001.pdb vs predicted_structure.pdb
```

---

## Common FASTA File Extensions

- `.fasta` - Most common
- `.fa` - Short version
- `.fas` - Alternative
- `.faa` - Amino acid sequences
- `.fna` - Nucleic acid sequences

**They're all the same format!** The extension just helps identify the content type.

---

## Viewing/Editing FASTA Files

**Any text editor works:**
- TextEdit (Mac)
- Notepad (Windows)
- VS Code
- Terminal: `cat sequence.fasta`

**No special software needed!**

---

## Quick Reference

**FASTA = Simple text format for biological sequences**

- Header: `>description`
- Sequence: `MKTAYIAKQRQ...` (one-letter amino acid codes)
- Used by: ColabFold, AlphaFold, BLAST, most bioinformatics tools
- Purpose: Represent sequences (1D) before predicting structures (3D)

---

## See Also

- `scripts/pdb_to_fasta.py` - Convert PDB to FASTA
- `docs/COLABFOLD_GUIDE.md` - Using FASTA with ColabFold
- `docs/glossary.md` - More terminology

