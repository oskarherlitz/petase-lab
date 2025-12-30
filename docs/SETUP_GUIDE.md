# Setup Guide: Where to Run Code and Environment Setup

## Where Do I Run the Code?

**Answer: In your terminal/command line (Terminal on Mac, Command Prompt/PowerShell on Windows, or any Linux terminal)**

All the scripts in this repository are designed to run from the command line. You'll navigate to the project directory and run commands there.

### Opening Terminal

**Mac/Linux:**
- Press `Cmd + Space` (Mac) or open Applications → Terminal
- Navigate to project: `cd ~/Desktop/petase-lab`

**Windows:**
- Open Command Prompt or PowerShell
- Navigate to project: `cd Desktop\petase-lab`

---

## What Environments Do I Need?

You need **two separate things**:

1. **Python Environment** (via Conda/Mamba) - for Python scripts
2. **Rosetta Software** (separate installation) - for protein design calculations

---

## Step 1: Install Conda/Mamba (If You Don't Have It)

Conda is a package manager that creates isolated Python environments. You need it to manage dependencies.

### Check if you have Conda:
```bash
conda --version
```

### If you don't have it:

**Option A: Install Miniconda (Recommended - smaller)**
1. Download from: https://docs.conda.io/en/latest/miniconda.html
2. Install for your operating system
3. Restart terminal

**Option B: Install Mamba (Faster alternative)**
```bash
# If you have conda, install mamba:
conda install mamba -n base -c conda-forge
```

---

## Step 2: Create Python Environment

The repository has separate environments for different tools. Start with the base environment:

```bash
# Navigate to project directory
cd ~/Desktop/petase-lab

# Create base environment (for Python scripts)
conda env create -f envs/base.yml

# Activate the environment
conda activate petase-lab
```

**What this installs:**
- Python 3.11
- pandas, numpy (data analysis)
- biopython (protein structure handling)
- rdkit (chemistry)
- jupyterlab (optional, for notebooks)

### Verify it worked:
```bash
python --version  # Should show Python 3.11
python -c "import pandas; print('✓ pandas works')"
```

---

## Step 3: Install Rosetta (Separate Installation)

**Important:** Rosetta is NOT installed via Conda. You need to:

1. **Get Rosetta License** (Academic/Commercial)
   - Academic: https://www.rosettacommons.org/software/license-and-download
   - You'll need to register and get a license

2. **Download and Install Rosetta**
   - Follow instructions from RosettaCommons
   - Typically involves downloading and compiling

3. **Set Environment Variable**
   ```bash
   # Find where you installed Rosetta, then:
   export ROSETTA_BIN=/path/to/rosetta/main/source/bin
   
   # To make it permanent, add to ~/.bashrc or ~/.zshrc:
   echo 'export ROSETTA_BIN=/path/to/rosetta/main/source/bin' >> ~/.zshrc
   ```

### Verify Rosetta:
```bash
$ROSETTA_BIN/relax.linuxgccrelease -version
```

**Note:** If you're on Mac, the binary might be `.macosclangrelease` instead of `.linuxgccrelease`. Check what's in your `$ROSETTA_BIN` directory.

---

## Step 4: Optional - Additional Environments

### For PyRosetta (Python interface to Rosetta):
```bash
conda env create -f envs/pyrosetta.yml
conda activate petase-pyrosetta
# Then install PyRosetta wheel manually
```

### For FoldX (if not already installed):
```bash
conda env create -f envs/foldx.yml
conda activate petase-foldx
# Install FoldX separately and add to PATH
```

---

## Quick Start: Run Your First Command

Once everything is set up:

```bash
# 1. Activate environment
conda activate petase-lab

# 2. Set Rosetta path (if not in .zshrc/.bashrc)
export ROSETTA_BIN=/path/to/rosetta/main/source/bin

# 3. Run setup script
bash scripts/setup_initial_data.sh

# 4. Run first Rosetta calculation
bash scripts/rosetta_relax.sh data/structures/5XJH/raw/PETase_raw.pdb
```

---

## Common Issues

### "Command not found: conda"
- Install Conda/Miniconda first
- Restart terminal after installation
- On Mac, you might need to run: `source ~/.zshrc` or `source ~/.bash_profile`

### "ROSETTA_BIN: unbound variable"
- You need to set the Rosetta path
- Run: `export ROSETTA_BIN=/path/to/rosetta/main/source/bin`
- Or add it to your shell config file

### "Permission denied" when running scripts
- Make scripts executable: `chmod +x scripts/*.sh`

### Wrong Rosetta binary name
- On Mac: might be `.macosclangrelease`
- On Linux: `.linuxgccrelease`
- Check what's in your `$ROSETTA_BIN` directory: `ls $ROSETTA_BIN/relax.*`

---

## Environment Summary

| Tool | Installation Method | Environment |
|------|---------------------|------------|
| Python scripts | Conda (`envs/base.yml`) | `petase-lab` |
| Rosetta | Separate download + license | System PATH |
| FoldX | Separate download | System PATH or `petase-foldx` |
| PyRosetta | Conda + manual wheel | `petase-pyrosetta` |

---

## Daily Workflow

```bash
# 1. Open terminal
# 2. Navigate to project
cd ~/Desktop/petase-lab

# 3. Activate environment
conda activate petase-lab

# 4. Set Rosetta (if needed)
export ROSETTA_BIN=/path/to/rosetta/main/source/bin

# 5. Run your commands
bash scripts/rosetta_relax.sh data/structures/5XJH/raw/PETase_raw.pdb
```

---

## Need Help?

- Check `docs/QUICKSTART.md` for step-by-step instructions
- Review `docs/RESEARCH_PLAN.md` for methodology
- See individual script files for command-line options

