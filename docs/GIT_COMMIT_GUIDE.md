# Git Commit Guide for External Directory

## Understanding Your Setup

Your `external/` directory contains **git submodules**:
- `external/progen2` → Points to `https://github.com/enijkamp/progen2.git`
- `external/rfdiffusion` → Points to `https://github.com/RosettaCommons/RFdiffusion.git`

## Scenario 1: Commit Changes to Submodule Itself

If you've modified files **inside** the submodule (e.g., `external/progen2/models/`):

### Option A: Commit to Original Repo (if you have access)

```bash
# Go into the submodule
cd external/progen2

# Add and commit changes
git add models/ requirements_macos.txt
git commit -m "Add models and macOS requirements"

# Push to original repo (if you have write access)
git push origin main

# Go back to main repo
cd ../..

# Update submodule reference
git add external/progen2
git commit -m "Update progen2 submodule"
git push
```

### Option B: Keep Changes Local (Don't Commit to Submodule)

If you don't have write access to the original repo, you can:

1. **Add submodule changes to main repo** (not recommended - mixes repos)
2. **Use .gitignore** to ignore the changes
3. **Fork the repo** and point submodule to your fork

## Scenario 2: Commit Submodule Reference Update

If you've updated the submodule to a different commit:

```bash
# Update submodule to latest version
cd external/progen2
git pull origin main
cd ../..

# Commit the submodule reference update
git add external/progen2
git commit -m "Update progen2 submodule to latest version"
git push
```

## Scenario 3: Commit Other Files in External (Not Submodules)

If you have files in `external/` that are **not** part of submodules:

```bash
# Add the files
git add external/your_file.py

# Commit
git commit -m "Add custom file to external"

# Push
git push
```

## Current Situation

Based on your git status, you have:
- **Untracked content** in `external/progen2` (models/, requirements_macos.txt)

### Recommended Approach

Since these are likely local additions (model files, macOS-specific requirements), you have two options:

#### Option 1: Ignore These Files (Recommended)

Add to `.gitignore`:

```bash
# Add to .gitignore
echo "external/progen2/models/" >> .gitignore
echo "external/progen2/requirements_macos.txt" >> .gitignore

# Commit the .gitignore update
git add .gitignore
git commit -m "Ignore local model files and macOS requirements in progen2 submodule"
git push
```

#### Option 2: Commit to Your Own Fork

1. Fork `https://github.com/enijkamp/progen2.git` to your GitHub
2. Update submodule to point to your fork
3. Commit changes in your fork
4. Update submodule reference in main repo

## Quick Commands

### Check What's Changed

```bash
# In main repo
git status

# In submodule
cd external/progen2
git status
```

### Commit Submodule Reference

```bash
git add external/progen2
git commit -m "Update progen2 submodule"
git push
```

### Update Submodule to Latest

```bash
git submodule update --remote external/progen2
git add external/progen2
git commit -m "Update progen2 submodule to latest"
git push
```

## Best Practices

1. **Don't commit large model files** - Use `.gitignore` or Git LFS
2. **Keep submodules clean** - Only commit if you have write access to original repo
3. **Use forks** - If you need to modify submodules, fork them first
4. **Document changes** - If you modify submodules, document why in your main repo

