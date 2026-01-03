# Using Virtual Environment on RunPod

## Should You Use a venv?

**Short answer: Yes, it's a good practice, but not strictly necessary on RunPod.**

### Pros of using venv:
- ✅ **Isolated dependencies** - No conflicts with system packages
- ✅ **Cleaner environment** - Easy to recreate
- ✅ **Better dependency management** - Can track exact versions
- ✅ **Easier to debug** - Can delete and recreate if broken

### Cons:
- ❌ **Slightly more setup** - Need to activate it each time
- ❌ **Takes extra disk space** - But minimal (~500MB)

### On RunPod specifically:
- **Less critical** - You're the only user, no system conflicts
- **But still helpful** - Makes dependency management cleaner
- **Recommended** - Especially if you're having dependency issues

---

## Quick Setup

```bash
cd /workspace/petase-lab

# Create and set up venv
bash scripts/setup_runpod_venv.sh

# Activate it
source venv_colabfold/bin/activate

# Now install packages (they go into venv)
# Run ColabFold
colabfold_batch ...
```

---

## Manual Setup

```bash
# 1. Create venv
python3 -m venv venv_colabfold

# 2. Activate it
source venv_colabfold/bin/activate

# 3. Install packages
pip install "colabfold[alphafold]" ...

# 4. Use it
colabfold_batch ...

# 5. Deactivate when done (optional)
deactivate
```

---

## Current Situation

**You're currently installing globally** (no venv), which is why you're getting conflicts.

**Using a venv would help**, but the **main issue is still the cuDNN error**, not the venv.

---

## Recommendation

**For now:** Focus on fixing the cuDNN error first.

**For future:** Use a venv for cleaner dependency management.

**If you want to try venv now:**
```bash
cd /workspace/petase-lab
bash scripts/setup_runpod_venv.sh
source venv_colabfold/bin/activate
# Then run ColabFold
```

---

## Summary

- **venv is good practice** but not required on RunPod
- **Current issues** are cuDNN-related, not venv-related
- **Using venv** would make dependency management cleaner
- **Your choice** - both approaches work on RunPod

