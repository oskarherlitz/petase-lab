# Running ColabFold Detached on RunPod

## Quick Answer

**Yes!** RunPod pods keep running even if you disconnect. Use `tmux` or `screen` to keep your session alive.

---

## Option 1: Using tmux (Recommended)

### Start ColabFold in tmux:

```bash
# Install tmux (if not already installed)
apt-get update && apt-get install -y tmux

# Start new tmux session
tmux new -s colabfold

# Run ColabFold (inside tmux)
cd petase-lab
colabfold_batch \
  --num-recycle 3 \
  --num-models 5 \
  --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu

# Detach: Press Ctrl+B, then D
# (You can now close your computer/terminal)
```

### Reattach later:

```bash
# Reconnect to RunPod
# Then reattach to tmux session
tmux attach -t colabfold
```

### Or use the helper script:

```bash
cd petase-lab
bash scripts/run_colabfold_detached.sh
```

---

## Option 2: Using screen (Alternative)

```bash
# Install screen
apt-get update && apt-get install -y screen

# Start screen session
screen -S colabfold

# Run ColabFold (inside screen)
cd petase-lab
colabfold_batch \
  --num-recycle 3 \
  --num-models 5 \
  --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu

# Detach: Press Ctrl+A, then D
```

### Reattach:

```bash
screen -r colabfold
```

---

## Option 3: Using nohup (Simplest, but less interactive)

```bash
cd petase-lab

# Run with nohup (outputs to nohup.out)
nohup colabfold_batch \
  --num-recycle 3 \
  --num-models 5 \
  --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu > colabfold.log 2>&1 &

# Check if it's running
ps aux | grep colabfold

# View output
tail -f colabfold.log
```

---

## Recommended: tmux Workflow

### Step 1: Start tmux session

```bash
tmux new -s colabfold
```

### Step 2: Run ColabFold

```bash
cd petase-lab
colabfold_batch \
  --num-recycle 3 \
  --num-models 5 \
  --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu
```

### Step 3: Detach

- Press: `Ctrl+B`, then `D`
- Or type: `tmux detach`
- You can now close your computer/terminal

### Step 4: Reattach later

```bash
# Reconnect to RunPod
# Then:
tmux attach -t colabfold
```

---

## Useful tmux Commands

| Action | Command |
|--------|---------|
| **Detach** | `Ctrl+B`, then `D` |
| **List sessions** | `tmux ls` |
| **Attach** | `tmux attach -t colabfold` |
| **Kill session** | `tmux kill-session -t colabfold` |
| **Scroll up** | `Ctrl+B`, then `[` (use arrow keys, press `q` to exit) |
| **Split window** | `Ctrl+B`, then `%` (vertical) or `"` (horizontal) |

---

## Monitoring Progress While Detached

### Option 1: Reattach to tmux

```bash
tmux attach -t colabfold
# See live output
```

### Option 2: Check log file

```bash
# If you redirected output
tail -f colabfold.log

# Or check ColabFold's log
tail -f runs/colabfold_predictions_gpu/log.txt
```

### Option 3: Check for completed files

```bash
# Count PDB files
ls -1 runs/colabfold_predictions_gpu/*.pdb 2>/dev/null | wc -l

# List recent files
ls -lht runs/colabfold_predictions_gpu/*.pdb | head -5
```

---

## Complete Example

```bash
# 1. On RunPod, start tmux
tmux new -s colabfold

# 2. Navigate and run
cd petase-lab
colabfold_batch \
  --num-recycle 3 \
  --num-models 5 \
  --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu

# 3. Detach: Ctrl+B, D
# (Close your computer - RunPod keeps running!)

# 4. Later, reconnect and check progress
tmux attach -t colabfold
# Or check files:
ls -lh runs/colabfold_predictions_gpu/*.pdb
```

---

## Important Notes

### RunPod Pod Behavior:

- ✅ **Pod keeps running** even if you disconnect
- ✅ **Processes continue** in background
- ⚠️ **But terminal sessions die** when you disconnect (unless using tmux/screen)
- ✅ **Files persist** on pod storage or network volume

### Best Practice:

**Use tmux** - it's the best balance of:
- Easy to use
- Can monitor progress
- Can reattach anytime
- Works reliably

---

## Troubleshooting

### "tmux: command not found"
```bash
apt-get update && apt-get install -y tmux
```

### "Session already exists"
```bash
# Attach to existing session
tmux attach -t colabfold

# Or kill old session and start new
tmux kill-session -t colabfold
tmux new -s colabfold
```

### "Can't find session"
```bash
# List all sessions
tmux ls

# Attach to the correct one
tmux attach -t <session-name>
```

---

## Summary

**Yes, you can turn off your computer!**

1. **Start tmux:** `tmux new -s colabfold`
2. **Run ColabFold** inside tmux
3. **Detach:** `Ctrl+B`, then `D`
4. **Turn off computer** - RunPod keeps running!
5. **Reattach later:** `tmux attach -t colabfold`

**The pod will keep running for hours/days until you stop it or it auto-stops.**

