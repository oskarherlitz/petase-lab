#!/bin/bash
# Clean up disk space on RunPod

echo "Checking disk space..."
df -h

echo ""
echo "Cleaning up..."

# Clean pip cache
echo "1. Cleaning pip cache..."
pip cache purge 2>/dev/null || rm -rf ~/.cache/pip/*

# Clean conda cache
echo "2. Cleaning conda cache..."
if command -v conda &> /dev/null; then
    conda clean -a -y 2>/dev/null || rm -rf ~/miniconda3/pkgs/cache/*
fi

# Clean apt cache
echo "3. Cleaning apt cache..."
apt-get clean 2>/dev/null || true
rm -rf /var/lib/apt/lists/* 2>/dev/null || true

# Clean temporary files
echo "4. Cleaning temp files..."
rm -rf /tmp/* 2>/dev/null || true
rm -rf /var/tmp/* 2>/dev/null || true

# Check largest directories
echo ""
echo "5. Largest directories:"
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -10 || \
du -h --max-depth=1 ~ 2>/dev/null | sort -hr | head -10

echo ""
echo "Disk space after cleanup:"
df -h

