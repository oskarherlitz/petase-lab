#!/bin/bash
# Remove RFdiffusion outputs from git tracking (but keep files locally)
# Run this if RFdiffusion files were already committed before adding to .gitignore

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================="
echo "Removing RFdiffusion Outputs from Git"
echo "=========================================="
echo ""
echo "This will remove RFdiffusion files from git tracking"
echo "but keep them on your local filesystem."
echo ""

# Find all RFdiffusion run directories
RFDIFFUSION_DIRS=$(find runs -type d -name "*rfdiffusion*" 2>/dev/null || true)

if [ -z "${RFDIFFUSION_DIRS}" ]; then
    echo "No RFdiffusion directories found in runs/"
    exit 0
fi

echo "Found RFdiffusion directories:"
echo "${RFDIFFUSION_DIRS}" | while read dir; do
    echo "  - ${dir}"
done
echo ""

read -p "Remove these from git tracking? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Removing from git tracking..."

# Remove each directory from git tracking
echo "${RFDIFFUSION_DIRS}" | while read dir; do
    if git ls-files --error-unmatch "${dir}"/*.pdb "${dir}"/*.trb "${dir}"/run_inference.log 2>/dev/null | grep -q .; then
        echo "  Removing ${dir}/*.pdb, *.trb, run_inference.log"
        git rm --cached "${dir}"/*.pdb "${dir}"/*.trb "${dir}"/run_inference.log 2>/dev/null || true
    fi
    
    # Remove trajectory and schedule folders if tracked
    if [ -d "${dir}/traj" ] && git ls-files --error-unmatch "${dir}/traj" 2>/dev/null | grep -q .; then
        echo "  Removing ${dir}/traj/"
        git rm --cached -r "${dir}/traj/" 2>/dev/null || true
    fi
    
    if [ -d "${dir}/schedules" ] && git ls-files --error-unmatch "${dir}/schedules" 2>/dev/null | grep -q .; then
        echo "  Removing ${dir}/schedules/"
        git rm --cached -r "${dir}/schedules/" 2>/dev/null || true
    fi
done

echo ""
echo "=========================================="
echo "Removal Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review changes: git status"
echo "  2. Commit the removal:"
echo "     git commit -m 'Remove RFdiffusion outputs from git tracking'"
echo "  3. Push to remote:"
echo "     git push"
echo ""
echo "Files are still on your local filesystem, just no longer tracked by git."
echo ""

