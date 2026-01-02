#!/usr/bin/env bash
# ColabFold Setup Script
#
# This script helps you set up ColabFold for local batch processing.
# You can also use ColabFold via the web interface (no setup needed).
#
# Usage:
#   bash scripts/setup_colabfold.sh [--method METHOD]
#
# Methods:
#   conda  - Install via conda (recommended if you use conda)
#   pip    - Install via pip (simpler, works with any Python)
#   web    - Just show web interface instructions (no installation)

set -euo pipefail

METHOD=${1:-ask}

echo "ColabFold Setup"
echo "==============="
echo ""

# Check if already installed
if command -v colabfold_batch &> /dev/null; then
    echo "✓ ColabFold is already installed!"
    colabfold_batch --version 2>&1 || echo "  (version check failed, but command exists)"
    echo ""
    echo "You can now use:"
    echo "  bash scripts/colabfold_predict.sh <fasta_file>"
    exit 0
fi

if python -c "import colabfold" 2>/dev/null; then
    echo "✓ ColabFold Python package is installed!"
    echo ""
    echo "You can now use:"
    echo "  bash scripts/colabfold_predict.sh <fasta_file>"
    exit 0
fi

echo "ColabFold is not currently installed."
echo ""
echo "You have 3 options:"
echo ""
echo "1. WEB INTERFACE (Easiest - No installation needed!)"
echo "   → Go to https://colabfold.com"
echo "   → Upload your FASTA file"
echo "   → Download results"
echo "   → Works immediately, no setup required"
echo ""
echo "2. LOCAL INSTALLATION (For batch processing)"
echo "   → Install ColabFold on your machine"
echo "   → Can process multiple sequences"
echo "   → Requires ~10GB disk space for databases"
echo ""
echo "3. SKIP SETUP (Use web interface)"
echo "   → No installation needed"
echo "   → Just use the web interface when needed"
echo ""

if [ "$METHOD" == "ask" ]; then
    read -p "Choose option (1=web, 2=local, 3=skip): " choice
    case $choice in
        1|web)
            echo ""
            echo "Great! No installation needed."
            echo ""
            echo "To use ColabFold:"
            echo "  1. Go to: https://colabfold.com"
            echo "  2. Upload your FASTA file (e.g., from Progen2 pipeline)"
            echo "  3. Click 'Search'"
            echo "  4. Wait 5-30 minutes"
            echo "  5. Download results"
            echo ""
            echo "Your script will also guide you if ColabFold isn't installed:"
            echo "  bash scripts/colabfold_predict.sh <fasta_file>"
            exit 0
            ;;
        2|local)
            METHOD="pip"  # Default to pip
            ;;
        3|skip)
            echo ""
            echo "Skipping installation. You can always run this script again later."
            exit 0
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
fi

# Local installation
if [ "$METHOD" == "conda" ] || [ "$METHOD" == "pip" ]; then
    echo ""
    echo "Setting up ColabFold locally..."
    echo ""
    
    # Check for conda
    if [ "$METHOD" == "conda" ] && command -v conda &> /dev/null; then
        echo "Using conda installation..."
        echo ""
        
        # Check if environment exists
        if conda env list | grep -q "petase-colabfold"; then
            echo "Environment 'petase-colabfold' already exists."
            read -p "Activate and install ColabFold? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                conda activate petase-colabfold || {
                    echo "Activating environment..."
                    source "$(conda info --base)/etc/profile.d/conda.sh"
                    conda activate petase-colabfold
                }
                pip install colabfold
            fi
        else
            echo "Creating conda environment from envs/colabfold.yml..."
            conda env create -f envs/colabfold.yml
            echo ""
            echo "✓ Environment created!"
            echo ""
            echo "To use ColabFold:"
            echo "  1. Activate environment: conda activate petase-colabfold"
            echo "  2. Run predictions: bash scripts/colabfold_predict.sh <fasta_file>"
            echo ""
            echo "Note: First run will download databases (~10GB, takes time)"
        fi
        
    elif [ "$METHOD" == "pip" ]; then
        echo "Using pip installation..."
        echo ""
        
        # Check Python
        if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
            echo "Error: Python not found. Please install Python first."
            exit 1
        fi
        
        PYTHON_CMD="python3"
        if ! command -v python3 &> /dev/null; then
            PYTHON_CMD="python"
        fi
        
        echo "Installing ColabFold via pip..."
        echo "This may take a few minutes..."
        echo ""
        
        $PYTHON_CMD -m pip install --upgrade pip
        $PYTHON_CMD -m pip install colabfold
        
        echo ""
        echo "✓ ColabFold installed!"
        echo ""
        echo "To use ColabFold:"
        echo "  bash scripts/colabfold_predict.sh <fasta_file>"
        echo ""
        echo "Note: First run will download databases (~10GB, takes time)"
        
    else
        echo "Error: Method '$METHOD' not available."
        if [ "$METHOD" == "conda" ]; then
            echo "Conda not found. Try: bash scripts/setup_colabfold.sh pip"
        fi
        exit 1
    fi
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Test with a small FASTA file:"
echo "     bash scripts/colabfold_predict.sh <your_fasta_file>"
echo ""
echo "  2. For Progen2 sequences:"
echo "     bash scripts/colabfold_predict.sh runs/run_*/candidates/candidates.ranked.fasta"
echo ""

