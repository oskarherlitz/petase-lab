#!/usr/bin/env bash
# Helper script to find Rosetta installation

echo "Searching for Rosetta installation..."
echo ""

# Common locations
LOCATIONS=(
    "$HOME/rosetta"
    "$HOME/Rosetta"
    "$HOME/rosetta_src"
    "$HOME/Rosetta_src"
    "$HOME/Desktop/rosetta"
    "$HOME/Desktop/Rosetta"
    "/opt/rosetta"
    "/usr/local/rosetta"
)

for loc in "${LOCATIONS[@]}"; do
    if [ -d "$loc" ]; then
        echo "Found directory: $loc"
        if [ -d "$loc/main/source/bin" ]; then
            echo "  ✓ Contains main/source/bin"
            echo "  Setting ROSETTA_BIN=$loc/main/source/bin"
            export ROSETTA_BIN="$loc/main/source/bin"
            echo ""
            echo "To make this permanent, add to ~/.zshrc:"
            echo "export ROSETTA_BIN=\"$loc/main/source/bin\""
            exit 0
        fi
    fi
done

echo "Rosetta not found in common locations."
echo ""
echo "Please tell me where you installed Rosetta, or run:"
echo "  find ~ -type d -name 'rosetta' 2>/dev/null"
echo ""
echo "Then set ROSETTA_BIN manually:"
echo "  export ROSETTA_BIN=/path/to/rosetta/main/source/bin"

