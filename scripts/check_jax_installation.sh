#!/usr/bin/env bash
# Check if JAX was installed with CUDA support

echo "Checking JAX installation..."
echo ""

# Check pip packages
echo "1. Installed JAX packages:"
pip show jax jaxlib | grep -E "Name|Version|Location"

echo ""
echo "2. Checking jaxlib for CUDA support..."
python3 << 'CHECK_JAXLIB'
import jaxlib
import os

print(f"JAXlib version: {jaxlib.__version__}")
print(f"JAXlib location: {jaxlib.__file__}")

# Check if CUDA libraries are in jaxlib
jaxlib_path = os.path.dirname(jaxlib.__file__)
print(f"\nJAXlib directory: {jaxlib_path}")

# Check for CUDA-related files
import glob
cuda_files = glob.glob(os.path.join(jaxlib_path, "**", "*cuda*"), recursive=True)
if cuda_files:
    print(f"\nFound CUDA files in jaxlib:")
    for f in cuda_files[:5]:
        print(f"  {f}")
else:
    print("\n⚠ No CUDA files found in jaxlib - this is CPU-only!")

# Check what backends are available
try:
    from jaxlib import xla_extension
    print(f"\nXLA extension available: {xla_extension}")
except Exception as e:
    print(f"\nError importing xla_extension: {e}")
CHECK_JAXLIB

echo ""
echo "3. Checking if CUDA-enabled jaxlib is available..."
pip list | grep -i jax

echo ""
echo "4. If jaxlib doesn't have CUDA, we need to reinstall..."
echo "   The issue is likely that jaxlib was installed without CUDA support."

