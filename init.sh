#!/bin/bash
# init.sh - Cross-platform initialization script for Linux/macOS
# Sets up the Rust AI Neural Network Library environment

set -e  # Exit on error

echo "=================================================="
echo "  Rust AI Neural Network Library - Init Script"
echo "=================================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Rust is installed
echo "Step 1/4: Checking Rust installation..."
if ! command -v rustc &> /dev/null; then
    echo -e "${RED}✗ Rust is not installed!${NC}"
    echo ""
    echo "Please install Rust by running:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo ""
    echo "Or visit: https://rustup.rs/"
    exit 1
fi

RUST_VERSION=$(rustc --version)
echo -e "${GREEN}✓ Rust is installed: ${RUST_VERSION}${NC}"

# Check for nightly toolchain (required for portable_simd)
echo ""
echo "Checking for nightly toolchain..."
if ! rustup toolchain list | grep -q "nightly"; then
    echo -e "${YELLOW}! Nightly toolchain not found. Installing...${NC}"
    rustup toolchain install nightly
    echo -e "${GREEN}✓ Nightly toolchain installed${NC}"
else
    echo -e "${GREEN}✓ Nightly toolchain is available${NC}"
fi

# Create training directory if it doesn't exist
echo ""
echo "Step 2/4: Setting up training directory..."
if [ ! -d "training" ]; then
    mkdir -p training
    echo -e "${GREEN}✓ Created training directory${NC}"
else
    echo -e "${GREEN}✓ Training directory already exists${NC}"
fi

# Extract MNIST datasets
echo ""
echo "Step 3/4: Extracting MNIST datasets..."

# Check if archive files exist
if [ ! -f "archive/mnist_train.csv.zip" ] || [ ! -f "archive/mnist_test.csv.zip" ]; then
    echo -e "${RED}✗ MNIST dataset archives not found in archive/ directory${NC}"
    echo "Please ensure the following files exist:"
    echo "  - archive/mnist_train.csv.zip"
    echo "  - archive/mnist_test.csv.zip"
    exit 1
fi

# Extract training data
if [ ! -f "training/mnist_train.csv" ]; then
    echo "Extracting mnist_train.csv..."
    unzip -q archive/mnist_train.csv.zip -d training/
    echo -e "${GREEN}✓ Extracted mnist_train.csv${NC}"
else
    echo -e "${GREEN}✓ mnist_train.csv already exists${NC}"
fi

# Extract test data
if [ ! -f "training/mnist_test.csv" ]; then
    echo "Extracting mnist_test.csv..."
    unzip -q archive/mnist_test.csv.zip -d training/
    echo -e "${GREEN}✓ Extracted mnist_test.csv${NC}"
else
    echo -e "${GREEN}✓ mnist_test.csv already exists${NC}"
fi

# Build the project
echo ""
echo "Step 4/4: Building the project..."
echo "This may take a few minutes on first run..."
echo ""

if cargo +nightly build; then
    echo ""
    echo -e "${GREEN}✓ Build successful!${NC}"
else
    echo ""
    echo -e "${RED}✗ Build failed!${NC}"
    echo "Please check the error messages above."
    exit 1
fi

# Success message
echo ""
echo "=================================================="
echo -e "${GREEN}  Setup completed successfully!${NC}"
echo "=================================================="
echo ""
echo "You can now run the examples:"
echo "  cargo +nightly run --release --example mnist_handwritten_digits"
echo "  cargo +nightly run --release --example mnist_conv_digits"
echo ""
echo "Or run tests:"
echo "  cargo +nightly test"
echo ""
echo "For more information, see README.md"
echo ""
