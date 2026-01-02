#!/usr/bin/env bash

# MNIST Benchmark Script for Linux/macOS
# Runs mnist_handwritten_digits and collects system information

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Output file
BENCHMARK_FILE="BENCHMARKS.md"
TEMP_OUTPUT=$(mktemp)

echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         MNIST Benchmark Collection Script            ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS"
    elif [[ "$OSTYPE" == "freebsd"* ]]; then
        echo "FreeBSD"
    else
        echo "Unknown"
    fi
}

# Function to get CPU info
get_cpu_info() {
    local os=$(detect_os)

    if [[ "$os" == "Linux" ]]; then
        # Linux CPU detection
        grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs
    elif [[ "$os" == "macOS" ]]; then
        # macOS CPU detection
        sysctl -n machdep.cpu.brand_string
    else
        echo "Unknown CPU"
    fi
}

# Function to get CPU core count
get_cpu_cores() {
    local os=$(detect_os)

    if [[ "$os" == "Linux" ]]; then
        grep -c "^processor" /proc/cpuinfo
    elif [[ "$os" == "macOS" ]]; then
        sysctl -n hw.ncpu
    else
        echo "?"
    fi
}

# Function to get RAM info
get_ram_info() {
    local os=$(detect_os)

    if [[ "$os" == "Linux" ]]; then
        # Linux RAM detection (in GB)
        local ram_kb=$(grep "MemTotal" /proc/meminfo | awk '{print $2}')
        local ram_gb=$(echo "scale=1; $ram_kb / 1024 / 1024" | bc)
        echo "${ram_gb} GB"
    elif [[ "$os" == "macOS" ]]; then
        # macOS RAM detection (in GB)
        local ram_bytes=$(sysctl -n hw.memsize)
        local ram_gb=$(echo "scale=1; $ram_bytes / 1024 / 1024 / 1024" | bc)
        echo "${ram_gb} GB"
    else
        echo "Unknown"
    fi
}

# Function to get OS version
get_os_version() {
    local os=$(detect_os)

    if [[ "$os" == "Linux" ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            echo "${NAME} ${VERSION_ID}"
        else
            echo "Linux (unknown distro)"
        fi
    elif [[ "$os" == "macOS" ]]; then
        echo "macOS $(sw_vers -productVersion)"
    else
        echo "Unknown"
    fi
}

# Function to get architecture
get_architecture() {
    uname -m
}

# Collect system information
echo -e "${BLUE}[1/4] Collecting system information...${NC}"
OS=$(detect_os)
OS_VERSION=$(get_os_version)
ARCH=$(get_architecture)
CPU=$(get_cpu_info)
CPU_CORES=$(get_cpu_cores)
RAM=$(get_ram_info)
TIMESTAMP=$(date -u +"%Y-%m-%d")

echo -e "  ${GREEN}✓${NC} OS: ${OS_VERSION}"
echo -e "  ${GREEN}✓${NC} Architecture: ${ARCH}"
echo -e "  ${GREEN}✓${NC} CPU: ${CPU} (${CPU_CORES} cores)"
echo -e "  ${GREEN}✓${NC} RAM: ${RAM}"
echo

# Check if Rust nightly is installed
echo -e "${BLUE}[2/4] Checking Rust installation...${NC}"
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Error: cargo not found. Please install Rust.${NC}"
    exit 1
fi

RUST_VERSION=$(rustc --version | cut -d' ' -f2)
echo -e "  ${GREEN}✓${NC} Rust version: ${RUST_VERSION}"
echo

# Run the benchmark
echo -e "${BLUE}[3/4] Running MNIST benchmark...${NC}"
echo -e "  ${YELLOW}This may take several minutes...${NC}"
echo

# Run cargo and capture output
if cargo run --release --example mnist_digits 2>&1 | tee "$TEMP_OUTPUT"; then
    echo
    echo -e "  ${GREEN}✓${NC} Benchmark completed successfully"
else
    echo -e "${RED}Error: Benchmark failed${NC}"
    rm -f "$TEMP_OUTPUT"
    exit 1
fi
echo

# Parse results
echo -e "${BLUE}[4/4] Parsing results and updating ${BENCHMARK_FILE}...${NC}"

# Extract metrics from output
TOTAL_TIME=$(grep "Total time to run:" "$TEMP_OUTPUT" | tail -1 | sed -E 's/.*Total time to run: ([0-9.]+)/\1/')
FINAL_ACCURACY=$(grep "Accuracy:" "$TEMP_OUTPUT" | tail -1 | grep -oP '\d+\.\d+(?=%)')

if [ -z "$TOTAL_TIME" ]; then
    echo -e "${YELLOW}Warning: Could not parse total time${NC}"
    TOTAL_TIME="N/A"
fi

if [ -z "$FINAL_ACCURACY" ]; then
    echo -e "${YELLOW}Warning: Could not parse accuracy${NC}"
    FINAL_ACCURACY="N/A"
fi

echo -e "  ${GREEN}✓${NC} Total time: ${TOTAL_TIME}s"
echo -e "  ${GREEN}✓${NC} Final accuracy: ${FINAL_ACCURACY}%"
echo

# Initialize benchmarks file if it doesn't exist
if [ ! -f "$BENCHMARK_FILE" ]; then
    cat > "$BENCHMARK_FILE" << 'EOF'
# 🚀 MNIST Training Benchmarks

This file contains real-world benchmark results from training the MNIST handwritten digits neural network across different hardware configurations.

## 📊 Benchmark Results

Each entry represents a complete training run (10 epochs, 60,000 samples, batch size 2000) using the `mnist_digits` example.

### How to Add Your Benchmark

Run the benchmark script for your platform:

**Linux/macOS:**
```bash
chmod +x benchmark.sh
./benchmark.sh
```

**Windows PowerShell:**
```powershell
.\benchmark.ps1
```

The script will automatically:
- ✅ Collect your system information
- ✅ Run the MNIST training example
- ✅ Parse the results
- ✅ Append your results to this file

---

## 📈 Results Table

| Date | OS | Architecture | CPU | Cores | RAM (GB) | Rust Version | Time (s) | Accuracy (%) |
|------|----|--------------|----|-------|----------|--------------|----------|--------------|
EOF
    echo -e "  ${GREEN}✓${NC} Created new ${BENCHMARK_FILE}"
fi

# Append benchmark entry as table row
echo "| ${TIMESTAMP} | ${OS_VERSION} | ${ARCH} | ${CPU} | ${CPU_CORES} | ${RAM} | ${RUST_VERSION} | ${TOTAL_TIME} | ${FINAL_ACCURACY} |" >> "$BENCHMARK_FILE"

echo -e "  ${GREEN}✓${NC} Results appended to ${BENCHMARK_FILE}"

# Cleanup
rm -f "$TEMP_OUTPUT"

echo
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║            Benchmark completed successfully!         ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo
echo -e "View results: ${CYAN}cat ${BENCHMARK_FILE}${NC}"
echo
