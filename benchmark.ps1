# MNIST Benchmark Script for Windows PowerShell
# Runs mnist_handwritten_digits and collects system information

$ErrorActionPreference = "Stop"

# Output file
$BENCHMARK_FILE = "BENCHMARKS.md"
$TEMP_OUTPUT = [System.IO.Path]::GetTempFileName()

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         MNIST Benchmark Collection Script            ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Function to get CPU info
function Get-CPUInfo {
    $cpu = Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1
    return $cpu.Name
}

# Function to get CPU core count
function Get-CPUCores {
    $cores = (Get-CimInstance -ClassName Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
    return $cores
}

# Function to get RAM info (in GB)
function Get-RAMInfo {
    $ram = Get-CimInstance -ClassName Win32_ComputerSystem
    $ramGB = [math]::Round($ram.TotalPhysicalMemory / 1GB, 1)
    return "$ramGB GB"
}

# Function to get OS version
function Get-OSVersion {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    return "$($os.Caption) (Build $($os.BuildNumber))"
}

# Function to get architecture
function Get-Architecture {
    return $env:PROCESSOR_ARCHITECTURE
}

# Collect system information
Write-Host "[1/4] Collecting system information..." -ForegroundColor Blue
$OS_VERSION = Get-OSVersion
$ARCH = Get-Architecture
$CPU = Get-CPUInfo
$CPU_CORES = Get-CPUCores
$RAM = Get-RAMInfo
$TIMESTAMP = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd")

Write-Host "  ✓ OS: $OS_VERSION" -ForegroundColor Green
Write-Host "  ✓ Architecture: $ARCH" -ForegroundColor Green
Write-Host "  ✓ CPU: $CPU ($CPU_CORES cores)" -ForegroundColor Green
Write-Host "  ✓ RAM: $RAM" -ForegroundColor Green
Write-Host ""

# Check if Rust is installed
Write-Host "[2/4] Checking Rust installation..." -ForegroundColor Blue
try {
    $RUST_VERSION = (rustc --version).Split()[1]
    Write-Host "  ✓ Rust version: $RUST_VERSION" -ForegroundColor Green
} catch {
    Write-Host "Error: cargo not found. Please install Rust." -ForegroundColor Red
    exit 1
}
Write-Host ""

# Run the benchmark
Write-Host "[3/4] Running MNIST benchmark..." -ForegroundColor Blue
Write-Host "  This may take several minutes..." -ForegroundColor Yellow
Write-Host ""

# Run cargo and capture output
try {
    $output = cargo run --release --example mnist_digits 2>&1 | Tee-Object -Variable cargoOutput
    $output | Out-File -FilePath $TEMP_OUTPUT -Encoding UTF8

    # Display output
    $cargoOutput | ForEach-Object { Write-Host $_ }

    Write-Host ""
    Write-Host "  ✓ Benchmark completed successfully" -ForegroundColor Green
} catch {
    Write-Host "Error: Benchmark failed" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Remove-Item -Path $TEMP_OUTPUT -ErrorAction SilentlyContinue
    exit 1
}
Write-Host ""

# Parse results
Write-Host "[4/4] Parsing results and updating $BENCHMARK_FILE..." -ForegroundColor Blue

# Read the output file
$outputContent = Get-Content -Path $TEMP_OUTPUT -Raw

# Extract metrics from output
$TOTAL_TIME = "N/A"
$FINAL_ACCURACY = "N/A"

if ($outputContent -match "Total time to run:\s*([\d.]+)") {
    $TOTAL_TIME = $matches[1]
}

if ($outputContent -match "Accuracy:\s*([\d.]+)%") {
    $allMatches = [regex]::Matches($outputContent, "Accuracy:\s*([\d.]+)%")
    if ($allMatches.Count -gt 0) {
        $FINAL_ACCURACY = $allMatches[$allMatches.Count - 1].Groups[1].Value
    }
}

if ($TOTAL_TIME -eq "N/A") {
    Write-Host "  Warning: Could not parse total time" -ForegroundColor Yellow
}

if ($FINAL_ACCURACY -eq "N/A") {
    Write-Host "  Warning: Could not parse accuracy" -ForegroundColor Yellow
}

Write-Host "  ✓ Total time: ${TOTAL_TIME}s" -ForegroundColor Green
Write-Host "  ✓ Final accuracy: ${FINAL_ACCURACY}%" -ForegroundColor Green
Write-Host ""

# Initialize benchmarks file if it doesn't exist
if (-not (Test-Path $BENCHMARK_FILE)) {
    $header = @"
# 🚀 MNIST Training Benchmarks

This file contains real-world benchmark results from training the MNIST handwritten digits neural network across different hardware configurations.

## 📊 Benchmark Results

Each entry represents a complete training run (10 epochs, 60,000 samples, batch size 2000) using the ``mnist_digits`` example.

### How to Add Your Benchmark

Run the benchmark script for your platform:

**Linux/macOS:**
``````bash
chmod +x benchmark.sh
./benchmark.sh
``````

**Windows PowerShell:**
``````powershell
.\benchmark.ps1
``````

The script will automatically:
- ✅ Collect your system information
- ✅ Run the MNIST training example
- ✅ Parse the results
- ✅ Append your results to this file

---

## 📈 Results Table

| Date | OS | Architecture | CPU | Cores | RAM (GB) | Rust Version | Time (s) | Accuracy (%) |
|------|----|--------------|----|-------|----------|--------------|----------|--------------|
"@
    Set-Content -Path $BENCHMARK_FILE -Value $header -Encoding UTF8
    Write-Host "  ✓ Created new $BENCHMARK_FILE" -ForegroundColor Green
}

# Append benchmark entry as table row
$entry = "| $TIMESTAMP | $OS_VERSION | $ARCH | $CPU | $CPU_CORES | $RAM | $RUST_VERSION | $TOTAL_TIME | $FINAL_ACCURACY |"

Add-Content -Path $BENCHMARK_FILE -Value $entry -Encoding UTF8
Write-Host "  ✓ Results appended to $BENCHMARK_FILE" -ForegroundColor Green

# Cleanup
Remove-Item -Path $TEMP_OUTPUT -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║            Benchmark completed successfully!         ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "View results: " -NoNewline
Write-Host "Get-Content $BENCHMARK_FILE" -ForegroundColor Cyan
Write-Host ""
