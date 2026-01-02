#!/usr/bin/env pwsh
# init.ps1 - Windows PowerShell initialization script
# Sets up the Rust AI Neural Network Library environment

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Rust AI Neural Network Library - Init Script" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Rust is installed
Write-Host "Step 1/4: Checking Rust installation..." -ForegroundColor Yellow

$rustcCommand = Get-Command rustc -ErrorAction SilentlyContinue
if (-not $rustcCommand) {
    Write-Host "[ERROR] Rust is not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Rust by running:" -ForegroundColor White
    Write-Host "  Invoke-WebRequest -Uri https://win.rustup.rs/x86_64 -OutFile rustup-init.exe" -ForegroundColor Gray
    Write-Host "  .\rustup-init.exe" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Or visit: https://rustup.rs/" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

$rustVersion = & rustc --version
Write-Host "[OK] Rust is installed: $rustVersion" -ForegroundColor Green

# Check for nightly toolchain
Write-Host ""
Write-Host "Checking for nightly toolchain..." -ForegroundColor Yellow

$nightlyInstalled = & rustup toolchain list | Select-String "nightly"
if (-not $nightlyInstalled) {
    Write-Host "[WARN] Nightly toolchain not found. Installing..." -ForegroundColor Yellow
    & rustup toolchain install nightly
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install nightly toolchain" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "[OK] Nightly toolchain installed" -ForegroundColor Green
} else {
    Write-Host "[OK] Nightly toolchain is available" -ForegroundColor Green
}

# Create training directory
Write-Host ""
Write-Host "Step 2/4: Setting up training directory..." -ForegroundColor Yellow

if (-not (Test-Path "training")) {
    New-Item -ItemType Directory -Path "training" | Out-Null
    Write-Host "[OK] Created training directory" -ForegroundColor Green
} else {
    Write-Host "[OK] Training directory already exists" -ForegroundColor Green
}

# Extract MNIST datasets
Write-Host ""
Write-Host "Step 3/4: Extracting MNIST datasets..." -ForegroundColor Yellow

# Check if archive files exist
if (-not (Test-Path "archive/mnist_train.csv.zip")) {
    Write-Host "[ERROR] archive/mnist_train.csv.zip not found!" -ForegroundColor Red
    Write-Host "Please ensure MNIST dataset archives exist in the archive/ directory" -ForegroundColor White
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path "archive/mnist_test.csv.zip")) {
    Write-Host "[ERROR] archive/mnist_test.csv.zip not found!" -ForegroundColor Red
    Write-Host "Please ensure MNIST dataset archives exist in the archive/ directory" -ForegroundColor White
    Read-Host "Press Enter to exit"
    exit 1
}

# Extract training data
if (-not (Test-Path "training/mnist_train.csv")) {
    Write-Host "Extracting mnist_train.csv..." -ForegroundColor Gray
    try {
        Expand-Archive -Path "archive/mnist_train.csv.zip" -DestinationPath "training/" -Force
        Write-Host "[OK] Extracted mnist_train.csv" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Failed to extract mnist_train.csv.zip: $_" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "[OK] mnist_train.csv already exists" -ForegroundColor Green
}

# Extract test data
if (-not (Test-Path "training/mnist_test.csv")) {
    Write-Host "Extracting mnist_test.csv..." -ForegroundColor Gray
    try {
        Expand-Archive -Path "archive/mnist_test.csv.zip" -DestinationPath "training/" -Force
        Write-Host "[OK] Extracted mnist_test.csv" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Failed to extract mnist_test.csv.zip: $_" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "[OK] mnist_test.csv already exists" -ForegroundColor Green
}

# Build the project
Write-Host ""
Write-Host "Step 4/4: Building the project..." -ForegroundColor Yellow
Write-Host "This may take a few minutes on first run..." -ForegroundColor Gray
Write-Host ""

& cargo +nightly build
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Build failed!" -ForegroundColor Red
    Write-Host "Please check the error messages above." -ForegroundColor White
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[OK] Build successful!" -ForegroundColor Green

# Success message
Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Setup completed successfully!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now run the examples:" -ForegroundColor White
Write-Host "  cargo +nightly run --release --example mnist_handwritten_digits" -ForegroundColor Gray
Write-Host "  cargo +nightly run --release --example mnist_conv_digits" -ForegroundColor Gray
Write-Host ""
Write-Host "Or run tests:" -ForegroundColor White
Write-Host "  cargo +nightly test" -ForegroundColor Gray
Write-Host ""
Write-Host "For more information, see README.md" -ForegroundColor White
Write-Host ""
