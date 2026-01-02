@echo off
REM init.bat - Windows initialization script (Batch)
REM Sets up the Rust AI Neural Network Library environment

echo ==================================================
echo   Rust AI Neural Network Library - Init Script
echo ==================================================
echo.

REM Check if Rust is installed
echo Step 1/4: Checking Rust installation...
where rustc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Rust is not installed!
    echo.
    echo Please install Rust by downloading and running:
    echo   https://rustup.rs/
    echo.
    echo Or use the direct installer:
    echo   https://win.rustup.rs/x86_64
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('rustc --version') do set RUST_VERSION=%%i
echo [OK] Rust is installed: %RUST_VERSION%

REM Check for nightly toolchain
echo.
echo Checking for nightly toolchain...
rustup toolchain list | findstr /C:"nightly" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Nightly toolchain not found. Installing...
    rustup toolchain install nightly
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install nightly toolchain
        pause
        exit /b 1
    )
    echo [OK] Nightly toolchain installed
) else (
    echo [OK] Nightly toolchain is available
)

REM Create training directory
echo.
echo Step 2/4: Setting up training directory...
if not exist "training" (
    mkdir training
    echo [OK] Created training directory
) else (
    echo [OK] Training directory already exists
)

REM Extract MNIST datasets
echo.
echo Step 3/4: Extracting MNIST datasets...

REM Check if archive files exist
if not exist "archive\mnist_train.csv.zip" (
    echo [ERROR] archive\mnist_train.csv.zip not found!
    echo Please ensure MNIST dataset archives exist in the archive\ directory
    pause
    exit /b 1
)

if not exist "archive\mnist_test.csv.zip" (
    echo [ERROR] archive\mnist_test.csv.zip not found!
    echo Please ensure MNIST dataset archives exist in the archive\ directory
    pause
    exit /b 1
)

REM Extract training data
if not exist "training\mnist_train.csv" (
    echo Extracting mnist_train.csv...
    powershell -Command "Expand-Archive -Path 'archive\mnist_train.csv.zip' -DestinationPath 'training\' -Force" >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [WARN] PowerShell extraction failed, trying tar...
        tar -xf "archive\mnist_train.csv.zip" -C training >nul 2>nul
        if %ERRORLEVEL% NEQ 0 (
            echo [ERROR] Failed to extract mnist_train.csv.zip
            echo Please extract manually using Windows Explorer or 7-Zip
            pause
            exit /b 1
        )
    )
    echo [OK] Extracted mnist_train.csv
) else (
    echo [OK] mnist_train.csv already exists
)

REM Extract test data
if not exist "training\mnist_test.csv" (
    echo Extracting mnist_test.csv...
    powershell -Command "Expand-Archive -Path 'archive\mnist_test.csv.zip' -DestinationPath 'training\' -Force" >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [WARN] PowerShell extraction failed, trying tar...
        tar -xf "archive\mnist_test.csv.zip" -C training >nul 2>nul
        if %ERRORLEVEL% NEQ 0 (
            echo [ERROR] Failed to extract mnist_test.csv.zip
            echo Please extract manually using Windows Explorer or 7-Zip
            pause
            exit /b 1
        )
    )
    echo [OK] Extracted mnist_test.csv
) else (
    echo [OK] mnist_test.csv already exists
)

REM Build the project
echo.
echo Step 4/4: Building the project...
echo This may take a few minutes on first run...
echo.

cargo +nightly build
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo [OK] Build successful!

REM Success message
echo.
echo ==================================================
echo   Setup completed successfully!
echo ==================================================
echo.
echo You can now run the examples:
echo   cargo +nightly run --release --example mnist_handwritten_digits
echo   cargo +nightly run --release --example mnist_conv_digits
echo.
echo Or run tests:
echo   cargo +nightly test
echo.
echo For more information, see README.md
echo.
pause
