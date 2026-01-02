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
| 2026-01-02 | Ubuntu 25.10 | x86_64 | Intel(R) Core(TM) i3-N300 | 8 | 14.8 GB | 1.94.0-nightly | 17.072264 | 92.109 |
