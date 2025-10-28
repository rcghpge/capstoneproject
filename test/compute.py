#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive CPU and GPU Environment Check Script
Tests hardware, libraries, and compute capabilities
Outputs logged both to console and environment_check_output.txt
"""

import sys
import platform
import subprocess
import os

def print_section(title, out=sys.stdout):
    out.write("\n" + "=" * 80 + f"\n {title}\n" + "=" * 80 + "\n")

def get_physical_cpu_count():
    system = platform.system()
    if system == "Windows":
        try:
            import wmi
            c = wmi.WMI()
            cpus = c.Win32_Processor()
            return len(cpus)
        except ImportError:
            return None
    elif system == "Linux":
        try:
            physical_ids = set()
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.strip().startswith('physical id'):
                        _, value = line.split(':')
                        physical_ids.add(int(value.strip()))
            return len(physical_ids) if physical_ids else None
        except:
            return None
    elif system == "Darwin":  # macOS
        try:
            import subprocess
            output = subprocess.check_output(['sysctl', '-n', 'hw.packages'])
            return int(output)
        except:
            return None
    else:
        return None

def check_python_environment(out=sys.stdout):
    print_section("PYTHON ENVIRONMENT", out)
    out.write(f"Python Version:    {sys.version}\n")
    out.write(f"Python Executable: {sys.executable}\n")
    out.write(f"Platform:          {platform.platform()}\n")
    out.write(f"Architecture:      {platform.machine()}\n")
    out.write(f"Processor:         {platform.processor()}\n")

def check_cpu_info(out=sys.stdout):
    print_section("CPU INFORMATION", out)
    try:
        import psutil
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        physical_packages = get_physical_cpu_count()
        out.write(f"Physical CPU Cores: {cpu_count_physical}\n")
        out.write(f"Logical CPU Cores:  {cpu_count_logical}\n")
        out.write(f"Physical CPU Packages (sockets): {physical_packages if physical_packages is not None else 'Unknown'}\n")
        if cpu_freq:
            out.write(f"CPU Frequency:      {cpu_freq.current:.2f} MHz (max: {cpu_freq.max:.2f} MHz)\n")

        mem = psutil.virtual_memory()
        out.write(f"\nTotal RAM:          {mem.total / (1024**3):.2f} GB\n")
        out.write(f"Available RAM:      {mem.available / (1024**3):.2f} GB\n")
        out.write(f"Used RAM:           {mem.used / (1024**3):.2f} GB ({mem.percent}%)\n")
    except ImportError:
        out.write("psutil not installed. Install with: pip install psutil\n")
        out.write(f"CPU Count (logical): {os.cpu_count()}\n")

def check_gpu_nvidia(out=sys.stdout):
    print_section("NVIDIA GPU INFORMATION", out)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            out.write("nvidia-smi output:\n")
            out.write(result.stdout + "\n")
        else:
            out.write("nvidia-smi command failed or no NVIDIA GPU detected\n")
            out.write(result.stderr + "\n")
    except FileNotFoundError:
        out.write("nvidia-smi not found. NVIDIA drivers may not be installed.\n")
    except Exception as e:
        out.write(f"Error running nvidia-smi: {e}\n")

    out.write("\nPyTorch CUDA Check:\n")
    try:
        import torch
        out.write(f"PyTorch version:        {torch.__version__}\n")
        out.write(f"CUDA available:         {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            out.write(f"CUDA version:           {torch.version.cuda}\n")
            out.write(f"cuDNN version:          {torch.backends.cudnn.version()}\n")
            out.write(f"Number of GPUs:         {torch.cuda.device_count()}\n")
            for i in range(torch.cuda.device_count()):
                out.write(f"  GPU {i}: {torch.cuda.get_device_name(i)}\n")
                props = torch.cuda.get_device_properties(i)
                out.write(f"    Compute Capability:  {props.major}.{props.minor}\n")
                out.write(f"    Total Memory:        {props.total_memory / (1024**3):.2f} GB\n")
    except ImportError:
        out.write("PyTorch not installed. Install with: pip install torch\n")
    except Exception as e:
        out.write(f"Error checking PyTorch CUDA: {e}\n")

def check_xgboost_gpu(out=sys.stdout):
    print_section("XGBOOST GPU SUPPORT", out)
    try:
        import xgboost as xgb
        out.write(f"XGBoost version: {xgb.__version__}\n")

        out.write("\nTesting XGBoost GPU training...\n")
        try:
            import numpy as np
            from sklearn.datasets import make_regression

            X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
            dtrain = xgb.DMatrix(X, label=y)

            params = {
                'tree_method': 'gpu_hist',
                'verbosity': 0
            }

            out.write("Training small model with tree_method='gpu_hist'...\n")
            bst = xgb.train(params, dtrain, num_boost_round=10)
            out.write("✓ XGBoost GPU training successful!\n")

        except Exception as e:
            out.write(f"✗ XGBoost GPU training failed: {e}\n")
            out.write("  Try CPU fallback with tree_method='hist'\n")

    except ImportError:
        out.write("XGBoost not installed. Install with: pip install xgboost\n")
    except Exception as e:
        out.write(f"Error checking XGBoost: {e}\n")

def check_tensorflow_gpu(out=sys.stdout):
    print_section("TENSORFLOW GPU SUPPORT", out)
    try:
        import tensorflow as tf
        out.write(f"TensorFlow version: {tf.__version__}\n")

        gpus = tf.config.list_physical_devices('GPU')
        out.write(f"Number of GPUs available: {len(gpus)}\n")

        if gpus:
            for i, gpu in enumerate(gpus):
                out.write(f"  GPU {i}: {gpu.name}\n")
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    out.write(f"    Details: {details}\n")
                except:
                    pass
        else:
            out.write("No GPUs detected by TensorFlow\n")

    except ImportError:
        out.write("TensorFlow not installed. Install with: pip install tensorflow\n")
    except Exception as e:
        out.write(f"Error checking TensorFlow: {e}\n")

def check_sklearn(out=sys.stdout):
    print_section("SCIKIT-LEARN", out)
    try:
        import sklearn
        out.write(f"scikit-learn version: {sklearn.__version__}\n")
    except ImportError:
        out.write("scikit-learn not installed. Install with: pip install scikit-learn\n")

def check_pandas_numpy(out=sys.stdout):
    print_section("DATA SCIENCE LIBRARIES", out)
    try:
        import pandas as pd
        out.write(f"pandas version: {pd.__version__}\n")
    except ImportError:
        out.write("pandas not installed. Install with: pip install pandas\n")

    try:
        import numpy as np
        out.write(f"numpy version:  {np.__version__}\n")
    except ImportError:
        out.write("numpy not installed. Install with: pip install numpy\n")

def run_simple_benchmark(out=sys.stdout):
    print_section("SIMPLE COMPUTE BENCHMARK", out)

    try:
        import torch
        import time

        if not torch.cuda.is_available():
            out.write("GPU not available for benchmark\n")
            return

        size = 5000
        out.write(f"Matrix multiplication benchmark ({size}x{size})...\n")

        # CPU benchmark
        cpu_tensor = torch.randn(size, size)
        start = time.time()
        cpu_result = torch.mm(cpu_tensor, cpu_tensor)
        cpu_time = time.time() - start
        out.write(f"  CPU time: {cpu_time:.4f} seconds\n")

        # GPU benchmark
        gpu_tensor = torch.randn(size, size).cuda()
        torch.cuda.synchronize()
        start = time.time()
        gpu_result = torch.mm(gpu_tensor, gpu_tensor)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        out.write(f"  GPU time: {gpu_time:.4f} seconds\n")

        speedup = cpu_time / gpu_time
        out.write(f"  GPU Speedup: {speedup:.2f}x faster\n")

    except Exception as e:
        out.write(f"Benchmark failed: {e}\n")

def main():
    with open('environment_check_output.txt', 'w') as f:
        # Write to both stdout and file
        import sys
        class Tee(object):
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()

        tee = Tee(sys.stdout, f)

        check_python_environment(out=tee)
        check_cpu_info(out=tee)
        check_gpu_nvidia(out=tee)
        check_xgboost_gpu(out=tee)
        check_tensorflow_gpu(out=tee)
        check_sklearn(out=tee)
        check_pandas_numpy(out=tee)
        run_simple_benchmark(out=tee)

        tee.write("\n=== ENVIRONMENT CHECK COMPLETE ===\n")

if __name__ == "__main__":
    main()
