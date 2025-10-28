#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive CPU and GPU Environment Check Script
Tests hardware, libraries, and compute capabilities
"""

import sys
import platform
import subprocess
import os

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def check_python_environment():
    """Check Python version and environment details"""
    print_section("PYTHON ENVIRONMENT")
    print(f"Python Version:    {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform:          {platform.platform()}")
    print(f"Architecture:      {platform.machine()}")
    print(f"Processor:         {platform.processor()}")


def check_cpu_info():
    """Check CPU information"""
    print_section("CPU INFORMATION")
    
    try:
        import psutil
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        
        print(f"Physical CPU Cores: {cpu_count_physical}")
        print(f"Logical CPU Cores:  {cpu_count_logical}")
        if cpu_freq:
            print(f"CPU Frequency:      {cpu_freq.current:.2f} MHz (max: {cpu_freq.max:.2f} MHz)")
        
        # Memory info
        mem = psutil.virtual_memory()
        print(f"\nTotal RAM:          {mem.total / (1024**3):.2f} GB")
        print(f"Available RAM:      {mem.available / (1024**3):.2f} GB")
        print(f"Used RAM:           {mem.used / (1024**3):.2f} GB ({mem.percent}%)")
    except ImportError:
        print("psutil not installed. Install with: pip install psutil")
        print(f"CPU Count (logical): {os.cpu_count()}")


def check_gpu_nvidia():
    """Check NVIDIA GPU availability and details"""
    print_section("NVIDIA GPU INFORMATION")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("nvidia-smi output:")
            print(result.stdout)
        else:
            print("nvidia-smi command failed or no NVIDIA GPU detected")
            print(result.stderr)
    except FileNotFoundError:
        print("nvidia-smi not found. NVIDIA drivers may not be installed.")
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
    
    # Check CUDA availability via PyTorch
    print("\nPyTorch CUDA Check:")
    try:
        import torch
        print(f"PyTorch version:        {torch.__version__}")
        print(f"CUDA available:         {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version:           {torch.version.cuda}")
            print(f"cuDNN version:          {torch.backends.cudnn.version()}")
            print(f"Number of GPUs:         {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Compute Capability:  {props.major}.{props.minor}")
                print(f"    Total Memory:        {props.total_memory / (1024**3):.2f} GB")
    except ImportError:
        print("PyTorch not installed. Install with: pip install torch")
    except Exception as e:
        print(f"Error checking PyTorch CUDA: {e}")


def check_xgboost_gpu():
    """Check XGBoost GPU support"""
    print_section("XGBOOST GPU SUPPORT")
    
    try:
        import xgboost as xgb
        print(f"XGBoost version: {xgb.__version__}")
        
        # Test GPU training
        print("\nTesting XGBoost GPU training...")
        try:
            import numpy as np
            from sklearn.datasets import make_regression
            
            X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
            dtrain = xgb.DMatrix(X, label=y)
            
            params = {
                'tree_method': 'gpu_hist',
                'verbosity': 0
            }
            
            print("Training small model with tree_method='gpu_hist'...")
            bst = xgb.train(params, dtrain, num_boost_round=10)
            print("✓ XGBoost GPU training successful!")
            
        except Exception as e:
            print(f"✗ XGBoost GPU training failed: {e}")
            print("  Try CPU fallback with tree_method='hist'")
            
    except ImportError:
        print("XGBoost not installed. Install with: pip install xgboost")
    except Exception as e:
        print(f"Error checking XGBoost: {e}")


def check_tensorflow_gpu():
    """Check TensorFlow GPU support"""
    print_section("TENSORFLOW GPU SUPPORT")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Number of GPUs available: {len(gpus)}")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    print(f"    Details: {details}")
                except:
                    pass
        else:
            print("No GPUs detected by TensorFlow")
            
    except ImportError:
        print("TensorFlow not installed. Install with: pip install tensorflow")
    except Exception as e:
        print(f"Error checking TensorFlow: {e}")


def check_sklearn():
    """Check scikit-learn version"""
    print_section("SCIKIT-LEARN")
    
    try:
        import sklearn
        print(f"scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("scikit-learn not installed. Install with: pip install scikit-learn")


def check_pandas_numpy():
    """Check pandas and numpy versions"""
    print_section("DATA SCIENCE LIBRARIES")
    
    try:
        import pandas as pd
        print(f"pandas version: {pd.__version__}")
    except ImportError:
        print("pandas not installed. Install with: pip install pandas")
    
    try:
        import numpy as np
        print(f"numpy version:  {np.__version__}")
    except ImportError:
        print("numpy not installed. Install with: pip install numpy")


def run_simple_benchmark():
    """Run a simple CPU vs GPU benchmark if available"""
    print_section("SIMPLE COMPUTE BENCHMARK")
    
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            print("GPU not available for benchmark")
            return
        
        size = 5000
        print(f"Matrix multiplication benchmark ({size}x{size})...")
        
        # CPU benchmark
        cpu_tensor = torch.randn(size, size)
        start = time.time()
        cpu_result = torch.mm(cpu_tensor, cpu_tensor)
        cpu_time = time.time() - start
        print(f"  CPU time: {cpu_time:.4f} seconds")
        
        # GPU benchmark
        gpu_tensor = torch.randn(size, size).cuda()
        torch.cuda.synchronize()
        start = time.time()
        gpu_result = torch.mm(gpu_tensor, gpu_tensor)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"  GPU time: {gpu_time:.4f} seconds")
        
        speedup = cpu_time / gpu_time
        print(f"  GPU Speedup: {speedup:.2f}x faster")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")


def main():
    """Main function to run all checks"""
    print("\n" + "=" * 80)
    print(" COMPUTE ENVIRONMENT CHECK")
    print(" Comprehensive CPU and GPU Hardware & Library Test")
    print("=" * 80)
    
    check_python_environment()
    check_cpu_info()
    check_gpu_nvidia()
    check_xgboost_gpu()
    check_tensorflow_gpu()
    check_sklearn()
    check_pandas_numpy()
    run_simple_benchmark()
    
    print("\n" + "=" * 80)
    print(" ENVIRONMENT CHECK COMPLETE")
    print("=" * 80)
    print("\nRecommendations:")
    print("  - If no GPU detected but you have one: Install CUDA drivers and toolkit")
    print("  - For XGBoost GPU: pip install xgboost (ensure CUDA-compatible version)")
    print("  - For PyTorch GPU: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("  - Monitor GPU usage: watch -n 1 nvidia-smi")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
