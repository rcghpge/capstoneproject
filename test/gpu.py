import subprocess
import sys
import xgboost as xgb

def check_nvidia_smi():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("nvidia-smi not found or no NVIDIA GPU detected.")
            return False
        print("nvidia-smi found, GPU Info:")
        print(result.stdout.decode())
        return True
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
        return False

def check_xgboost_version():
    print(f"XGBoost version: {xgb.__version__}")

def check_xgboost_gpu_support():
    try:
        booster = xgb.Booster(params={'tree_method': 'gpu_hist'})
        print("XGBoost GPU hist tree method is available.")
        return True
    except Exception as e:
        print(f"XGBoost GPU hist tree method not available: {e}")
        return False

def main():
    print("Checking GPU availability...")
    gpu_available = check_nvidia_smi()
    check_xgboost_version()
    if gpu_available:
        check_xgboost_gpu_support()
    else:
        print("Skipping XGBoost GPU check due to no GPU detected.")

if __name__ == "__main__":
    main()
