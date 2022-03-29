ROOT_PATH="/paddle"

import shutil
import subprocess
import sys

# Add ALT path to sys.path
ALT_PATH = f"{ROOT_PATH}/chao/AcrossLevelTracer"
sys.path.append(ALT_PATH)
import AcrossLevelTracer as ALT

# -------------------------------------------------------------------------------------------------------------
# Parsing arguments

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--configs", help="Specify configurations of training model.")
parser.add_argument("-e", "--extra", help="Passing extra arguments for training.")
parser.add_argument("-v", "--visualize", action="store_true", help="Enable visualization.")
args = parser.parse_args()

if not args.configs:
    model_configs = f"{ROOT_PATH}/chao/PaddleClas/ppcls/configs/ImageNet/ResNet/ResNet50.yaml"
    print(f"[WARNING]: -c or --configs not specified! Using {model_configs} by default ...\n")
else:
    model_configs = f"{ROOT_PATH}/chao/PaddleClas/{args.configs}"
    print(f"[INFO]: Using {model_configs} ...\n")

extra_args = args.extra if args.extra else ""

# -------------------------------------------------------------------------------------------------------------
### Helper functions ###

python_bin = shutil.which("python3.7")

def run_alt(cuda_injection64_path, model_configs, extra_arguments="", use_alt=True):
    ALT = "_ALT" if use_alt else ""
    print("[INFO] Running paddle profiler...")
    with subprocess.Popen([python_bin, f"{ROOT_PATH}/chao/PaddleClas/tools/train{ALT}.py", "-c", model_configs]
                                                                + extra_arguments.split(), 
                                                    env=cuda_injection64_path) as process:
        pid = process.pid
        process.wait()
        if process.returncode != 0: 
            print("[ERROR] Something went wrong! pid:", pid)
            exit()
        else:
            print("[SUCCESS] pid:", pid)

    return pid

# -------------------------------------------------------------------------------------------------------------
### DynamicDumper + Profiler ###

DD_PATH = f"{ROOT_PATH}/chao/DynamicDumper/lib/libDynamicDumper.so"
CUDA_INJECTION = {"CUDA_INJECTION64_PATH":DD_PATH}

pid_dd = run_alt(CUDA_INJECTION, model_configs, extra_args)

# -------------------------------------------------------------------------------------------------------------
### Correlation ###

print("[INFO] Running correlation...")
tracer = ALT.AcrossLevelTracer(framework=0, ops_status=1)
tracer.runPostProcess(pid_dd) 
print("[SUCCESS] Correlation completed!\n")

# -------------------------------------------------------------------------------------------------------------
### Visualization ###

if args.visualize:
    process_vis = subprocess.Popen([python_bin, f"{ROOT_PATH}/chao/PaddleClas/operator-cuda-kernel-viz.py", "-p", str(pid_dd)])

# -------------------------------------------------------------------------------------------------------------
### CUptiTracer ###

CT_PATH = f"{ROOT_PATH}/chao/CUptiTracer/lib/libCUptiTracer.so"
CUDA_INJECTION = {"CUDA_INJECTION64_PATH":CT_PATH}

pid_cupti = run_alt(CUDA_INJECTION, model_configs, extra_args, use_alt=False)

# -------------------------------------------------------------------------------------------------------------
### Wait for visualization Process ###

if args.visualize:
    process_vis.wait()
