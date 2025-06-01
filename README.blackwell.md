# 🧠 Fine-tuning LLMs with Unsloth + PyTorch Nightly + xFormers on Blackwell (RTX 50xx) GPUs

This guide walks you through setting up a full environment for training large language models (LLMs) using Unsloth + QLoRA, fully accelerated with CUDA for NVIDIA Blackwell GPUs (e.g., RTX 5070 Ti, SM 12.0). Includes Linux (WSL) and Windows instructions.

---

## ✅ LINUX / WSL2 SETUP (Recommended)

### 1. Install Poetry and Build Dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv git cmake ninja-build build-essential \
    libopenblas-dev python3-dev libomp-dev gcc-11 g++-11
pip install poetry
poetry install
```

---

### 2. Install PyTorch Nightly with CUDA 12.8 (Blackwell Support)

```bash
poetry run python -m pip uninstall torch torchvision torchaudio -y
poetry run python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

### 3. Install CUDA Toolkit (for Build Only, not Driver)

```bash
sudo apt install -y nvidia-cuda-toolkit
```

If using **WSL2**, follow the instructions from:
[CUDA WSL Install](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network)

---

### 4. Build and Install xFormers from Source (Blackwell-Compatible)

#### Step-by-step:

```bash
# Go to your xformers directory
poetry env activate   # Enter the virtualenv
cd /mnt/d/GitHub/EffycentAI/MirAI/temp/xformers

# Clean previous builds
pip uninstall -y xformers
rm -rf build/ dist/ xformers.egg-info/ .eggs/ __pycache__/

# Clone and sync
git fetch origin
git checkout main
git submodule update --init --recursive
```

#### Set environment variables:

```bash
# CUDA Toolkit paths (adjust if needed)
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Blackwell SM 120 (and fallback to 9.0 if needed)
# 12.0;9.0
export TORCH_CUDA_ARCH_LIST="12.0"  

# Build and compilation flags
export BUILD_EXTENSIONS=1
export FORCE_CUDA=1
export PIP_NO_BUILD_ISOLATION=1
export MAX_JOBS=16


export CC=gcc-11
export CXX=g++-11
```

#### Build and install (without editable `-e`):

```bash
python -m pip install . \
  --no-build-isolation \
  --no-use-pep517 \
  --no-deps -vv
```


> ✅ `--no-build-isolation`: avoids isolated build that ignores your installed PyTorch
> ✅ `--no-use-pep517`: uses legacy `setup.py` instead of `pyproject.toml`
> ✅ `-vv`: enables verbose output for debug

---

### 5. Validate Installation

```bash
poetry run python -c "import xformers; print(xformers.__version__)"
poetry run python -m xformers.info
```

You should see `memory_efficient_attention.cutlassF-pt` and similar entries marked as **available**.

---

## ⚠️ Explanation of Build Environment Variables

| Variable                  | Purpose                                                                  |
| ------------------------- | ------------------------------------------------------------------------ |
| `BUILD_EXTENSIONS=1`      | Ensures C++/CUDA extensions are compiled                                 |
| `FORCE_CUDA=1`            | Forces CUDA support even if `torch.cuda.is_available()` is false         |
| `PIP_NO_BUILD_ISOLATION`  | Prevents `pip` from ignoring your current environment during build       |
| `CC=gcc-11`, `CXX=g++-11` | Ensures compatibility with C++17 requirements (some distros use gcc-12+) |
| `MAX_JOBS=8`              | Controls parallel build jobs to avoid RAM exhaustion                     |
| `TORCH_CUDA_ARCH_LIST`    | Specifies which architectures (e.g. 12.0 = Blackwell) to compile for     |

---

## 🪟 WINDOWS SETUP (Advanced, Less Stable)

### 1. Start Fresh

```bash
conda deactivate
conda remove --name env_llm_qlora --all -y
```

### 2. Create CUDA-Enabled Conda Env

```bash
conda create --name env_llm_qlora python=3.12 pytorch-cuda=12.1 pytorch cudatoolkit \
    -c pytorch -c nvidia -y
conda activate env_llm_qlora
```

### 3. Install Dependencies

```bash
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install ipywidgets ipykernel jmespath tensorboard
pip install --upgrade accelerate
```

### 4. Install Visual Studio Build Tools

* [Download here](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
* Required components:

  * MSVC C++ x64/x86 build tools
  * CMake tools for Windows
  * Windows 10/11 SDK

Run commands in **Developer Command Prompt for VS**.

---

### 5. Install PyTorch Nightly with CUDA 12.8

```bash
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

### 6. Build xFormers from Source (Windows)

```bat
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive

python -m venv venv
.\venv\Scripts\activate

rmdir /s /q build dist xformers.egg-info .eggs __pycache__
pip install -r requirements.txt
pip install cmake ninja

set BUILD_EXTENSIONS=1
set FORCE_CUDA=1
REM Optional: set TORCH_CUDA_ARCH_LIST=12.0

python setup.py bdist_wheel
pip install dist\*.whl
```

Here's the full section you can add to your guide, documenting how to rebuild **Triton** and **cut-cross-entropy (CCE)** for Blackwell GPUs (e.g., RTX 5070 Ti, SM 12.0), and how to bypass Triton issues temporarily:

---

## ⚙️ \[Optional] Rebuilding Triton and CCE for Blackwell GPU (SM 12.0)

If you encounter the following error during training with Unsloth:

```
Assertion `false && "computeCapability not supported"' failed.
…
PassManager::run failed
```

This means **Triton (used by `cut-cross-entropy`) does not yet recognize compute capability 12.0 (Blackwell)**. You have two options:

---

### 🧪 Option 1: Rebuild Triton + CCE with Blackwell Support (Recommended)

To fully leverage fast training on RTX 50xx GPUs, rebuild Triton and CCE to support SM 12.0.

---

#### ✅ Step-by-Step Instructions

**1. Remove old Triton and CCE**

```bash
poetry run pip uninstall -y triton cut-cross-entropy
```
sudo apt update && sudo apt install -y \
  cmake \
  ninja-build \
  gcc \
  g++ \
  git \
  python3-dev \
  libffi-dev \
  libssl-dev \
  libxml2-dev \
  libyaml-dev \
  zlib1g-dev \
  build-essential \
  libncurses5-dev \
  libncursesw5-dev \
  libreadline-dev \
  libsqlite3-dev \
  llvm-dev \
  libclang-dev

---


```bash
# CUDA Toolkit paths (adjust if needed)
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Blackwell SM 120 (and fallback to 9.0 if needed)
# 12.0;9.0
export TORCH_CUDA_ARCH_LIST="12.0"  

# Build and compilation flags
export BUILD_EXTENSIONS=1
export FORCE_CUDA=1
export PIP_NO_BUILD_ISOLATION=1
export MAX_JOBS=16


export CC=gcc-11
export CXX=g++-11
```

**2. Clone the Triton repository**
DIRECTLY: pip install git+https://github.com/openai/triton.git
```bash
cd /mnt/d/GitHub/EffycentAI/MirAI/temp
git clone https://github.com/openai/triton.git
cd triton
```

Make sure you're in the root folder where `pyproject.toml` and `setup.py` exist.

---

**3. Install Triton from source inside Poetry environment**

```bash
# Use regular pip inside the Poetry shell
source $(poetry env info --path)/bin/activate
pip install pybind11
pip install . --no-build-isolation --no-use-pep517
```

This compiles Triton with your installed CUDA (12.8), properly enabling SM 12.0.

> ⚠️ If you encounter errors here, verify that:
>
> * CUDA toolkit is correctly installed (e.g., `/usr/local/cuda-12.8`)
> * `nvcc`, `nvidia-smi`, and `nvidia-cuda-toolkit` are available
> * You have at least `gcc-11` and `g++-11` available and exported via `CC` and `CXX`

---

### 🛠 Option 2: Workaround — Disable Triton Compilation in Unsloth

Temporarily disable Unsloth’s Triton-based optimizations:

```bash
# In your shell before launching training
export UNSLOTH_COMPILE_DISABLE=1
# Alternatively: ignore errors but continue
export UNSLOTH_COMPILE_IGNORE_ERRORS=1

# Then launch your training script
poetry run python -m src.main.training.main_training_base_custom
```

This fallback disables JIT compilation and runs slower but avoids the Triton assertion error.

---

**4. Reinstall `cut-cross-entropy` (CCE)**

Now that Triton is compiled correctly, recompile the fused loss kernels:

```bash
poetry run pip install --no-binary cut-cross-entropy cut-cross-entropy
```

This ensures that CCE uses the Triton version you just compiled, and compiles kernels compatible with `sm_120`.

---

**5. Validate with Triton Info (Optional)**

You can inspect your compiled kernel targets and confirm that Triton now knows about `sm_120`.

If using Triton CLI tools or debugging mode:

```bash
poetry run python -c "import torch; print(torch.cuda.get_device_capability())"
```

Expected output:

```python
(12, 0)
```

---

## ✅ Environment Variables Recap

| Variable                          | Description                                                        |
| --------------------------------- | ------------------------------------------------------------------ |
| `UNSLOTH_COMPILE_DISABLE=1`       | Disables all Triton-based optimizations (fallback mode)            |
| `UNSLOTH_COMPILE_IGNORE_ERRORS=1` | Tries Triton, but continues silently if kernels fail               |
| `TORCH_CUDA_ARCH_LIST=12.0`       | Ensures all custom CUDA builds target Blackwell (SM 120)           |
| `CC=gcc-11`, `CXX=g++-11`         | Forces compatibility for C++ kernels during Triton/CCE compilation |

---

## 🚀 Recommended Order (for Full Performance on RTX 50xx)

1. ✅ Setup Poetry and PyTorch Nightly with CUDA 12.8
2. ✅ Rebuild xFormers (SM 12.0 support)
3. ✅ Clone and install Triton from source
4. ✅ Recompile `cut-cross-entropy`
5. ✅ Set `TORCH_CUDA_ARCH_LIST="12.0"` and train your model with Unsloth

---

## ✅ Final Notes

* Always run `poetry shell` before installing to ensure environment isolation.
* If you're using `Unsloth`, make sure `cut_cross_entropy` is **disabled** or replaced if incompatible with your GPU.
* Validate final training runs using `torch.cuda.get_device_capability()` and `xformers.info`.
* Set `TORCH_CUDA_ARCH_LIST="12.0"` for full Blackwell support. If unsure, also add `"9.0"` for compatibility.

---

Let me know if you want this exported as `.md`, `.pdf`, or auto-pushed to a GitHub repo with a working example!
