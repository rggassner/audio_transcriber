#!/bin/bash

set -e

# ============================================================
# Offline Portuguese Whisper Transcriber
# Ubuntu + NVIDIA GPU
# ============================================================

VENV="$HOME/venvs/whisper"

echo
echo "============================================================"
echo " OFFLINE PORTUGUESE WHISPER INSTALLER"
echo "============================================================"
echo

# ------------------------------------------------------------
# Check that we're running on Linux
# ------------------------------------------------------------

if [ "$(uname -s)" != "Linux" ]; then
    echo "ERROR: This installer is intended for Linux."
    exit 1
fi

# ------------------------------------------------------------
# Check Python
# ------------------------------------------------------------

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is not installed."
    echo
    echo "Install it with:"
    echo "  sudo apt install python3 python3-venv"
    exit 1
fi

echo "Python:"
python3 --version
echo

# ------------------------------------------------------------
# Check NVIDIA driver
# ------------------------------------------------------------

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo
else
    echo "WARNING: nvidia-smi was not found."
    echo "The transcription software may not be able to use CUDA."
    echo
fi

# ------------------------------------------------------------
# Install system dependencies
# ------------------------------------------------------------

echo "Installing system dependencies..."

sudo apt update

sudo apt install -y \
    python3 \
    python3-venv \
    python3-pip \
    ffmpeg \
    libportaudio2

echo
echo "System dependencies installed."
echo

# ------------------------------------------------------------
# Create virtual environment
# ------------------------------------------------------------

if [ -d "$VENV" ]; then
    echo "Virtual environment already exists:"
    echo "  $VENV"
    echo
else
    echo "Creating Python virtual environment:"
    echo "  $VENV"
    echo

    mkdir -p "$HOME/venvs"

    python3 -m venv "$VENV"
fi

# ------------------------------------------------------------
# Upgrade Python packaging tools
# ------------------------------------------------------------

echo "Updating pip/setuptools/wheel..."

"$VENV/bin/python" -m pip install --upgrade \
    pip \
    setuptools \
    wheel

# ------------------------------------------------------------
# Install Python packages
# ------------------------------------------------------------

echo
echo "Installing Python packages..."

"$VENV/bin/python" -m pip install \
    faster-whisper \
    sounddevice \
    scipy \
    numpy

# ------------------------------------------------------------
# Install NVIDIA CUDA runtime libraries required by
# faster-whisper / CTranslate2
# ------------------------------------------------------------

echo
echo "Installing NVIDIA CUDA runtime libraries..."

"$VENV/bin/python" -m pip install \
    nvidia-cublas-cu12 \
    nvidia-cudnn-cu12

# ------------------------------------------------------------
# Create helper activation script
# ------------------------------------------------------------

ACTIVATE_HELPER="$VENV/activate_whisper.sh"

cat > "$ACTIVATE_HELPER" <<EOF
#!/bin/bash
source "$VENV/bin/activate"

export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib:\$LD_LIBRARY_PATH"

echo "Whisper environment activated."
echo "Python: \$(which python)"
echo
EOF

chmod +x "$ACTIVATE_HELPER"

# ------------------------------------------------------------
# Create transcription directory
# ------------------------------------------------------------

mkdir -p "$HOME/transcriptions"

# ------------------------------------------------------------
# Test imports
# ------------------------------------------------------------

echo
echo "Testing Python environment..."

"$VENV/bin/python" - <<'PY'
import faster_whisper
import sounddevice
import scipy
import numpy

print("  faster-whisper : OK")
print("  sounddevice    : OK")
print("  scipy          : OK")
print("  numpy          : OK")
PY

# ------------------------------------------------------------
# Test CUDA libraries
# ------------------------------------------------------------

echo
echo "Testing CUDA libraries..."

export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

"$VENV/bin/python" - <<'PY'
import ctypes

libraries = [
    "libcublas.so.12",
    "libcudnn.so.9",
]

failed = False

for library in libraries:
    try:
        ctypes.CDLL(library)
        print(f"  {library}: OK")
    except OSError as exc:
        print(f"  {library}: FAILED")
        print(f"    {exc}")
        failed = True

if failed:
    raise SystemExit(1)
PY

# ------------------------------------------------------------
# Finished
# ------------------------------------------------------------

echo
echo "============================================================"
echo " INSTALLATION COMPLETE"
echo "============================================================"
echo
echo "Virtual environment:"
echo "  $VENV"
echo
echo "Activate it with:"
echo "  source $VENV/bin/activate"
echo
echo "Or activate it with CUDA libraries configured:"
echo "  source $VENV/activate_whisper.sh"
echo
echo "Transcriptions:"
echo "  $HOME/transcriptions"
echo
echo "Run your transcriber directly with:"
echo "  $VENV/bin/python ~/mic_transcriber.py"
echo
echo "============================================================"
echo
