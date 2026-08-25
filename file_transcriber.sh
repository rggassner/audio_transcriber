#!/bin/bash

set -e

TRANSCRIBER="$HOME/file_transcriber.py"
VENV="$HOME/venvs/whisper"

echo "Starting offline Brazilian Portuguese transcriber..."
echo

# Activate virtual environment
source "$VENV/bin/activate"

# Start transcriber
python "$TRANSCRIBER"
