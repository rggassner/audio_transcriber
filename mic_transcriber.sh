#!/bin/bash

set -e

TRANSCRIBER="$HOME/mic_transcriber.py"
VENV="$HOME/venvs/whisper"

mkdir -p "$HOME/transcriptions"

echo "Starting offline Portuguese transcriber..."
echo "Transcriptions: $HOME/transcriptions"
echo

# Activate virtual environment
source "$VENV/bin/activate"

# Start transcriber
python "$TRANSCRIBER"
