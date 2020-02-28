#!/usr/bin/env bash

if [ ! -d "./vibe-env" ] then
  echo "Creating virtual environment"
  python -m venv vibe-env
fi

echo "Activating virtual environment"
source $PWD/vibe-env/bin/activate

echo "Installing system dependencies"
apt-get install unzip ffmpeg

echo "Installing pip dependencies"
$PWD/vibe-env/bin/pip install numpy torch torchvision gdown
$PWD/vibe-env/bin/pip install -r requirements.txt

sh prepare_data.sh

