#!/usr/bin/env bash
original_directory=$PWD
script_directory=$(dirname -- "$0")

cd script_directory

if [ ! -d "./vibe-env" ]
then
  echo "Creating virtual environment"
  if [[ "$(python3 -V)" =~ "Python 3" ]]
  then
    python3 -m venv vibe-env
  elif [[ "$(python -V)" =~ "Python 3" ]]
  then
    python -m venv vibe-env
  else
    echo "No Python 3 install found"
    exit
  fi
fi

echo "Activating virtual environment"
source ./vibe-env/bin/activate

echo "Installing system dependencies"
apt-get install unzip ffmpeg

echo "Installing pip dependencies"
./vibe-env/bin/pip install numpy torch torchvision gdown
./vibe-env/bin/pip install -r requirements.txt

./prepare_data.sh

cd original_directory