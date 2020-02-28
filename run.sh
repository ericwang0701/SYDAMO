#!/usr/bin/env bash

current_directory=$PWD
script_directory=$(dirname -- "$0")

echo "Going to run VIBE on video directory $1"

cd $script_directory
./vibe-env/bin/python demo.py --videos $1 --output_folder output/
cd $current_directory