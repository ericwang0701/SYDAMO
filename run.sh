#!/usr/bin/env bash

script_directory=$(dirname -- "$0")

echo "Going to run VIBE on video directory $1"

$script_directory/vibe-env/bin/python $script_directory/demo.py --videos $1 --output_folder output/