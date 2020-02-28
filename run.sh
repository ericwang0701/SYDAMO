#!/usr/bin/env bash

echo "Going to run VIBE on video directory $1"

$PWD/vibe-env/bin/python $PWD/demo.py --videos $1 --output_folder output/