# Synthetic Datasets for Human Motion Recognition

![Model architecture](model.png)

A pipeline that generates synthetic datasets from video to train human motion recognition models on.

## Installation

Requirements:
 - ffmpeg
 - unzip (for now)
 - gdown (for now)

1. Create a new virtualenv with Python >= 3.7.
2. `pip install -r requirements.txt`

## Usage

See `python main.py --help` for all options.

Examples:
`python main.py --video_folder ~/videos`