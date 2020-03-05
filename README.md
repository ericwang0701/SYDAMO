# Synthetic Datasets for Human Motion Recognition

![Model architecture](model.png)

A pipeline that generates synthetic datasets from video to train human motion recognition models on.

## Installation

1. Install `ffmpeg` and Blender.
2. Create a new virtualenv with Python >= 3.7. (e.g. `python -m venv venv`).
3. Activate environment (e.g. `source venv/bin/activate`).
4. Install dependencies `pip install -r requirements.txt`.

## Usage

See `python main.py --help` for all options.

Examples:
`python main.py --video_folder ~/videos`