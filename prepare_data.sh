#!/usr/bin/env bash

if [ ! -d ./data ]; then
  echo "Downloading VIBE data from Google Drive.."
  mkdir -p data
  cd data
  gdown https://drive.google.com/uc?id=1untXhYOLQtpNEy4GTY_0fL_H-k6cTf_r
  unzip vibe_data.zip
  rm vibe_data.zip
  cd ..
  mv data/vibe_data/sample_video.mp4 .
else
  echo "./data directory exists, assuming it's up-to-date"
fi