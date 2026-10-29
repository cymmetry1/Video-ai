#!/usr/bin/env bash
set -e
sudo apt-get update
sudo apt-get install -y ffmpeg libgl1 libglib2.0-0
python -m pip install --upgrade pip
pip install fastapi 'uvicorn[standard]' python-multipart
if [ ! -d Deep-Live-Cam ]; then
  git clone --depth 1 https://github.com/hacksider/Deep-Live-Cam.git
fi
cd Deep-Live-Cam
pip install -r requirements.txt
cd ..
echo 'Video AI setup complete.'
