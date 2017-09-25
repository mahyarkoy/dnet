#!/bin/bash
set -e
fname=$1
target=$2
mkdir -p $fname
python run_baby.py -l $fname
ffmpeg -framerate 60 -pattern_type glob -i $fname/fields/'*.png' -c:v libx264 -pix_fmt yuv420p $fname/baby_log.mp4
rm -r $fname/fields
cp -r $fname $target
#ffmpeg -framerate 60 -pattern_type glob -i $target/$fname/fields/'*.png' -c:v libx264 -pix_fmt yuv420p $target/$fname/baby_log.mp4
rm -r $fname
