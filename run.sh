#!/bin/bash

fname=baby_log_r0
target=/media/evl/Public/Mahyar/baby_logs
python run_baby.py -l $fname
cp -r $fname $target
ffmpeg -framerate 60 -i $target/$fname/fields/field_%d.png -c:v libx264 -pix_fmt yuv420p $target/$fname/baby_log.mp4
rm -r $fname
