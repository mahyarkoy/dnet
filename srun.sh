#!/bin/bash
set -e
fname=$1
target=$2
eval_step=1000
mkdir -p $fname
mkdir -p $target

python run_baby.py -l $fname -e $eval_step -s 0
cp -r $fname $target