#!/bin/bash
set -e
pfname=$1
target=$2
eval_step=1000
mkdir -p $pfname
mkdir -p $target

for i in {0..2}
do
	fname=$pfname/run_$i
	python run_baby.py -l $fname -e $eval_step -s $i
done
cp -r $pfname $target