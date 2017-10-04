#!/bin/bash
set -e
fname=$1
target=$2
mkdir -p $fname/samples
mkdir -p $target
python run_baby.py -l $fname
ffmpeg -framerate 60 -pattern_type glob -i $fname/fields/'*.png' -c:v libx264 -pix_fmt yuv420p $fname/baby_log.mp4
### saving 10 sample fileds
dn=$fname/fields
count=$(find $dn -maxdepth 1 -type f | wc -l)
let counter=1
let interval=1000
for f in $dn/*.png
do
	if ((counter == 1))
	then
		cp $f $fname/samples/
	elif ((counter%interval == 0))
	then
		cp $f $fname/samples/
	fi
	let counter+=1
done
### clearing the fields and copy to destination
rm -r $fname/fields
cp -r $fname $target
#ffmpeg -framerate 60 -pattern_type glob -i $target/$fname/fields/'*.png' -c:v libx264 -pix_fmt yuv420p $target/$fname/baby_log.mp4
rm -r $fname