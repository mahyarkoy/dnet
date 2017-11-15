#!/bin/bash
set -e
fname=$1
target=$2
mkdir -p $fname/samples
mkdir -p $target
python run_baby.py -l $fname
ffmpeg -framerate 60 -pattern_type glob -i $fname/fields/'*.png' -c:v libx264 -pix_fmt yuv420p $fname/baby_log.mp4
ffmpeg -framerate 60 -pattern_type glob -i $fname/manifolds/'*.png' -c:v libx264 -pix_fmt yuv420p $fname/baby_manifold.mp4
### saving 10 sample fileds
function sample_png {
	dn=$1
	count=$(find $dn -maxdepth 1 -type f | wc -l)
	let counter=1
	let interval=$3
	for f in $dn/*.png
	do
		if ((counter == 1))
		then
			cp $f $2
		elif ((counter%interval == 0))
		then
			cp $f $2
		fi
		let counter+=1
	done
}
sample_png $fname/fields $fname/samples 1000
sample_png $fname/manifolds $fname/samples 1000
### clearing the fields and copy to destination
rm -r $fname/fields
rm -r $fname/manifolds
cp -r $fname $target
#ffmpeg -framerate 60 -pattern_type glob -i $target/$fname/fields/'*.png' -c:v libx264 -pix_fmt yuv420p $target/$fname/baby_log.mp4
rm -r $fname
