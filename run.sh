#!/bin/bash
set -e
pfname=$1
target=$2
eval_step=100
mkdir -p $pfname
mkdir -p $target

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
	cp $f $2
}

for i in {0..2}
do
	fname=$pfname/run_$i
	mkdir -p $fname/samples
	python run_baby.py -l $fname -e $eval_step  -s $i
	ffmpeg -framerate 30 -pattern_type glob -i $fname/fields/'*.png' -c:v libx264 -pix_fmt yuv420p $fname/baby_log.mp4
	#ffmpeg -framerate 60 -pattern_type glob -i $fname/manifolds/'*.png' -c:v libx264 -pix_fmt yuv420p $fname/baby_manifold.mp4

	sample_png $fname/fields $fname/samples 1000
	#sample_png $fname/manifolds $fname/samples 100
	### clearing the fields
	rm -r $fname/fields
	rm -r $fname/manifolds
done
cp -r $pfname $target
#ffmpeg -framerate 60 -pattern_type glob -i $target/$fname/fields/'*.png' -c:v libx264 -pix_fmt yuv420p $target/$fname/baby_log.mp4
rm -r $pfname
