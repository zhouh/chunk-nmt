#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=gpu3,floatX=float32
#export THEANO_FLAGS=device=gpu0,optimizer=None,floatX=float32,exception_verbosity=high
datadir=/home/zhouh/Data/nmt

i=3
for i in $(seq 2 1 5)
do
{ python ./output_align.py \
        ./model_hal.iter398000.npz  \
        ./model_hal.npz.pkl  \
	$datadir/hms.ch.filter.pkl \
	$datadir/hms.en.filter.chunked.pkl \
	$datadir/hms.en.filter.chunked.chunktag.pkl \
	$datadir/devntest/MT0${i}/MT0${i}.src\
	$datadir/devntest/MT0${i}/reference0.tag.chunked.chunked\
	./align.output >> boundary.log.$i

}&
done
