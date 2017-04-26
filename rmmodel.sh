#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=gpu0,floatX=float32

modeldir=./

for i in $(seq 1000 1000 200000)
do 
modelfile=$modeldir/model_hal.iter${i}.npz
rm $modelfile
done

