#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2/models/memory-set_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=cpu,floatX=float32

#cd $PBS_O_WORKDIR
python ./translate_gpu.py -n \
        ./model_hal.npz  \
        ./model_hal.npz.pkl  \
	././../../nmtdata/small.ch.pkl \
	././../../nmtdata/small.en.chunked.chunktag.pkl \
	././../../nmtdata/small.test \
	./small.result
