#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=168:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q GpuQ

#export THEANO_FLAGS=device=gpu0,optimizer=None,floatX=float32,exception_verbosity=high
export THEANO_FLAGS=device=gpu2,floatX=float32

python ./train_nmt_zh2en.py



