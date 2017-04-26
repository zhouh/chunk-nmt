#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=168:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q GpuQ

export THEANO_FLAGS=device=cpu,optimizer=None,floatX=float32,exception_verbosity=high

python ./train_nmt_zh2en_pc.py



