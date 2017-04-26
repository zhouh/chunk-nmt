#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q ShortQ


export THEANO_FLAGS=device=gpu2,floatX=float32
datadir=/home/zhouh/Data/nmt

iter=398000

#cd $PBS_O_WORKDIR
for i in $(seq 2 1 7)
do
python ./translate_gpu.py -n \
	./model_hal.npz  \
	./model_hal.npz.pkl  \
	$datadir/hms.ch.filter.pkl \
	$datadir/hms.en.filter.chunked.pkl \
	$datadir/devntest/MT0${i}/MT0${i}.src \
	./test.result.chunk.${i}
echo $i >> 3to6.log
perl ../BLEU/multi-bleu.perl /home/zhouh/Data/nmt/devntest/MT0${i}/reference < test.result.chunk.${i} >> 3to6.log
done


