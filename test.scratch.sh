#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=gpu0,floatX=float32

modeldir=./
output=outputs

#cd $PBS_O_WORKDIR
#python ./translate_gpu.py -n -p 4 -ck 8 -wk 3 $modeldir/model_hal.npz $modeldir/model_hal.npz.pkl /home/zhouh/Data/nmt/corpus.ch.pkl /home/zhouh/Data/nmt/corpus.en.pkl /home/zhouh/Data/nmt/devntest/MT02/MT02.src ./outputs/$output
if [ ! -f $output ]; then
	mkdir $outputs
fi


for i in $(seq 20000 10000 200000)
do 
modelfile=$modeldir/model_hal.iter${i}.npz
while [ ! -f $modelfile ];do
sleep 1m;
done;
sleep 1m
python ./translate_gpu.py -n -p 4 -ck $1 -wk $2 $modeldir/model_hal.iter${i}.npz $modeldir/model_hal.npz.pkl /home/zhouh/Data/nmt/corpus.ch.filter.pkl /home/zhouh/Data/nmt/corpus.en.filter.pkl /home/zhouh/Data/nmt/devntest/MT02/MT02.src ./outputs/MT02.trans${i}.scratch.en.$1.$2
done

for i in $(seq 201000 1000 500000)
do 
modelfile=$modeldir/model_hal.iter${i}.npz
while [ ! -f $modelfile ];do
sleep 1m;
done;
sleep 1m
python ./translate_gpu.py -n -p 4 -ck $1 -wk $2 $modeldir/model_hal.iter${i}.npz $modeldir/model_hal.npz.pkl /home/zhouh/Data/nmt/corpus.ch.filter.pkl /home/zhouh/Data/nmt/corpus.en.filter.pkl /home/zhouh/Data/nmt/devntest/MT02/MT02.src ./outputs/MT02.trans${i}.scratch.en.$1.$2
done

