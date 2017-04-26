#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q ShortQ


export THEANO_FLAGS=device=gpu1,floatX=float32

#cd $PBS_O_WORKDIR

modeldir=./
gap=`expr $1 \* 1000`
echo $gap

mkdir outputs
rm test.log

for i in $(seq 250000 $gap 500000)
do
    for j in $(seq 1000 1000 $gap)
    do
    {   iter=`expr $i + $j`

        modelfile=$modeldir/model_hal.iter${iter}.npz
        while [ ! -f $modelfile ];do
        sleep 1m;
        done;

        outputfile=./outputs/MT02.trans${iter}.en

        python ./translate_gpu.py -n $modelfile $modeldir/model_hal.npz.pkl /home/zhouh/Data/nmt/hms.ch.filter.pkl /home/zhouh/Data/nmt/hms.en.filter.chunked.pkl /home/zhouh/Data/nmt/devntest/MT02/MT02.src $outputfile
	echo ${iter} >> test.log
        perl ../BLEU/multi-bleu.perl ~/Data/nmt/devntest/MT02/reference < $outputfile >>test.log
    }&


    done
    wait
done



