#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2/models/memory-set_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=cpu,floatX=float32

#cd $PBS_O_WORKDIR
python ./translate.py -n -p 8 \
        ./models/model_hal.npz  \
	$HOME/Data/nmt/corpus.ch.pkl \
	$HOME/Data/nmt/corpus.en.pkl \
	$HOME/Data/nmt/devntest/MT02/MT02.src\
	./result/MT02.trans.en

python ./translate.py -n -p 8 \
        ./models/model_hal.npz  \
        $HOME/Data/nmt/corpus.ch.pkl \
        $HOME/Data/nmt/corpus.en.pkl \
        $HOME/Data/nmt/devntest/MT03/MT03.src\
        ./result/MT03.trans.en

python ./translate.py -n -p 8 \
        ./models/model_hal.npz  \
        $HOME/Data/nmt/corpus.ch.pkl \
        $HOME/Data/nmt/corpus.en.pkl \
        $HOME/Data/nmt/devntest/MT04/MT04.src\
        ./result/MT04.trans.en

python ./translate.py -n -p 8 \
        ./models/model_hal.npz  \
        $HOME/Data/nmt/corpus.ch.pkl \
        $HOME/Data/nmt/corpus.en.pkl \
        $HOME/Data/nmt/devntest/MT05/MT05.src\
        ./result/MT05.trans.en

python ./translate.py -n -p 8 \
        ./models/model_hal.npz  \
        $HOME/Data/nmt/corpus.ch.pkl \
        $HOME/Data/nmt/corpus.en.pkl \
        $HOME/Data/nmt/devntest/MT06/MT06.src\
        ./result/MT06.trans.en
