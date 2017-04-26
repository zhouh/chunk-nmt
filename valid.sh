#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=gpu1,floatX=float32



modeldir=.

datadir=/home/zhouh/Data/nmt/

modelfile=$modeldir/model_hal
python ./validate.py $modelfile ./model_hal.npz.pkl ./bleu.log ./outputs/test.result ../BLEU/multi-bleu.perl $datadir/hms.ch.filter.pkl $datadir/hms.en.filter.chunked.pkl  $datadir/hms.en.filter.chunked.chunktag.pkl $datadir/devntest/MT02/MT02.src $datadir/devntest/MT02/reference0.tag.chunked.chunked $datadir/devntest/MT02/reference

