# Chunk-Based Bi-Scaled Decoder for Neural Machine Translation

------

This is the code for paper "Chunk-Based Bi-Scaled Decoder for Neural Machine Translation".

The chunk-based neural machine translation system is on basis of session 2 of [dl4mt-tutorial](https://github.com/nyu-dl/dl4mt-tutorial), which is a attention based encoder-decoder machine translation model. 

The main difference between our proposed model and dl4mt is that we use a bi-scaled decoder to leverage the target-side phrase information for better translation, and propose the phrase attention for phrase level soft alignments. 

## Reguired Software
 * Python 2.7
 * [Theano](http://deeplearning.net/software/theano/)

## Training

    export THEANO_FLAGS=device=gpu2,floatX=float32
    python ./train_nmt_zh2en.py

## Translating

    export THEANO_FLAGS=device=gpu2,floatX=float32
    datadir=/home/zhouh/Data/nmt
    modeldir=./
    
    python ./translate_gpu.py -n -jointProb \
    	$modeldir/model_hal.iter.npz  \
    	$modeldir/model_hal.npz.pkl  \
        $datadir/hms.ch.filter.pkl \
    	$datadir/hms.en.filter.chunked.pkl \
        $datadir/devntest/MT0${i}/MT0${i}.src \
    	./test.result.chunk.${i} 



------


[1]: Hao Zhou, Zhaopeng Tu, Shujian Huang, Xiaohua Liu, Hang Li and Jiajun Chen. Chunk-based Bi-Scale Decoder for Neural Machine Translation. In Proceeding of ACL 2017, short paper.