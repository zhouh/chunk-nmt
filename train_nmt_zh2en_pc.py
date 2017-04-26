import numpy
import os

import numpy
import os

from nmt import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim_chunk=params['dim_chunk'][0],
                     dim_chunk_hidden=params['dim_chunk_hidden'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     patience=1000,
                     batch_size=2,
                     valid_batch_size=2,
                     validFreq=3,
                     dispFreq=10,
                     saveFreq=10,
                     sampleFreq=10,
                     maxlen_chunk=30,  # maximum length of the description
                     maxlen_chunk_words=50,  # maximum length of the description
                     datasets=['/home/zhouh/workspace/python/nmtdata/small.ch',
                               '/home/zhouh/workspace/python/nmtdata/small.en.chunked'],
                     valid_datasets=['/home/zhouh/workspace/python/nmtdata/small.ch',
                                     '/home/zhouh/workspace/python/nmtdata/small.en.chunked'],
                     dictionaries=['/home/zhouh/workspace/python/nmtdata/small.ch.pkl',
                                   '/home/zhouh/workspace/python/nmtdata/small.en.chunked.pkl'],
                     dictionary_chunk='/home/zhouh/workspace/python/nmtdata/small.en.chunked.chunktag.pkl',
                     use_dropout=params['use-dropout'][0],
                     overwrite=False)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['model_hal.npz'],
        'dim_word': [30],
        'dim_chunk': [50],
        'dim_chunk_hidden' : [60],
        'dim': [40],
        'n-words': [100],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [True]})
