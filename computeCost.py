'''
Translates a source file using a translation model.
'''
import argparse

import numpy
import theano
import cPickle as pkl

from nmt import (build_model, pred_probs, load_params,
                 init_params, init_tparams, prepare_training_data)

from training_data_iterator import TrainingTextIterator



def main(model,
         pklmodel,
         valid_datasets=['../data/dev/newstest2011.en.tok',
                          '../data/dev/newstest2011.fr.tok'],
         dictionaries=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],
         dictionary_chunk='/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
         result_file='./cost.result'):





    # load the dictionaries of both source and target
    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        with open(dd, 'rb') as f:
            worddicts[ii] = pkl.load(f)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    # dict for chunk label
    worddict_chunk = [None]
    worddict_r_chunk = [None]
    with open(dictionary_chunk, 'rb') as f:
        worddict_chunk = pkl.load(f)
    worddict_r_chunk = dict()
    for kk, vv in worddict_chunk.iteritems():
        worddict_r_chunk[vv] = kk
    print worddict_chunk

    print 'load model model_options'
    with open('%s' % pklmodel, 'rb') as f:
        options = pkl.load(f)


    # build valid set
    valid = TrainingTextIterator(valid_datasets[0], valid_datasets[1],
                                 dictionaries[0], dictionaries[1], dictionary_chunk,
                                 n_words_source=options['n_words_src'], n_words_target=options['n_words'],
                                 batch_size=options['batch_size'],
                                 max_chunk_len=options['maxlen_chunk'], max_word_len=options['maxlen_chunk_words'])


    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    trng, use_noise, \
    x, x_mask, y_chunk, y_mask, y_cw, y_chunk_indicator, \
    opt_ret, \
    cost, cost_cw= \
        build_model(tparams, options)


    inps = [x, x_mask, y_chunk, y_mask, y_cw, y_chunk_indicator]



    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=False)
    f_log_probs_cw = theano.function(inps, cost_cw, profile=False)
    print 'Done'

    valid_errs, valid_errs_cw = pred_probs(f_log_probs, f_log_probs_cw, prepare_training_data,
                                            options, valid)

    valid_err = valid_errs.mean()
    valid_err_cw = valid_errs_cw.mean()

    with open(result_file, 'w') as result_file:
        print >> result_file, valid_err, valid_err_cw



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('pklmodel', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('dictionary_chunk', type=str)
    parser.add_argument('valid_source', type=str)
    parser.add_argument('valid_target', type=str)
    parser.add_argument('result_file', type=str)

    args = parser.parse_args()

    main(args.model,
         args.pklmodel,
         valid_datasets=[args.valid_source, args.valid_target],
         dictionaries=[args.dictionary, args.dictionary_target],
         dictionary_chunk=args.dictionary_chunk,
         result_file=args.result_file)
