'''
Translates a source file using a translation model.
'''
import argparse

import numpy
import theano
import cPickle as pkl
from training_data_iterator import TrainingTextIterator

from nmt import (build_sampler, gen_sample, load_params,
                 init_params, init_tparams, build_alignment, prepare_training_data)


from multiprocessing import Process, Queue


def main(model, pklmodel, dictionary, dictionary_target,dictionary_chunk, source_file,target_file, saveto, ck=5, wk=5, k=20,
         normalize=False, n_process=5, chr_level=False, jointProb=False, show_boundary=False):
    print 'load model model_options'
    with open('%s' % pklmodel, 'rb') as f:
        options = pkl.load(f)

    print 'load source dictionary and invert'
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    print 'load target dictionary and invert'
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'



    # dict for chunk label
    worddict_chunk = [None]
    worddict_r_chunk = [None]
    with open(dictionary_chunk, 'rb') as f:
        worddict_chunk = pkl.load(f)
    worddict_r_chunk = dict()
    for kk, vv in worddict_chunk.iteritems():
        worddict_r_chunk[vv] = kk


    def _seqs2wordsByChunk(caps, boundary, chunk, dictionary):
        capsw = []
        for cc, bb, ch in zip(caps, boundary, chunk):
            if cc == 0:
                continue
            # if w == -10000:
            #     ww.append('| NOTEND')
            #     continue
            if cc < 0:
                # ww.append('|' +  str(w))
                continue


            if bb == 0:

                capsw[-1] = capsw[-1] + "_" + (dictionary[cc])

            else:
                capsw.append(dictionary[cc])


        return capsw


    # output in the chunk format:
    # w1, POS, chunk_boundary-chunk_tag
    def _seqs2wordsByChunkFormat(caps, boundary, chunk, dictionary, chunk_dic):
        capsw = []
        current_tag = ''

        for cc, bb, ch in zip(caps, boundary, chunk):
            if cc == 0:
                continue
            # if w == -10000:
            #     ww.append('| NOTEND')
            #     continue
            if cc < 0:
                # ww.append('|' +  str(w))
                continue


            if bb == 0:

                capsw.append(dictionary[cc] + ' ' + 'I-'+chunk_dic[ch])

            else:
                capsw.append(dictionary[cc] + ' ' + 'B-'+chunk_dic[ch])


        return capsw


    # utility function
    def _seqs2words(caps, dictionary):
        capsw = []
        ww = []
        for w in caps:
            if w == 0:
                continue
            ww.append(dictionary[w])
        return ww




    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)


    f_align = build_alignment(tparams, options)


    # begin to read by iterators
    train = TrainingTextIterator(source_file, target_file,
                                 dictionary, dictionary_target, dictionary_chunk,
                                 n_words_source=30000, n_words_target=30000,
                                 batch_size=1,
                                 max_chunk_len=50, max_word_len=10000)


    boundary_right = 0.0
    tag_right = 0.0

    boundary_total = 0.0
    tag_total = 0.0

    for x, y_chunk, y_cw in train:

        x, x_mask, y_c, y_cw, chunk_indicator, y_mask = \
            prepare_training_data(x,
                                  y_chunk,
                                  y_cw,
                                  maxlen_chunk=100000,
                                  maxlen_cw=100000,
                                  n_words_src=30000,
                                  n_words=30000)




        align, chunk_tag, chunk_boundary = f_align(x, x_mask, y_c, y_cw, y_mask, chunk_indicator)


        x = x.reshape((x.shape[0],) )
        y_cw = y_cw.reshape((y_cw.shape[0],) )
        y_c = y_c.reshape((y_c.shape[0],) )
        chunk_indicator = chunk_indicator.reshape((chunk_indicator.shape[0],))


        print '\n'.join(_seqs2wordsByChunkFormat(numpy.ndarray.tolist(y_cw),
                                          numpy.ndarray.tolist(chunk_boundary),
                                          numpy.ndarray.tolist(chunk_tag),
                                          word_idict_trg, worddict_r_chunk))

        for gold_boundary, gold_chunk_tag, predict_boundary, predict_chunk_tag in zip(numpy.ndarray.tolist(chunk_indicator),
                                                                                      numpy.ndarray.tolist(y_c),
                                                                                      numpy.ndarray.tolist(chunk_boundary),
                                                                                      numpy.ndarray.tolist(chunk_tag)):
            boundary_total += 1
            tag_total += 1

            if gold_boundary == predict_boundary:
                boundary_right += 1

                if gold_chunk_tag == predict_chunk_tag:
                    tag_right += 1


        # for tag, boundary in zip(numpy.ndarray.tolist(chunk_tag), numpy.ndarray.tolist(chunk_boundary)):
        #     print
        #
        # # filter alignment
        # filter_align = []
        # for b, align in zip(numpy.ndarray.tolist(chunk_indicator), numpy.ndarray.tolist(align[0])):
        #     if b == 1.0:
        #         filter_align.append(align)
        #
        #
        # print 'align =',
        # # a = numpy.ndarray.tolist(filter_align)
        # a = numpy.array(filter_align)
        # a = numpy.transpose(a)
        # a = numpy.ndarray.tolist(a)
        #
        # print a







    print 'boundary prec: ', boundary_right / boundary_total
    print 'tag prec: ', tag_right / tag_total
    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ck', type=int, default=3)
    parser.add_argument('-wk', type=int, default=5)
    parser.add_argument('-k', type=int, default=8)
    parser.add_argument('-p', type=int, default=5)
    parser.add_argument('-n', action="store_true", default=False)
    parser.add_argument('-jointProb', action="store_true", default=False)
    parser.add_argument('-c', action="store_true", default=False)
    parser.add_argument('-show_boundary', action="store_true", default=False)
    parser.add_argument('model', type=str)
    parser.add_argument('pklmodel', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('dictionary_chunk', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args.model, args.pklmodel, args.dictionary, args.dictionary_target,args.dictionary_chunk, args.source,args.target,
         args.saveto, ck=args.ck, wk=args.wk, normalize=args.n, n_process=args.p,
         chr_level=args.c, jointProb=args.jointProb, show_boundary=args.show_boundary)
