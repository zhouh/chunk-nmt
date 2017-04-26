'''
Translates a source file using a translation model.
'''
import argparse

import numpy
import theano
import cPickle as pkl

from nmt import (build_sampler, gen_sample, load_params,
                 init_params, init_tparams)

from multiprocessing import Process, Queue


def main(model, pklmodel, dictionary, dictionary_target, source_file, saveto, ck=5, wk=5, k=20,
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

    # utility function
    def _seqs2words(caps, boundary, chunk):
        capsw = []
        for cc, bb, ch in zip(caps, boundary, chunk):
            ww = []
            for w, b, c in zip(cc, bb, ch):
                if w == 0:
                    continue
                # if w == -10000:
                #     ww.append('| NOTEND')
                #     continue
                elif w < 0:
                    # ww.append('|' +  str(w))
                    continue

                if show_boundary:
                    if b == 1.0:
                        ww.append('|')
                ww.append(word_idict_trg[w])
            capsw.append(' '.join(ww))
        return capsw

    def _seqs2wordsByChunk(caps, boundary, chunk):
        capsw = []
        for cc, bb, ch in zip(caps, boundary, chunk):
            ww = []
            for w, b, c in zip(cc, bb, ch):
                if w == 0:
                    continue
                # if w == -10000:
                #     ww.append('| NOTEND')
                #     continue
                elif w < 0:
                    # ww.append('|' +  str(w))
                    continue

                if b == 1.0:
                    ww.append('| ' + str(c))
                ww.append(word_idict_trg[w])
            capsw.append(' '.join(ww))
        return capsw

    def _send_jobs(fname):
        retval = []
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                if chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()
                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words_src'] else 1, x)
                x += [0]
                retval.append(x)
        return retval

    print 'Translating ', source_file, '...'

    print 'look up table'
    n_samples = _send_jobs(source_file)

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    f_init, f_next_chunk, f_next_word = build_sampler(tparams, options, trng, use_noise)

    def _translate(seq):

        be_stochastic = False
        # sample given an input sequence and obtain scores
        sample, boundary, chunk, score = gen_sample(tparams, f_init, f_next_chunk, f_next_word,
                                             numpy.array(seq).reshape([len(seq), 1]),
                                             options, trng=trng, maxlen=200, k_chunk=ck, k_word=wk, k=k,
                                             stochastic=be_stochastic, argmax=True, jointProb=False)

        if be_stochastic:
            return sample

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths

        # print 'score', score
        # print 'candidates', sample

        sidx = numpy.argmin(score)
        return sample[sidx], boundary[sidx], chunk[sidx]



    ys = []
    yb = []
    yc = []
    idx = 0
    for x in n_samples:
        y, y_boundary, y_chunk = _translate(x)
        ys.append(y)
        yb.append(y_boundary)
        yc.append(y_chunk)
        print idx
        idx += 1


    # print ys
    # print yb
    trans = _seqs2words(ys, yb, yc)
    trans_chunk = _seqs2wordsByChunk(ys, yb, yc)

    with open(saveto, 'w') as f:
        print >> f, '\n'.join(trans)
    with open(saveto+'chunk', 'w') as f:
        print >> f, '\n'.join(trans_chunk)
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
    parser.add_argument('source', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args.model, args.pklmodel, args.dictionary, args.dictionary_target, args.source,
         args.saveto, ck=args.ck, wk=args.wk, normalize=args.n, n_process=args.p,
         chr_level=args.c, jointProb=args.jointProb, show_boundary=args.show_boundary)
