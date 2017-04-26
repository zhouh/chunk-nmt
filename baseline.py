'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import ipdb
import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

from training_data_iterator import TrainingTextIterator
from data_iterator import TextIterator


profile = False


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def get_tensor_weight(n, nin, nout, scale=0.01):

    W = scale * numpy.random.randn(n, nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out



# batch preparation
def prepare_training_data(seqs_x, seqs_y_c, seqs_y_cw, maxlen_chunk=None, maxlen_cw=None, n_words_src=30000,
                 n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [ len(s) for s in seqs_y_cw]

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y_c = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    y_cw = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    chunk_indicator = numpy.zeros((maxlen_y, n_samples)).astype('float32')

    for idx, [s_x, s_y_c, s_y_cw] in enumerate(zip(seqs_x, seqs_y_c, seqs_y_cw)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        # print 'yc', y_c
        # print 'shape yc', y_c.shape
        # print 'idx', idx
        # print 'max', maxlen_y[idx]
        # print 'syc', s_y_c
        # print 'shape syc', s_y_c.shape
        y_c[:lengths_y[idx], idx] = s_y_c
        y_cw[:lengths_y[idx], idx] = s_y_cw
        y_mask[:lengths_y[idx]+1, idx] = 1.


        indicator_mask = [1 if cc != 1 else 0 for cc in s_y_c]

        # indicator here is a chunk begin or not (1 if True )
        chunk_indicator[:lengths_y[idx], idx] = indicator_mask



    # print y_cw

    return x, x_mask, y_c, y_cw, chunk_indicator, y_mask

# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=30000,
                 n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None, nin_chunk=None, dim_chunk_hidden=None, nin_nonlin_chunk=None):


    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim
    if nin_chunk is None:
        nin_chunk = nin
    if nin_nonlin_chunk is None:
        nin_nonlin_chunk = nin_chunk
    if dim_chunk_hidden is None:
        dim_chunk_hidden = dim


    chunk_label_num = options['n_chunks']

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[_p(prefix, 'Wx')] = Wx

    W_use_current_chunk = norm_weight(dim_chunk_hidden, dim)  # TODO the dimention here need to be careful
    params[_p(prefix, 'W_use_current_chunk')] = W_use_current_chunk


    W_current_chunk_c = norm_weight(dim_chunk_hidden, dim * 2)
    params[_p(prefix, 'W_current_chunk_c')] = W_current_chunk_c


    Ux = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U_nl')] = U_nl
    params[_p(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')

    Ux_nl = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux_nl')] = Ux_nl
    params[_p(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[_p(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[_p(prefix, 'Wc_att')] = Wc_att


    # attention: combined -> hidden
    W_cu_chunk_att = norm_weight(dim_chunk_hidden, dimctx)
    params[_p(prefix, 'W_cu_chunk_att')] = W_cu_chunk_att


    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att


    # new the chunking parameters


    params[_p(prefix, 'chunk_transform_matrix')] = get_tensor_weight(chunk_label_num, dim_nonlin, nin_chunk)


    W_chunk = numpy.concatenate([norm_weight(nin_chunk, dim_chunk_hidden),
                           norm_weight(nin_chunk, dim_chunk_hidden)], axis=1) # nin * 2 dim
    params[_p(prefix, 'W_chunk')] = W_chunk
    params[_p(prefix, 'b_chunk')] = numpy.zeros((2 * dim_chunk_hidden,)).astype('float32')

    U_chunk = numpy.concatenate([ortho_weight(dim_chunk_hidden),
                           ortho_weight(dim_chunk_hidden)], axis=1)
    params[_p(prefix, 'U_chunk')] = U_chunk

    Wx_chunk = norm_weight(nin_nonlin_chunk, dim_chunk_hidden)
    params[_p(prefix, 'Wx_chunk')] = Wx_chunk
    Ux_chunk = ortho_weight(dim_chunk_hidden)
    params[_p(prefix, 'Ux_chunk')] = Ux_chunk
    params[_p(prefix, 'bx_chunk')] = numpy.zeros((dim_chunk_hidden,)).astype('float32')

    U_nl_chunk = numpy.concatenate([ortho_weight(dim_chunk_hidden),
                              ortho_weight(dim_chunk_hidden)], axis=1)
    params[_p(prefix, 'U_nl_chunk')] = U_nl_chunk
    params[_p(prefix, 'b_nl_chunk')] = numpy.zeros((2 * dim_chunk_hidden,)).astype('float32')

    Ux_nl_chunk = ortho_weight(dim_chunk_hidden)
    params[_p(prefix, 'Ux_nl_chunk')] = Ux_nl_chunk
    params[_p(prefix, 'bx_nl_chunk')] = numpy.zeros((dim_chunk_hidden,)).astype('float32')

    # context to LSTM
    Wc_chunk = norm_weight(dimctx, dim_chunk_hidden*2)
    params[_p(prefix, 'Wc_chunk')] = Wc_chunk

    Wcx_chunk = norm_weight(dimctx, dim_chunk_hidden)
    params[_p(prefix, 'Wcx_chunk')] = Wcx_chunk

    # attention: combined -> hidden
    W_comb_att_chunk = norm_weight(dim_chunk_hidden, dimctx)
    params[_p(prefix, 'W_comb_att_chunk')] = W_comb_att_chunk

    # attention: context -> hidden
    Wc_att_chunk = norm_weight(dimctx)
    params[_p(prefix, 'Wc_att_chunk')] = Wc_att_chunk

    # attention: hidden bias
    b_att_chunk = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att_chunk')] = b_att_chunk

    # attention:
    U_att_chunk = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att_chunk')] = U_att_chunk
    c_att_chunk = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt_chunk')] = c_att_chunk



    return params


def gru_cond_layer(tparams, emb, chunk_index, options, prefix='gru',
                   mask=None, chunk_boundary_indicator=None, context=None,
                   one_step=False, one_step_chunk=False, one_step_word=False,
                   init_state_chunk=None,init_state_chunk_words=None,
                   current_chunk_hidden=None,last_chunk_end_word_hidden1=None, current_word_hidden1=None,
                   context_mask=None, **kwargs):


    assert context, 'Context must be provided'


    # nsteps = chunk_index.shape[0]

    if chunk_index is not None:
        nsteps = chunk_index.shape[0]

    if one_step_chunk:
        assert init_state_chunk, 'previous state must be provided'
        assert init_state_chunk_words, 'previous state must be provided'

    # if this is a sample or decode process, we may use a sample = 1 predict
    if emb is not None:
        if emb.ndim == 3:
            n_samples = emb.shape[1]
        else:
            n_samples = emb.shape[0]
    else:
        n_samples = current_word_hidden1.shape[0]


    # the hidden dim
    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # chunk hidden dim
    chunk_hidden_dim = tparams[_p(prefix, 'Wcx_chunk')].shape[1]

    # if mask is None, it is the sample process
    if mask is None:
        mask = tensor.alloc(1., n_samples, 1)

    # initial/previous state
    if init_state_chunk is None:
        init_state_chunk = tensor.alloc(0., n_samples, chunk_hidden_dim)
    if init_state_chunk_words is None:
        init_state_chunk_words = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    #
    # chunking prediction
    #
    # We have to firstly compute the word hidden 1 and then compute the
    # chunk hidden given the word hidden 1
    def predict_chunk(  m_, cw_x_, cw_xx_, h_chunk,
                        h_cw, h1_last_chunk_end_word,
                        # non sequences
                        pctx_chunk, cc,
                        chunk_transform_matrix, U_chunk, Wc_chunk, W_comb_att_chunk, U_att_chunk, c_tt_chunk,
                        Ux_chunk, Wcx_chunk, U_nl_chunk, Ux_nl_chunk, b_nl_chunk, bx_nl_chunk, Wx_chunk, bx_chunk,
                        W_chunk, b_chunk,
                        W_current_chunk_hidden, W_current_chunk_c, U, Wc, W_comb_att, W_cu_chunk_att, U_att, c_tt, Ux, Wcx, U_nl, Ux_nl,
                        b_nl, bx_nl):


        #
        # incorporate the last words into word hidden 1 : h1
        #
        preact1 = tensor.dot(h_cw, U)
        preact1 += cw_x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_cw, Ux)
        preactx1 *= r1
        preactx1 += cw_xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_cw + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_cw

        ret_word_hidden1 = h1

        ########### end compute word h1

        #
        # compute the chunk embedding ######################
        #
        last_chunk_emb = h1 - h1_last_chunk_end_word


        transform = chunk_transform_matrix[0]
        last_chunk_emb =  tensor.dot(last_chunk_emb, transform) # TODO, make sure that here the chunkindex is last chunk index


        #
        # compute the current chunk hidden
        #
        chunk_xx_ = tensor.dot(last_chunk_emb, Wx_chunk) + \
                    bx_chunk
        chunk_x_ = tensor.dot(last_chunk_emb, W_chunk) + \
                   b_chunk


        preact1 = tensor.dot(h_chunk, U_chunk)
        preact1 += chunk_x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, chunk_hidden_dim)
        u1 = _slice(preact1, 1, chunk_hidden_dim)

        preactx1 = tensor.dot(h_chunk, Ux_chunk)
        preactx1 *= r1
        preactx1 += chunk_xx_

        h1 = tensor.tanh(preactx1)


        h1 = u1 * h_chunk + (1. - u1) * h1


        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_chunk

        #
        # attention
        #
        pstate_ = tensor.dot(h1, W_comb_att_chunk)
        pctx__ = pctx_chunk + pstate_[None, :, :]
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att_chunk)+c_tt_chunk
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc * alpha[:, :, None]).sum(0)  # current context


        preact2 = tensor.dot(h1, U_nl_chunk)+b_nl_chunk
        preact2 += tensor.dot(ctx_, Wc_chunk)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, chunk_hidden_dim)
        u2 = _slice(preact2, 1, chunk_hidden_dim)

        preactx2 = tensor.dot(h1, Ux_nl_chunk)+bx_nl_chunk
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_, Wcx_chunk)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        chunk_hidden2 = h2
        chunk_ctx = ctx_
        chunk_alpha = alpha.T



        return ret_word_hidden1, last_chunk_emb, chunk_hidden2, chunk_ctx, chunk_alpha


    #
    # given word hidden1, chunk hidden, compute the
    # word hidden2
    #
    def predict_word_hidden2(m_, word_hidden1, chunk_hidden,
                             pctx_, cc_,
                             W_current_chunk_hidden, W_current_chunk_c, U, Wc, W_comb_att,
                             W_cu_chunk_att, U_att, c_tt, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl):


        m = tensor.alloc(0., chunk_hidden.shape[0], chunk_hidden.shape[1])

        chunk_hidden = m * chunk_hidden
        #
        # given the word hidden1 and chunk hidden , compute
        # word attention
        pstate_ = tensor.dot(word_hidden1, W_comb_att)
        pstate_chunk = tensor.dot(chunk_hidden, W_cu_chunk_att)



        ################
        # revise
        ################
        pctx__ = pctx_ + pstate_[None, :, :] + pstate_chunk[None, :, :]
        # pctx__ = pctx_ + pstate_[None, :, :]
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context



        preact2 = tensor.dot(word_hidden1, U_nl)+b_nl


        ################
        # revise
        ################
        preact2 += tensor.dot(ctx_, Wc) + tensor.dot(chunk_hidden, W_current_chunk_c)
        # preact2 += tensor.dot(ctx_, Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(word_hidden1, Ux_nl)+bx_nl
        preactx2 *= r2


        ################
        # revise
        ################
        preactx2 += tensor.dot(ctx_, Wcx) + tensor.dot(chunk_hidden, W_current_chunk_hidden) # here we add current chunk representation
        # preactx2 += tensor.dot(ctx_, Wcx) # here we add current chunk representation


        h2 = tensor.tanh(preactx2)

        h2 = u2 * word_hidden1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * word_hidden1

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u



    def scan_step(  # seq
                    m_, chunk_boundary, cw_x_, cw_xx_,
                    # outputs info
                    h_chunk, position_chunk_hidden2, ctx_chunk, alpha_chunk, chunk_true,
                    h_cw, position_h1, h1_last_chunk_end_word, ctx_cw, alpha_cw,
                    # non sequences
                    pctx_chunk, pctx_cw, cc,
                    chunk_transform_matrix, U_chunk, Wc_chunk, W_comb_att_chunk, U_att_chunk, c_tt_chunk,
                    Ux_chunk, Wcx_chunk, U_nl_chunk, Ux_nl_chunk, b_nl_chunk, bx_nl_chunk, Wx_chunk, bx_chunk,
                    W_chunk, b_chunk,
                    W_current_chunk_hidden, W_current_chunk_c, U, Wc, W_comb_att, W_cu_chunk_att, U_att, c_tt, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):


        word_hidden1, \
        last_chunk_emb, \
        current_position_hypo_chunk_hidden, \
        chunk_ctx, chunk_alpha = \
            predict_chunk(m_, cw_x_, cw_xx_, h_chunk,
                          h_cw, h1_last_chunk_end_word,
                          pctx_chunk, cc,
                          chunk_transform_matrix, U_chunk, Wc_chunk, W_comb_att_chunk, U_att_chunk, c_tt_chunk,
                          Ux_chunk, Wcx_chunk, U_nl_chunk, Ux_nl_chunk, b_nl_chunk, bx_nl_chunk, Wx_chunk, bx_chunk,
                          W_chunk, b_chunk,
                          W_current_chunk_hidden, W_current_chunk_c, U, Wc, W_comb_att, W_cu_chunk_att, U_att, c_tt, Ux,
                          Wcx, U_nl, Ux_nl, b_nl, bx_nl)


        #
        # if current chunk indocator is 1, then this is a begin of a new chunk,
        # the chunk hidden is cuttent chunk hidde, otherwise, this word is still in the old chunk
        # the last chunk hidden will still be used.
        #
        chunk_hidden = chunk_boundary[:, None] * current_position_hypo_chunk_hidden \
                       + (1. - chunk_boundary)[:, None] * h_chunk

        h1_last_chunk_end_word = chunk_boundary[:, None] * word_hidden1 \
                                 + (1. - chunk_boundary)[:, None] * h1_last_chunk_end_word


        word_hidden2, \
        word_ctx_, \
        word_alpha = predict_word_hidden2(m_, word_hidden1, chunk_hidden,
                                          pctx_cw, cc,
                                          W_current_chunk_hidden, W_current_chunk_c, U, Wc, W_comb_att,
                                          W_cu_chunk_att, U_att, c_tt, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl)



        return chunk_hidden, current_position_hypo_chunk_hidden, chunk_ctx, chunk_alpha, last_chunk_emb, \
               word_hidden2, word_hidden1, h1_last_chunk_end_word, word_ctx_, word_alpha



    _step = scan_step

    word_shared_vars = [tparams[_p(prefix, 'W_use_current_chunk')],
                        tparams[_p(prefix, 'W_current_chunk_c')],
                        tparams[_p(prefix, 'U')],
                        tparams[_p(prefix, 'Wc')],
                        tparams[_p(prefix, 'W_comb_att')],
                        tparams[_p(prefix, 'W_cu_chunk_att')],
                        tparams[_p(prefix, 'U_att')],
                        tparams[_p(prefix, 'c_tt')],
                        tparams[_p(prefix, 'Ux')],
                        tparams[_p(prefix, 'Wcx')],
                        tparams[_p(prefix, 'U_nl')],
                        tparams[_p(prefix, 'Ux_nl')],
                        tparams[_p(prefix, 'b_nl')],
                        tparams[_p(prefix, 'bx_nl')]]

    chunk_shared_vars = [tparams[_p(prefix, 'chunk_transform_matrix')],
                         tparams[_p(prefix, 'U_chunk')],
                         tparams[_p(prefix, 'Wc_chunk')],
                         tparams[_p(prefix, 'W_comb_att_chunk')],
                         tparams[_p(prefix, 'U_att_chunk')],
                         tparams[_p(prefix, 'c_tt_chunk')],
                         tparams[_p(prefix, 'Ux_chunk')],
                         tparams[_p(prefix, 'Wcx_chunk')],
                         tparams[_p(prefix, 'U_nl_chunk')],
                         tparams[_p(prefix, 'Ux_nl_chunk')],
                         tparams[_p(prefix, 'b_nl_chunk')],
                         tparams[_p(prefix, 'bx_nl_chunk')],
                         tparams[_p(prefix, 'Wx_chunk')],
                         tparams[_p(prefix, 'bx_chunk')],
                         tparams[_p(prefix, 'W_chunk')],
                         tparams[_p(prefix, 'b_chunk')]]

    # compute the word hidden1 and chunk hidden during sample
    if one_step_chunk:

        chunk_pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att_chunk')]) + \
                      tparams[_p(prefix, 'b_att_chunk')]


        # projected x
        state_belowx = tensor.dot(emb, tparams[_p(prefix, 'Wx')]) + \
                       tparams[_p(prefix, 'bx')]
        state_below_ = tensor.dot(emb, tparams[_p(prefix, 'W')]) + \
                       tparams[_p(prefix, 'b')]



        seqs = [mask, state_below_, state_belowx, init_state_chunk,
                init_state_chunk_words, last_chunk_end_word_hidden1]
        rval = predict_chunk(*(seqs + [chunk_pctx_, context] +
                               chunk_shared_vars + word_shared_vars))
        return rval[0], rval[1], rval[2], rval[3], rval[4], None, None, None, None, None
        # ret_word_hidden1, last_chunk_emb, chunk_hidden2, chunk_ctx, chunk_alpha

    # given the word hidden1 and chunk hidden, compute the word hidden 2
    elif one_step_word:

        # word pctx
        pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) + \
                tparams[_p(prefix, 'b_att')]

        seqs = [mask, current_word_hidden1, current_chunk_hidden,
                pctx_, context]

        rval = predict_word_hidden2(*(seqs + word_shared_vars ))
        return rval[0], rval[1], rval[2], None, None, None, None, None, None, None
        # word hidden2, word ctx, word attention alpha


    # word pctx
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) + \
            tparams[_p(prefix, 'b_att')]

    # chunk pctx
    chunk_pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att_chunk')]) + \
                  tparams[_p(prefix, 'b_att_chunk')]



    # projected x
    state_belowx = tensor.dot(emb, tparams[_p(prefix, 'Wx')]) + \
                   tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(emb, tparams[_p(prefix, 'W')]) + \
                   tparams[_p(prefix, 'b')]

    # the sequence is
    # @mask the word mask for batch training
    # @chunk_boundary_indicator 1: this is a begin of a chunk, 0: this is a inter part of a chunk
    # @state_below_ W*y_emb
    # @state_belowx W*y_emb, for different usage
    seqs = [mask, chunk_boundary_indicator, state_below_, state_belowx]



    # outputs_info of the training scan process
    init_chunk_ctx = tensor.alloc(0., n_samples, context.shape[2])
    init_chunk_alpha = tensor.alloc(0., n_samples, context.shape[0])
    h1_last_chunk_end_word =  tensor.alloc(0., n_samples, dim) # set last chunk hidden 0
    position_h1 =  tensor.alloc(0., n_samples, dim)
    last_chunk_emb =  tensor.alloc(0., n_samples, options['dim_chunk'])

    init_word_ctx = tensor.alloc(0., n_samples, context.shape[2])
    init_word_alpha = tensor.alloc(0., n_samples, context.shape[0])


    # chunk_hidden, current_position_hypo_chunk_hidden, chunk_ctx, chunk_alpha, last_chunk_emb, \
    #           word_hidden1, word_hidden2, h1_last_chunk_end_word, word_ctx_, word_alpha

    outputs = [init_state_chunk,
               init_state_chunk, # only for output
               init_chunk_ctx,
               init_chunk_alpha,
               last_chunk_emb,
               init_state_chunk_words,
               position_h1,   # current position computed word hidden1
               h1_last_chunk_end_word,
               init_word_ctx,
               init_word_alpha]


    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=outputs,
                                # here pctx is the tranformation of the source context
                                non_sequences=[chunk_pctx_, pctx_, context]+chunk_shared_vars+word_shared_vars,
                                name=_p(prefix, '_layers'),
                                #n_steps=n_chunk_step,
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)

    return rval
    # chunk_hidden, chunk_ctx, chunk_alpha, word_hidden2, h1_last_chunk_end_word, word_ctx_, word_alpha




# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])

    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

    # encoder: bidirectional RNN
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder_r',
                                              nin=options['dim_word'],
                                              dim=options['dim'])


    ctxdim = 2 * options['dim']


    #
    # generate the initial hidden representation for word and chunk
    #

    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix='ff_state_chunk',
                                nin=ctxdim, nout=options['dim_chunk_hidden'])


    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix='ff_state_chunk_words',
                                nin=ctxdim, nout=options['dim'])



    # decoder
    params = get_layer(options['decoder'])[0](options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              dimctx=ctxdim,
                                              nin_chunk=options['dim_chunk'],
                                              dim_chunk_hidden=options['dim_chunk_hidden'])


    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_using_chunk_hidden',
                                nin=options['dim_chunk_hidden'], nout=options['dim_word'],
                                ortho=False)
    # params = get_layer('ff')[0](options, params, prefix='ff_logit_chunk_hidden',
    #                             nin=ctxdim, nout=options['dim_word'],
    #                             ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'])

    # readout

    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm_chunk',
                                nin=options['dim_chunk_hidden'], nout=options['dim_chunk'],
                                ortho=False)

    # we should note here, we use word dim
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev_chunk',
                                nin=options['dim_chunk'],
                                nout=options['dim_chunk'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx_chunk',
                                nin=ctxdim, nout=options['dim_chunk'],
                                ortho=False)

    params = get_layer('ff')[0](options, params, prefix='logit_ctx_last_word',
                                nin=options['dim_word'],
                                nout=options['dim_chunk'],
                                ortho=False)

    params = get_layer('ff')[0](options, params, prefix='logit_ctx_current_word_hidden1',
                                nin=options['dim'],
                                nout=options['dim_chunk'],
                                ortho=False)


    params = get_layer('ff')[0](options, params, prefix='ff_logit_chunk',
                                nin=options['dim_chunk'],
                                nout=options['dim_chunk'],
                                ortho=False)


    return params


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')

    y_chunk = tensor.matrix('y_chunk', dtype='int64')
    y_chunk_words = tensor.matrix('y_chunk_words', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')
    chunk_indicator = tensor.matrix('chunk_indicator', dtype='float32')

    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[0]
    n_timesteps_y = y_chunk.shape[0]
    n_samples = x.shape[1]

    # word embedding for forward rnn (source)
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask)
    # word embedding for backward rnn (source)
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r',
                                             mask=xr_mask)

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state for both
    init_state_chunk = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state_chunk', activ='tanh')
    init_state_chunk_words = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state_chunk_words', activ='tanh')

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.


    # shift the word embeddings in the chunk
    emb = tparams['Wemb_dec'][y_chunk_words.flatten()]
    emb = emb.reshape([n_timesteps_y, n_samples, options['dim_word']])

    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    y_chunk_shift = tensor.zeros_like(y_chunk)
    y_chunk_shift = tensor.set_subtensor(y_chunk_shift[1:], y_chunk[:-1])

    #
    # decoder
    chunk_hidden, \
    current_position_hypo_chunk_hidden, \
    chunk_ctx, \
    chunk_alpha, \
    last_chunk_emb, \
    word_hidden2, \
    word_hidden1, \
    h1_last_chunk_end_word, \
    word_ctx_, \
    word_alpha = get_layer(options['decoder'])[1](tparams, emb, y_chunk_shift,
                                                  options,
                                                  prefix='decoder',
                                                  mask=y_mask,
                                                  chunk_boundary_indicator=chunk_indicator,
                                                  context=ctx,
                                                  context_mask=x_mask,
                                                  init_state_chunk=init_state_chunk,
                                                  init_state_chunk_words=init_state_chunk_words)
    #
    opt_ret['dec_alphas_chunk'] = chunk_alpha


    logit_lstm_chunk = get_layer('ff')[1](tparams, current_position_hypo_chunk_hidden, options,
                                    prefix='ff_logit_lstm_chunk', activ='linear')
    logit_prev_chunk = get_layer('ff')[1](tparams, last_chunk_emb, options,
                                    prefix='ff_logit_prev_chunk', activ='linear')
    logit_ctx_chunk = get_layer('ff')[1](tparams, chunk_ctx, options,
                                   prefix='ff_logit_ctx_chunk', activ='linear')

    logit_ctx_last_word = get_layer('ff')[1](tparams, emb, options,
                                   prefix='logit_ctx_last_word', activ='linear')
    logit_ctx_current_word_hidden1 = get_layer('ff')[1](tparams, word_hidden1, options,
                                   prefix='logit_ctx_current_word_hidden1', activ='linear')



    logit_chunk = tensor.tanh(logit_lstm_chunk+logit_prev_chunk+logit_ctx_chunk+logit_ctx_last_word+logit_ctx_current_word_hidden1)

    if options['use_dropout']:
        logit_chunk = dropout_layer(logit_chunk, use_noise, trng)
    logit_chunk = get_layer('ff')[1](tparams, logit_chunk, options,
                               prefix='ff_logit_chunk', activ='linear')
    logit_shp_chunk = logit_chunk.shape
    probs_chunk = tensor.nnet.softmax(logit_chunk.reshape([logit_shp_chunk[0]*logit_shp_chunk[1],
                                               logit_shp_chunk[2]]))

    # cost
    y_flat_chunk = y_chunk.flatten()
    y_flat_idx_chunk = tensor.arange(y_flat_chunk.shape[0]) * options['n_chunks'] + y_flat_chunk
    cost = -tensor.log(probs_chunk.flatten()[y_flat_idx_chunk])
    cost = cost.reshape([y_chunk.shape[0], y_chunk.shape[1]])


    m = tensor.alloc(0., y_mask.shape[0], y_mask.shape[1])
    cost = m * cost


    # weights (alignment matrix)
    opt_ret['dec_alphas_cw'] = word_ctx_

    # compute word probabilities
    logit_lstm_cw = get_layer('ff')[1](tparams, word_hidden2, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev_cw = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx_cw = get_layer('ff')[1](tparams, word_ctx_, options,
                                   prefix='ff_logit_ctx', activ='linear')


    logit_ctx_using_current_chunk_hidden = get_layer('ff')[1](tparams, chunk_hidden, options,
                                  prefix='ff_logit_using_chunk_hidden', activ='linear')

    m = tensor.alloc(0., logit_ctx_using_current_chunk_hidden.shape[0], logit_ctx_using_current_chunk_hidden.shape[1], logit_ctx_using_current_chunk_hidden.shape[2])

    logit_ctx_using_current_chunk_hidden = m * logit_ctx_using_current_chunk_hidden


    logit_cw = tensor.tanh(logit_lstm_cw+logit_prev_cw+logit_ctx_cw+logit_ctx_using_current_chunk_hidden)
    # logit_cw = tensor.tanh(logit_lstm_cw+logit_prev_cw+logit_ctx_cw)

    if options['use_dropout']:
        logit_cw = dropout_layer(logit_cw, use_noise, trng)
    logit_cw = get_layer('ff')[1](tparams, logit_cw, options,
                               prefix='ff_logit', activ='linear')
    logit_shp_cw = logit_cw.shape
    probs_cw = tensor.nnet.softmax(logit_cw.reshape([logit_shp_cw[0]*logit_shp_cw[1],
                                               logit_shp_cw[2]]))

    # cost
    y_flat_cw = y_chunk_words.flatten()
    y_flat_idx_cw = tensor.arange(y_flat_cw.shape[0]) * options['n_words'] + y_flat_cw

    cost_cw = -tensor.log(probs_cw.flatten()[y_flat_idx_cw])
    cost_cw = cost_cw.reshape([y_chunk_words.shape[0], y_chunk_words.shape[1]])


    cost = cost + cost_cw
    # cost = cost_cw
    cost = (cost * y_mask).sum(0)

    return trng, use_noise, x, x_mask, y_chunk, y_mask, y_chunk_words, chunk_indicator,\
           opt_ret, cost, cost_cw

# build a sampler
def build_sampler(tparams, options, trng, use_noise):


    x = tensor.matrix('x', dtype='int64')

    xr = x[::-1]

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source), forward and backward
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder')
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r')

    # concatenate forward and backward rnn hidden states
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state for both
    init_state_chunk = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state_chunk', activ='tanh')
    init_state_chunk_words = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state_chunk_words', activ='tanh')


    print 'Building f_init...',
    outs = [init_state_chunk, init_state_chunk_words, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print 'Done'




    #
    # build predict word hidden 1 and chunk hidden2
    #

    # TODO note that here the y_chunk and y_chunk_words are both vector, because it only conduct one steps!
    # y_chunk = tensor.vector('y_sample_chunk', dtype='int64')
    y_chunk_words = tensor.vector('y_sample_chunk_words', dtype='int64')

    chunk_boundary = tensor.vector('chunk_boundary', dtype='float32')

    init_state_chunk = tensor.matrix('init_state_chunk', dtype='float32')
    init_state_chunk_words = tensor.matrix('init_state_chunk_words', dtype='float32')

    last_chunk_end_word_hidden1 = tensor.matrix('last_chunk_end_word_hidden1', dtype='float32')


    current_chunk_hidden = tensor.matrix('current_chunk_hidden', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb_chunk_word = tensor.switch(y_chunk_words[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y_chunk_words])



    #
    # decoder
    #
    retval_predict_chunk = get_layer(options['decoder'])[1](tparams,
                                                            emb_chunk_word,
                                                            None,
                                                            options,
                                                            prefix='decoder',
                                                            context=ctx,
                                                            one_step=True,
                                                            one_step_word=False,
                                                            one_step_chunk=True,
                                                            init_state_chunk=init_state_chunk,
                                                            init_state_chunk_words=init_state_chunk_words,
                                                            last_chunk_end_word_hidden1=last_chunk_end_word_hidden1)
    word_hidden1 = retval_predict_chunk[0]
    last_chunk_emb = retval_predict_chunk[1]
    current_position_hypo_chunk_hidden = retval_predict_chunk[2]
    chunk_ctx = retval_predict_chunk[3]
    chunk_alpha = retval_predict_chunk[4]


    #
    # get the chunk prediction
    #
    logit_lstm_chunk = get_layer('ff')[1](tparams, current_position_hypo_chunk_hidden, options,
                                    prefix='ff_logit_lstm_chunk', activ='linear')
    logit_prev_chunk = get_layer('ff')[1](tparams, last_chunk_emb, options,
                                    prefix='ff_logit_prev_chunk', activ='linear')
    logit_ctx_chunk = get_layer('ff')[1](tparams, chunk_ctx, options,
                                   prefix='ff_logit_ctx_chunk', activ='linear')

    logit_ctx_last_word = get_layer('ff')[1](tparams, emb_chunk_word, options,
                                   prefix='logit_ctx_last_word', activ='linear')
    logit_ctx_current_word_hidden1 = get_layer('ff')[1](tparams, word_hidden1, options,
                                   prefix='logit_ctx_current_word_hidden1', activ='linear')



    logit_chunk = tensor.tanh(logit_lstm_chunk+logit_prev_chunk+logit_ctx_chunk+logit_ctx_last_word+logit_ctx_current_word_hidden1)

    if options['use_dropout']:
        logit_chunk = dropout_layer(logit_chunk, use_noise, trng)
    logit_chunk = get_layer('ff')[1](tparams, logit_chunk, options,
                               prefix='ff_logit_chunk', activ='linear')
    probs_chunk = tensor.nnet.softmax(logit_chunk)

    next_sample_chunk = trng.multinomial(pvals=probs_chunk).argmax(1)


    print 'Building f_next_chunk..'
    inps = [y_chunk_words, ctx, init_state_chunk, init_state_chunk_words, last_chunk_end_word_hidden1]
    outs = [probs_chunk, next_sample_chunk, word_hidden1, current_position_hypo_chunk_hidden]
    f_next_chunk = theano.function(inps, outs, name='f_next_chunk', profile=profile)
    print 'End Building f_next_chunk..'








    #
    # begin to predict the word hidden2
    #


    chunk_boundary = tensor.vector('chunk_boundary', dtype='float32')
    current_chunk_hidden = tensor.matrix('current_chunk_hidden', dtype='float32')
    current_position_hypo_chunk_hidden = tensor.matrix('current_position_hypo_chunk_hidden', dtype='float32')
    word_hidden1 = tensor.matrix('word_hidden1', dtype='float32')
    last_chunk_end_word_hidden1 = tensor.matrix('last_chunk_end_word_hidden1', dtype='float32')



    # given the chunk indicator, compute the word hidden2
    chunk_hidden = chunk_boundary[:, None] * current_position_hypo_chunk_hidden \
                   + (1. - chunk_boundary)[:, None] * current_chunk_hidden

    h1_last_chunk_end_word = chunk_boundary[:, None] * word_hidden1 \
                             + (1. - chunk_boundary)[:, None] * last_chunk_end_word_hidden1


    #
    # decoder for word hidden2
    #
    retval_predict_chunk = get_layer(options['decoder'])[1](tparams,
                                                            None,
                                                            None,
                                                            options,
                                                            prefix='decoder',
                                                            context=ctx,
                                                            one_step=True,
                                                            one_step_word=True,
                                                            one_step_chunk=False,
                                                            current_chunk_hidden=chunk_hidden,
                                                            current_word_hidden1=word_hidden1)


    word_hidden2 = retval_predict_chunk[0]
    word_ctx = retval_predict_chunk[1]
    word_alpha = retval_predict_chunk[2]


    # compute word probabilities
    logit_lstm_cw = get_layer('ff')[1](tparams, word_hidden2, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev_cw = get_layer('ff')[1](tparams, emb_chunk_word, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx_cw = get_layer('ff')[1](tparams, word_ctx, options,
                                   prefix='ff_logit_ctx', activ='linear')


    logit_ctx_using_current_chunk_hidden = get_layer('ff')[1](tparams, chunk_hidden, options,
                                  prefix='ff_logit_using_chunk_hidden', activ='linear')


    m = tensor.alloc(0., logit_ctx_using_current_chunk_hidden.shape[0], logit_ctx_using_current_chunk_hidden.shape[1])

    logit_ctx_using_current_chunk_hidden = m * logit_ctx_using_current_chunk_hidden



    logit_cw = tensor.tanh(logit_lstm_cw+logit_prev_cw+logit_ctx_cw+logit_ctx_using_current_chunk_hidden)

    if options['use_dropout']:
        logit_cw = dropout_layer(logit_cw, use_noise, trng)
    logit_cw = get_layer('ff')[1](tparams, logit_cw, options,
                               prefix='ff_logit', activ='linear')
    probs_cw = tensor.nnet.softmax(logit_cw)
    next_sample_cw = trng.multinomial(pvals=probs_cw).argmax(1)




    # sample from softmax distribution to get the sample
    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next_word..'
    inps = [y_chunk_words,
            ctx,
            chunk_boundary,
            current_chunk_hidden,
            current_position_hypo_chunk_hidden,
            word_hidden1,
            last_chunk_end_word_hidden1]
    outs = [probs_cw, next_sample_cw, word_hidden2, h1_last_chunk_end_word, chunk_hidden]
    f_next_chunk_word = theano.function(inps, outs, name='f_next_chunk_word', profile=profile)
    print 'Done'

    return f_init, f_next_chunk, f_next_chunk_word




# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams, f_init, f_next_chunk, f_next_word, x,
               options, trng=None, k_chunk=1, k_word=1, k=5, maxlen=50,
               stochastic=True, argmax=False, jointProb=True):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []
    hyp_chunk_states = []
    hyp_last_chunk_last_word_hidden1 = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(x)
    next_state_chunk, next_state_word, ctx0 = ret[0], ret[1], ret[2]
    last_chunk_last_word_hidden1 = numpy.zeros((1, options['dim'])).astype('float32')


    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    # next_chunk = -1 * numpy.ones((1,)).astype('int64')  # bos indicator
    #
    # word_hidden1 = None


    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        inps = [next_w,
                ctx,
                next_state_chunk,
                next_state_word,
                last_chunk_last_word_hidden1]
        ret = f_next_chunk(*inps)
        next_chunk_p, next_chunk, word_hidden1, hypo_chunk_hidden = ret[0], ret[1], ret[2], ret[3]


        # get the chunk boundrary indocator
        next_chunk = next_chunk_p.argmax(1)
        chunk_boundary = numpy.zeros((next_chunk.shape[0],)).astype('float32')

        for i in xrange(next_chunk.shape[0]):
            if next_chunk[i] != 1:
                chunk_boundary[i] = 1.0

        inps = [next_w,
                ctx,
                chunk_boundary,
                next_state_chunk,
                hypo_chunk_hidden,
                word_hidden1,
                last_chunk_last_word_hidden1]

        ret = f_next_word(*inps)

        next_word_p, \
        next_w, \
        next_state_word, \
        last_chunk_last_word_hidden1, \
        next_state_chunk \
            = ret[0], ret[1], ret[2], ret[3], ret[4]


        if jointProb:
            indicator_score = next_chunk_p.max(1)
            indicator_score = indicator_score.reshape(indicator_score.shape[0], 1)
            next_word_p = indicator_score * next_word_p


        if stochastic:
            if argmax:
                nw = next_word_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score -= numpy.log(next_word_p[0, nw])
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - numpy.log(next_word_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_word_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            new_hyp_chunk_states = []
            new_hyp_last_chunk_last_word_hidden1 = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state_word[ti]))
                new_hyp_chunk_states.append(copy.copy(next_state_chunk[ti]))
                new_hyp_last_chunk_last_word_hidden1.append(copy.copy(last_chunk_last_word_hidden1[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            hyp_chunk_states = []
            hyp_last_chunk_last_word_hidden1 = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    hyp_chunk_states.append(new_hyp_chunk_states[idx])
                    hyp_last_chunk_last_word_hidden1.append(new_hyp_last_chunk_last_word_hidden1[idx])

            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state_word = numpy.array(hyp_states)
            next_state_chunk = numpy.array(hyp_chunk_states)
            last_chunk_last_word_hidden1 = numpy.array(hyp_last_chunk_last_word_hidden1)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True):
    probs = []

    n_done = 0

    for x, y_chunk, y_cw in iterator:
        n_done += len(x)

        x, x_mask, y_c, y_cw, chunk_indicator, y_mask = prepare_data(x, y_chunk, y_cw,
                                            n_words_src=options['n_words_src'],
                                            n_words=options['n_words'])

        pprobs = f_log_probs(x, x_mask, y_c, y_mask, y_cw, chunk_indicator)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)


# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost, beta1=0.9, beta2=0.999, e=1e-8):

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

    updates = []

    t_prev = theano.shared(numpy.float32(0.))
    t = t_prev + 1.
    lr_t = lr * tensor.sqrt(1. - beta2**t) / (1. - beta1**t)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0., p.name + '_mean')
        v = theano.shared(p.get_value() * 0., p.name + '_variance')
        m_t = beta1 * m + (1. - beta1) * g
        v_t = beta2 * v + (1. - beta2) * g**2
        step = lr_t * m_t / (tensor.sqrt(v_t) + e)
        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))

    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up,
                                    profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def train(dim_word=100,  # word vector dimensionality
          dim_chunk=50,
          dim=1000,  # the number of LSTM units
          dim_chunk_hidden=2000,
          encoder='gru',
          decoder='gru_cond',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.01,  # learning rate
          n_words_src=100000,  # source vocabulary size
          n_words=100000,  # target vocabulary size
          n_chunks=1000,  # target vocabulary size
          maxlen_chunk=10,  # maximum length of the description
          maxlen_chunk_words=50,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'],
          valid_datasets=['../data/dev/newstest2011.en.tok',
                          '../data/dev/newstest2011.fr.tok'],
          dictionaries=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],
          dictionary_chunk='/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
          use_dropout=False,
          reload_=False,
          overwrite=False):

    # Model options
    model_options = locals().copy()

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
    model_options['n_chunks'] = len(worddict_chunk)
    print 'chunk_dict size: ', model_options['n_chunks']
    print worddict_chunk

    # reload options
    if reload_ and os.path.exists(saveto) and os.path.exists(saveto + '.pkl'):
        print 'Reloading model options'
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    print 'Loading data'

    # begin to read by iterators
    train = TrainingTextIterator(datasets[0], datasets[1],
                                 dictionaries[0], dictionaries[1], dictionary_chunk,
                                 n_words_source=n_words_src, n_words_target=n_words,
                                 batch_size=batch_size,
                                 max_chunk_len=maxlen_chunk, max_word_len=maxlen_chunk_words)
    valid = TrainingTextIterator(valid_datasets[0], valid_datasets[1],
                                 dictionaries[0], dictionaries[1], dictionary_chunk,
                                 n_words_source=n_words_src, n_words_target=n_words,
                                 batch_size=valid_batch_size,
                                 max_chunk_len=maxlen_chunk, max_word_len=maxlen_chunk_words)

    print 'Building model'


    # init all the parameters for model
    params = init_params(model_options)


    # reload parameters
    if reload_ and os.path.exists(saveto):
        print 'Reloading model parameters'
        params = load_params(saveto, params)


    tparams = init_tparams(params)
    # modify the module of build model!
    # especially the inputs and outputs
    trng, use_noise, \
    x, x_mask, y_chunk, y_mask, y_cw, y_chunk_indicator, \
    opt_ret, \
    cost, cost_cw= \
        build_model(tparams, model_options)

    inps = [x, x_mask, y_chunk, y_mask, y_cw, y_chunk_indicator]

    print 'Building sampler'
    f_init, f_next_chunk, f_next_word = build_sampler(tparams, model_options, trng, use_noise)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas_chunk'].sum(0))**2).sum(1).mean()
        alpha_reg += alpha_c * (
            (tensor.cast(y_mask.sum(0).sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas_cw'].sum(0).sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    best_p = None
    bad_counter = 0
    uidx = 0
    estop = False
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        rmodel = numpy.load(saveto)
        history_errs = list(rmodel['history_errs'])
        if 'uidx' in rmodel:
            uidx = rmodel['uidx']

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    # print 'train length', len(train)

    for eidx in xrange(max_epochs):
        n_samples = 0

        for x, y_chunk, y_cw in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask, y_c, y_cw, chunk_indicator, y_mask = prepare_training_data(x, y_chunk, y_cw, maxlen_chunk=maxlen_chunk, maxlen_cw=maxlen_chunk_words,
                                                                              n_words_src=n_words_src,
                                                                              n_words=n_words)

            if x is None:
                print 'Minibatch with zero sample under chunk length ', maxlen_chunk, 'word length: ', maxlen_chunk_words
                uidx -= 1
                continue

            ud_start = time.time()



            # compute cost, grads and copy grads to sh            self.target_buffer = _tcbufared variables
            cost = f_grad_shared(x, x_mask, y_c, y_mask, y_cw, chunk_indicator)

            # print 'Epoch ', eidx, 'processed one batch'

            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving the best model...',
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, uidx=uidx, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print 'Done'

                # save with uidx
                if not overwrite:
                    print 'Saving the model at iteration {}...'.format(uidx),
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    numpy.savez(saveto_uidx, history_errs=history_errs,
                                uidx=uidx, **unzip(tparams))
                    print 'Done'


            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5, x.shape[1])):
                    stochastic = True
                    sample, score = gen_sample(tparams, f_init, f_next_chunk, f_next_word,
                                               x[:, jj][:, None],
                                               model_options, trng=trng, k=1,
                                               stochastic=stochastic,
                                               argmax=False)
                    print 'Source ', jj, ': ',
                    for vv in x[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[0]:
                            print worddicts_r[0][vv],
                        else:
                            print 'UNK',
                    print
                    print 'Truth ', jj, ' : ',
                    ci = 0
                    # print y_chunk[: , jj]
                    for chunk_index, word_index in zip(y_c[:, jj], y_cw[:, jj]):

                        if word_index == 0:
                            break
                        if chunk_index in worddict_r_chunk and chunk_index != 1: # not NULL
                            print '|', worddict_r_chunk[chunk_index],
                        if word_index in worddicts_r[1]:
                            print worddicts_r[1][word_index],
                        else:
                            print 'UNK',
                        ci += 1
                    print
                    print 'Sample ', jj, ': ',
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            continue
                        if vv < 0:
                            vv = vv * -1
                            # print vv,
                            print '|', worddict_r_chunk[vv],
                            continue
                        if vv in worddicts_r[1]:
                            print worddicts_r[1][vv],
                        else:
                            print 'UNK',
                    print

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_probs(f_log_probs, prepare_training_data,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                print 'Valid ', valid_err

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = pred_probs(f_log_probs, prepare_training_data,
                           model_options, valid).mean()

    print 'Valid ', valid_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                uidx=uidx,
                **params)

    return valid_err


def sgd(lr, tparams, grads, x, mask, y, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update


if __name__ == '__main__':
    pass
