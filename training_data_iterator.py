import numpy

import cPickle as pkl
import gzip
import sys
import codecs


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


unk_idx = 1
sentence_end_idx = 0


class TrainingTextIterator:
    """Simple Bitext iterator."""
    #
    # max_chunk_len,  # max size of chunks in a sentence
    # max_word_len,   # max size of words in a chunk
    #
    def __init__(self, source, target,
                 source_dict, target_dict, target_chunk_dict,
                 batch_size=128,
                 max_chunk_len=20,  # max size of chunks in a sentence
                 max_word_len=5,   # max size of words in a chunk
                 n_words_source=-1,
                 n_words_target=-1):
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(target_dict, 'rb') as f:
            self.target_dict = pkl.load(f)

        with open(target_chunk_dict, 'rb') as f:
            self.target_chunk_dict = pkl.load(f)

        self.batch_size = batch_size
        self.max_chunk_len = max_chunk_len
        self.max_word_len = max_word_len

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.source_buffer = []
        self.target_chunk_buffer = []
        self.target_chunk_words_buffer = []
        self.k = batch_size * 50

        self.end_of_data = False



    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)

    def readNextChunkSent(self):
        chunk_words = []
        chunk_tag = []

        while(True):
            chunk_line = self.target.readline()

            if(chunk_line == '' and len(chunk_tag) == 0):
                return None, None

            # read until meeting empty line
            if(len(chunk_line.strip()) == 0):
                break

            # the chunk and the  is seperated by \t, and words are sperated by space
            tokens = chunk_line.strip().split('\t')

            words = tokens[1].strip().split()
            ctags = ['NULL'] * len(words)   # index of 'NULL' in chunk dictionary is 1
            ctags[0] = tokens[0]
            chunk_tag.extend( ctags )
            chunk_words.extend( words )

        assert len(chunk_tag) == len(chunk_words)

        return chunk_tag, chunk_words

    def readBuffer(self):

        # print 'read the buffer'

        # read k items into the buffer
        for k_ in xrange(self.k):
            ss = self.source.readline()

            if ss == "":
                break

            # print ss
            chunk_tags, chunk_words = self.readNextChunkSent()
            if chunk_tags is None and chunk_words is None:
                break

            # print chunk_words

            self.source_buffer.append(ss.strip().split())
            self.target_chunk_buffer.append(chunk_tags)
            self.target_chunk_words_buffer.append(chunk_words)

        # sort by target buffer
        tlen = numpy.array([len(t) for t in self.target_chunk_buffer])
        tidx = tlen.argsort()

        _sbuf = [self.source_buffer[i] for i in tidx]
        _tcbuf = [self.target_chunk_buffer[i] for i in tidx]
        _tcwbuf = [self.target_chunk_words_buffer[i] for i in tidx]

        self.source_buffer = _sbuf
        self.target_chunk_buffer = _tcbuf
        self.target_chunk_words_buffer = _tcwbuf

        if len(self.source_buffer) == 0 or len(self.target_chunk_buffer) == 0:

            # print len(self.source_buffer),  len(self.target_chunk_buffer)
            self.end_of_data = False
            self.reset()
            raise StopIteration


    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target_chunk = []
        target_chunk_words = []

        get_none_items = False

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_chunk_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:

            self.readBuffer()


        # retrieval index for each string token
        try:

            # print 'get next'

            # actual work here
            while True:

                if len(self.source_buffer) == 0 and len(source) == 0:
                    self.readBuffer()

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break

                # print 'source before', ' '.join(ss)
                ss = [self.source_dict[w] if w in self.source_dict else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from target file and map to word index
                tt = self.target_chunk_buffer.pop()

                # print 'target chunk before', tt


                tt = [self.target_chunk_dict[w] for w in tt]

                #
                # mark all the chunk tag in the dictionary as 0 and 1,
                # we only want to predict the boundary
                #
                # tt = [1 if w == 1 else 0 for w in tt]

                # print 'target chunk after', tt
                # tt = [w if w < self.n_words_target else 1 for w in tt]

                # read from target file and map to word index
                tcw = self.target_chunk_words_buffer.pop()

                # print 'target before', tcw
                tcw = [self.target_dict[w] if w in self.target_dict else 1 for w in tcw]
                if self.n_words_target > 0:
                    tcw = [w if w < self.n_words_target else 1 for w in tcw]

                # print 'target after', tcw


                # if the source or target chunk or words in target chunk exceed max len, just skip
                # if len(ss) > self.max_word_len and len(tt) > self.max_chunk_len:
                #     continue
                if len(ss) > self.max_word_len or len(tt) > self.max_word_len:

                    # print 'skip', len(ss), len(tt)
                    continue
                # else:
                #     print 'not skip', len(ss), len(tt)

                source.append(ss)
                target_chunk.append(tt)
                target_chunk_words.append(tcw)

                if len(source) >= self.batch_size or \
                        len(target_chunk) >= self.batch_size:
                    break


        except IOError:
            print 'IOError'
            self.end_of_data = True

        if len(source) <= 0 or len(target_chunk) <= 0 or len(target_chunk_words) <= 0:

            # print len(source) ,len(target_chunk) , len(target_chunk_words)
            print 'StopIteration'
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target_chunk, target_chunk_words
