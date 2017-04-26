__author__ = 'zhouh'



from training_data_iterator import TrainingTextIterator
from nmt import prepare_training_data


# train = TrainingTextIterator('/home/zhouh/workspace/python/nmtdata/small.ch',
#                              '/home/zhouh/workspace/python/nmtdata/small.en.chunked',
#                              '/home/zhouh/workspace/python/nmtdata/small.ch.pkl',
#                              '/home/zhouh/workspace/python/nmtdata/small.en.chunked.pkl',
#                              '/home/zhouh/workspace/python/nmtdata/small.en.chunked.chunktag.pkl',
#                               n_words_source=10000, n_words_target=10000,
#                               batch_size=2, max_chunk_len=50, max_word_len=50)

train = TrainingTextIterator('/home/zhouh/workspace/python/nmtdata/hms.ch.filter',
                             '/home/zhouh/workspace/python/nmtdata/hms.en.filter.chunked',
                             '/home/zhouh/workspace/python/nmtdata/hms.ch.filter.pkl',
                             '/home/zhouh/workspace/python/nmtdata/hms.en.filter.chunked.pkl',
                             '/home/zhouh/workspace/python/nmtdata/hms.en.filter.chunked.chunktag.pkl',
                              n_words_source=10000, n_words_target=10000,
                              batch_size=1, max_chunk_len=30, max_word_len=50)



n = 0
batch = 0
for i in train:
    print batch
    batch += 1

    s = i[0]
    tc = i[1]
    tcw = i[2]

    print 's', s
    print 'tc', tc
    print 'tcw', tcw


    x, x_mask, y_c, y_cw, chunk_indicator, y_mask = prepare_training_data(s, tc, tcw, maxlen_chunk=10, maxlen_cw=50,
                                                                      n_words_src=1000,
                                                                      n_words=1000)
    print 'x', x
    print 'x_mask', x_mask
    print 'y_c', y_c
    print 'chunk_indicator', chunk_indicator
    print 'y_cw', y_cw
    print 'y_mask', y_mask




print batch

