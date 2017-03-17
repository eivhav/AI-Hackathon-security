
from keras.engine import Input
from keras.layers import merge, Embedding, Dropout, Convolution1D, Lambda, LSTM, Dense
from keras import backend as K
from keras.models import Model
import numpy as np
import json
from gensim.models import Word2Vec
import os
from gensim.utils import simple_preprocess
import codecs






class ConvolutionalLSTM():
    def __init__(self):
        self.model = None

    def build_and_compile(self, config):
        doc_input = Input(shape=(config['doc_len'],), dtype='float32', name='doc_base')

        # add embedding layers
        weights = np.load(config['initial_embed_weights'])
        doc_embedding = Embedding(input_dim=config['n_words'],
                              output_dim=weights.shape[1],
                              weights=[weights],
                              trainable=True)(doc_input)

        f_rnn = LSTM(141, return_sequences=True, consume_less='mem')(doc_embedding)
        b_rnn = LSTM(141, return_sequences=True, consume_less='mem', go_backwards=True)(doc_embedding)

        doc_pooling = merge([f_rnn, b_rnn], mode='concat', concat_axis=-1)

        # cnn
        cnns = [Convolution1D(filter_length=filter_length,
                              nb_filter=500,
                              activation='tanh',
                              border_mode='same') for filter_length in [1, 2, 3, 5]]

        doc_cnn = merge([cnn(doc_pooling) for cnn in cnns], mode='concat')

        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        maxpool.supports_masking = True
        doc_end_pooling = maxpool(doc_cnn)

        dense_output = Dense(input_dim=1000, output_dim=1, name='final_output')(doc_end_pooling)

        self.model = Model(input=doc_input, output=dense_output, name='lstm_cnn_model')
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(self.model.summary())



file_path_pos = "/home/havikbot/Downloads/pos_distinct.txt"
file_path_neg = "/home/havikbot/Downloads/neg_distinct.txt"


data_pos = codecs.open(file_path_pos, "r", "utf-8").readlines()
data_neg = codecs.open(file_path_neg, "r", "utf-8").readlines()

all_data = [[p, 1] for p in data_pos] + [[n, 0] for n in data_neg]
from random import shuffle
shuffle(all_data)
text_data = [x[0] for x in all_data]
labels = [x[1] for x in all_data]

tokenize = lambda x: simple_preprocess(x)


def create_embeddings(data_dir, text_data, embeddings_path='embeddings.npz', vocab_path='map.json', **params):
    """
    Generate embeddings from a batch of text
    :param embeddings_path: where to save the embeddings
    :param vocab_path: where to save the word-index map
    """

    class SentenceGenerator(object):
        def __init__(self, data_dir, text_data):
            self.text_data = text_data

        def __iter__(self):
            #lines = codecs.open(self.file_path, "r", "utf-8").readlines()
            for line in text_data:
                yield tokenize(line)

    sentences = SentenceGenerator(data_dir, text_data)

    model = Word2Vec(sentences, workers=10, size=300, min_count=3)
    weights = model.syn0
    np.save(open(data_dir+embeddings_path, 'wb'), weights)

    vocab = dict([(k, v.index) for k, v in model.vocab.items()])
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vocab))


def load_vocab(vocab_path='map.json'):
    """
    Load word -> index and index -> word mappings
    :param vocab_path: where the word-index map is saved
    :return: word2idx, idx2word
    """

    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word


if __name__ == '__main__':
    # specify embeddings in this environment variable
    data_path = '/home/havikbot/Downloads/'

    # variable arguments are passed to gensim's word2vec model
    create_embeddings(data_path, text_data, size=300, min_count=3, window=10, sg=1, iter=100)

    word2idx, idx2word = load_vocab()

    config = {'doc_len': 200,
              'initial_embed_weights': '/home/havikbot/Downloads/embeddings.npz',
              'n_words': 3680}

    model_class = ConvolutionalLSTM()
    model_class.build_and_compile(config)


    train_data = np.zeros((len(text_data), 200), dtype=np.float32)
    for l in range(len(text_data)):
        new_line1 = tokenize(text_data[l])
        for i in range(len(new_line1)):
            try:
                train_data[l][i] = word2idx[new_line1[i]]
            except:
                continue


    y_pred = np.array(labels).reshape(len(labels), 1)

    model_class.model.fit(train_data, y_pred, nb_epoch=20, batch_size=256, verbose=1, validation_split=0.2)

