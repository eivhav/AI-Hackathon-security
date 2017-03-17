
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
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        print(self.model.summary())



file_path_pos = "/home/havikbot/Downloads/pos_distinct.txt"
file_path_neg = ""


data_pos = codecs.open(file_path_pos, "r", "utf-8").readlines()
#data_neg = open(file_path_neg).readlines()

tokenize = lambda x: simple_preprocess(x)


def create_embeddings(data_dir, file_path,  embeddings_path='embeddings.npz', vocab_path='map.json', **params):
    """
    Generate embeddings from a batch of text
    :param embeddings_path: where to save the embeddings
    :param vocab_path: where to save the word-index map
    """

    class SentenceGenerator(object):
        def __init__(self, data_dir, file_path):
            self.file_path = data_dir+file_path

        def __iter__(self):
            lines = codecs.open(self.file_path, "r", "utf-8").readlines()
            for line in lines:
                yield tokenize(line)

    sentences = SentenceGenerator(data_dir, file_path)

    model = Word2Vec(sentences, **params)
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
    create_embeddings(data_path, 'pos_distinct.txt', size=100, min_count=5, window=5, sg=1, iter=25)

    word2idx, idx2word = load_vocab()

    config = {'doc_len': 50,
              'initial_embed_weights': '/home/havikbot/Downloads/embeddings.npz',
              'n_words': 1999}

    model_class = ConvolutionalLSTM()
    model_class.build_and_compile(config)


    train_pos_data = np.zeros((len(data_pos), 50), dtype=np.float32)
    for l in range(len(data_pos)):
        new_line1 = tokenize(data_pos[l])
        for i in range(len(new_line1)):
            try:
                train_pos_data[l][i] = word2idx[new_line1[i]]
            except:
                continue

    y_pred = np.random.rand(3014)

    output = model_class.model.fit(train_pos_data[0:100], y_pred[0:100], nb_epoch=20, batch_size=4, verbose=0, validation_split=0.2)
    print(output)
