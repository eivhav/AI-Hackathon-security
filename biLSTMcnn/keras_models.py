
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
                              nb_filter=300,
                              activation='tanh',
                              border_mode='same') for filter_length in [1, 2, 3, 5]]

        doc_cnn = merge([cnn(doc_pooling) for cnn in cnns], mode='concat')

        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        maxpool.supports_masking = True
        doc_end_pooling = maxpool(doc_cnn)

        dense_output = Dense(input_dim=1000, activation='sigmoid', output_dim=1, name='final_output', )(doc_end_pooling)

        self.model = Model(input=doc_input, output=dense_output, name='lstm_cnn_model')

        def my_binary_crossentropy(y_true, y_pred):
            return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

        self.model.compile(loss='mean_squared_error', optimizer='adam')

        print(self.model.summary())



file_path_pos = "/home/havikbot/Downloads/full_coments_pos.txt"
file_path_neg = "/home/havikbot/Downloads/full_coments_neg.txt"


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
    create_embeddings(data_path, text_data, size=300, min_count=2, window=10, sg=1, iter=100)

    word2idx, idx2word = load_vocab()

    config = {'doc_len': 250,
              'initial_embed_weights': '/home/havikbot/Downloads/embeddings.npz',
              'n_words': 19040}

    model_class = ConvolutionalLSTM()
    model_class.build_and_compile(config)



    data = np.zeros((len(text_data), 250), dtype=np.float32)
    for l in range(len(text_data)):
        new_line1 = tokenize(text_data[l])
        for i in range(len(new_line1)):
            try:
                data[l][i] = word2idx[new_line1[i]]
            except:
                continue

    y_pred = np.array(labels).reshape(len(labels), 1)

    train_data = data[0:int(len(data)*0.9)]
    test_data = data[int(len(data) * 0.9):]


    #model_class.model.fit(train_data, y_pred[0:int(len(data)*0.9)], nb_epoch=15, batch_size=64, verbose=1, validation_split=0.2)
    #model_class.model.save_weights("300model.h5")
    model_class.model.load_weights("300model.h5")
    rel_docs = []
    re_docs_txt = []

    for i in range(0, 100):
        txt_index = int(len(data) * 0.9) + i

        prediction = model_class.model.predict(test_data[i].reshape(1, 250))
        pred = "Relevant"
        if prediction[0][0] <= 0.5: pred = "Not Relevant"
        line = ""
        for s in tokenize(text_data[txt_index]): line = line + " " +s
        print(pred, '\t', line)

        if pred == "Relevant" :
            rel_docs.append(line.split(" "))

    from  gensim.models import LdaModel
    from gensim import corpora
    dictionary = corpora.Dictionary(rel_docs)
    corpus = [dictionary.doc2bow(text) for text in rel_docs]

    ldamodel = LdaModel(corpus, num_topics=50, id2word=dictionary, passes=20)
    for i in range(len(rel_docs)):
        print(ldamodel[dictionary.doc2bow(rel_docs[i])])
        print()

    for i in range(len(rel_docs)):

        print(rel_docs)














