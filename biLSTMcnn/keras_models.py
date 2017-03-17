



class ConvolutionalLSTM(LanguageModel):
    def build(self):
        question = self.question
        answer = self.get_answer()

        # add embedding layers
        weights = np.load(self.config['initial_embed_weights'])
        embedding = Embedding(input_dim=self.config['n_words'],
                              output_dim=weights.shape[1],
                              weights=[weights],
                              trainable=True)

        question_embedding = embedding(question)
        answer_embedding = embedding(answer)

        f_rnn = LSTM(141, return_sequences=True, consume_less='mem')
        b_rnn = LSTM(141, return_sequences=True, consume_less='mem', go_backwards=True)

        qf_rnn = f_rnn(question_embedding)
        qb_rnn = b_rnn(question_embedding)
        question_pool = merge([qf_rnn, qb_rnn], mode='concat', concat_axis=-1)

        af_rnn = f_rnn(answer_embedding)
        ab_rnn = b_rnn(answer_embedding)
        answer_pool = merge([af_rnn, ab_rnn], mode='concat', concat_axis=-1)

        # cnn
        cnns = [Convolution1D(filter_length=filter_length,
                              nb_filter=500,
                              activation='tanh',
                              border_mode='same') for filter_length in [1, 2, 3, 5]]

        question_cnn = merge([cnn(question_pool) for cnn in cnns], mode='concat')
        answer_cnn = merge([cnn(answer_pool) for cnn in cnns], mode='concat')

        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        maxpool.supports_masking = True
        question_pool = maxpool(question_cnn)
        answer_pool = maxpool(answer_cnn)


        return question_pool, answer_pool
