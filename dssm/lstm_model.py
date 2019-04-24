import tensorflow as tf

from dssm.cosine import cosine_similarity


class LSTMModel(tf.keras.Model):

    def __init__(self, params=None):
        super(LSTMModel, self).__init__(name='lstm_model')

        self.embedding = tf.keras.layers.Embedding(100, 128)

        self.query_dense = tf.keras.layers.Dense(1024)
        self.doc_dense = tf.keras.layers.Dense(1024)

        self.query_lstm = tf.keras.layers.LSTM(32)
        self.doc_lstm = tf.keras.layers.LSTM(32)

        self.query_dense_2 = tf.keras.layers.Dense(256, name='query_vec')
        self.doc_dense_2 = tf.keras.layers.Dense(256, name='doc_vec')

        self.cosine = tf.keras.layers.Lambda(lambda x: cosine_similarity(x), name='similarity')

        self.dot = tf.keras.layers.Dot(axes=1, normalize=True, name='dot')

        self.out = tf.keras.layers.Dense(1, activation='sigmoid', name='out')

    def call(self, inputs, training=True, mask=None):
        query, doc = inputs
        query_embedding = self.embedding(query)
        doc_embedding = self.embedding(doc)
        print('query embedding shape: ', query_embedding.shape)
        print('doc embedding shape: ', doc_embedding.shape)

        query_lstm = self.query_lstm(query_embedding)
        doc_lstm = self.doc_lstm(doc_embedding)
        print('query lstm shape: ', query_lstm.shape)
        print('doc lstm shape: ', doc_lstm.shape)

        query_vec = self.query_dense_2(query_lstm)
        doc_vec = self.doc_dense_2(doc_lstm)

        cos = self.cosine([query_vec, doc_vec])
        dot = self.dot([query_vec, doc_vec])
        out = self.out(dot)
        # To avoid 'An operation has `None` for gradient.', use `cos` in somewhere. e.g a dict
        # Note: when calculate loss, you need specify the name. All outputs tensors' name is {output_$NUMBER}
        # e.g
        # model.compile(loss={'output_4': 'binary_crossentropy'})
        return {'query_vec': query_vec, 'doc_vec': doc_vec, 'cos': cos, 'sim': out}
