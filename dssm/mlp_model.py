import tensorflow as tf

from dssm.cosine import cosine_similarity


class MLPModel(tf.keras.Model):

    def __init__(self, params=None, name='mlp_model'):
        super(MLPModel, self).__init__(name=name)

        default_params = self.default_params()
        if params:
            default_params.update(params)
        self.params = default_params

        self.embedding = tf.keras.layers.Embedding(
            self.params['vocab_size'], self.params['embedding_size'], name='embedding')
        self.avg = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1), name='avg')
        self.query_dense = tf.keras.layers.Dense(self.params['dense_units'], name='query_dense')
        self.query_vec = tf.keras.layers.Dense(self.params['vec_dim'], name='query_vec')
        self.doc_dense = tf.keras.layers.Dense(self.params['dense_units'], name='doc_dense')
        self.doc_vec = tf.keras.layers.Dense(self.params['vec_dim'], name='doc_vec')
        self.cos = tf.keras.layers.Lambda(lambda x: cosine_similarity(x), name='cos')
        self.dot = tf.keras.layers.Dot(axes=1, normalize=True, name='dot')
        self.out = tf.keras.layers.Dense(1, activation='sigmoid', name='out')

    def call(self, inputs, training=None, mask=None):
        query, doc = inputs
        query_embedding = self.embedding(query)
        doc_embedding = self.embedding(doc)

        query_embedding = self.avg(query_embedding)
        doc_embedding = self.avg(doc_embedding)

        query_dense = self.query_dense(query_embedding)
        doc_dense = self.doc_dense(doc_embedding)

        query_vec = self.query_vec(query_dense)
        doc_vec = self.doc_vec(doc_dense)

        cos = self.cos([query_vec, doc_vec])

        dot = self.dot([query_vec, doc_vec])
        out = self.out(dot)
        # To avoid 'An operation has `None` for gradient.', use `cos` in somewhere. e.g a dict
        # Note: when calculate loss, you need specify the name. All outputs tensors' name is {output_$NUMBER}
        # e.g
        # model.compile(loss={'output_4': 'binary_crossentropy'})
        return {
            'query_vec': query_vec,
            'doc_vec': doc_vec,
            'cos': cos,
            'sim': out
        }

    def default_params(self):
        params = {
            'vocab_size': 1000,
            'embedding_size': 256,
            'dense_units': 1024,
            'vec_dim': 256,
        }
        return params
