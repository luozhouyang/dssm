import tensorflow as tf

keras = tf.keras


def build_mlp_model(config):
    query = keras.layers.Input(shape=(config['query_max_len'],), dtype=tf.int64, name='query_input')
    doc = keras.layers.Input(shape=(config['doc_max_len'],), dtype=tf.int64, name='doc_input')

    embedding = keras.layers.Embedding(config['vocab_size'], config['embedding_size'], name='embedding')
    # average embedding
    avg_embedding = keras.layers.Lambda(lambda x: tf.reduce_mean(x, 1))

    query_embed = embedding(query)
    query_embed = avg_embedding(query_embed)
    doc_embed = embedding(doc)
    doc_embed = avg_embedding(doc_embed)

    query_dense0 = keras.layers.Dense(1024)(query_embed)
    query_vec = keras.layers.Dense(config['vec_dim'], name='query_vec')(query_dense0)

    doc_dense0 = keras.layers.Dense(1024)(doc_embed)
    doc_vec = keras.layers.Dense(config['vec_dim'], name='doc_vec')(doc_dense0)

    cos = keras.layers.Dot(axes=-1, normalize=True, name='cosine')([query_vec, doc_vec])
    out = keras.layers.Dense(1, activation='sigmoid', name='out')(cos)

    model = keras.Model(inputs=[query, doc], outputs={'out': out, 'cosine': cos})
    model.compile(optimizer='sgd',
                  loss={'out': 'binary_crossentropy'},
                  metrics={"out": [keras.metrics.Accuracy(), keras.metrics.Precision(), keras.metrics.Recall()]})
    return model


def build_lstm_model(config):
    query = keras.layers.Input(shape=(config['query_max_len'],), dtype=tf.int64, name='query_input')
    doc = keras.layers.Input(shape=(config['doc_max_len'],), dtype=tf.int64, name='doc_input')

    embedding = keras.layers.Embedding(config['vocab_size'], config['embedding_size'], name='embedding')
    query_embed = embedding(query)
    doc_embed = embedding(doc)

    query_lstm = keras.layers.LSTM(256)(query_embed)
    doc_lstm = keras.layers.LSTM(256)(doc_embed)

    query_vec = keras.layers.Dense(config['vec_dim'])(query_lstm)
    doc_vec = keras.layers.Dense(config['vec_dim'])(doc_lstm)

    cos = keras.layers.Dot(axes=-1, normalize=True, name='cosine')([query_vec, doc_vec])
    out = keras.layers.Dense(1, activation='sigmoid', name='out')(cos)

    model = keras.Model(inputs=[query, doc], outputs={'out': out, 'cosine': cos})
    model.compile(optimizer='sgd',
                  loss={'out': 'binary_crossentropy'},
                  metrics={"out": [keras.metrics.Accuracy(), keras.metrics.Precision(), keras.metrics.Recall()]})

    return model
