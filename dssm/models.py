import tensorflow as tf


def build_simple_model(config):
    query_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='query_input')
    doc_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='doc_input')

    embedding = tf.keras.layers.Embedding(config['vocab_size'], config['embedding_size'])
    query_emb = embedding(query_input)
    doc_emb = embedding(doc_input)

    lstm = tf.keras.layers.LSTM(config['lstm_units'])
    query_lstm = lstm(query_emb)
    doc_lstm = lstm(doc_emb)

    query_dense_1 = tf.keras.layers.Dense(1024)(query_lstm)
    doc_dense_1 = tf.keras.layers.Dense(1024)(doc_lstm)

    query_vec = tf.keras.layers.Dense(256, name='query_vec')(query_dense_1)
    doc_vec = tf.keras.layers.Dense(256, name='doc_vec')(doc_dense_1)

    dot = tf.keras.layers.Dot(axes=1, normalize=True, name='similarity')([query_vec, doc_vec])
    out = tf.keras.layers.Dense(1, activation='sigmoid')(dot)

    model = tf.keras.Model(inputs=[query_input, doc_input], outputs=[out])
    return model
