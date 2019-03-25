import tensorflow as tf


def build_model():
    # query
    query_input = tf.keras.layers.Input(shape=(768,), dtype=tf.float32)
    query = tf.keras.layers.Dense(1024, activation='relu')(query_input)
    query = tf.keras.layers.Dense(256, activation='relu')(query)

    # document
    doc_input = tf.keras.layers.Input(shape=(768,), dtype=tf.float32)
    doc = tf.keras.layers.Dense(1024, activation='relu')(doc_input)
    doc = tf.keras.layers.Dense(256, activation='relu')(doc)

    # dot
    sim = tf.keras.layers.Dot(axes=1, normalize=True)([query, doc])
    # model
    model = tf.keras.Model(inputs=[query_input, doc_input], outputs=[sim])
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.SGD(lr=1.0, momentum=0.9, decay=1.0/4),
        metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    model.summary()
    return model


model = build_model()
