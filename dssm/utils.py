import tensorflow as tf


def build_dummy_query_doc_label():
    q = [
        [0, 1, 2, 3, 4],
        [2, 4, 3, 2, 1],
    ]
    d = [
        [0, 2, 5, 4, 2, 1, 0],
        [5, 4, 3, 2, 6, 2, 0]
    ]
    l = [
        [0],
        [1]
    ]
    return q, d, l


def build_dummy_train_dataset():
    q, d, l = build_dummy_query_doc_label()
    q_d = tf.data.Dataset.from_tensor_slices(q)
    d_d = tf.data.Dataset.from_tensor_slices(d)
    l_d = tf.data.Dataset.from_tensor_slices(l)
    d = tf.data.Dataset.zip((q_d, d_d, l_d))
    d = d.repeat(10)
    d = d.shuffle(1000)
    d = d.map(
        lambda q, d, l: ((q, d), l)
    )
    d = d.batch(2)
    return d
