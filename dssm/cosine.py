import tensorflow as tf


def cosine_similarity(x):
    q, d = x
    q_norm = tf.sqrt(tf.reduce_sum(tf.square(q), 1, True))
    d_norm = tf.sqrt(tf.reduce_sum(tf.square(d), 1, True))
    p = tf.reduce_sum(tf.multiply(q, d), 1, True)
    p_norm = tf.multiply(q_norm, d_norm)
    cos = tf.truediv(p, p_norm)
    return cos
