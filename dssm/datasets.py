import tensorflow as tf
from tensorflow.python.ops import lookup_ops


def build_train_dataset(config):
    train_files = config['train_files'].split(',')
    dataset = tf.data.Dataset.from_tensor_slices(train_files)
    dataset = dataset.flat_map(lambda x: tf.data.TextLineDataset(x))
    dataset = dataset.map(
        lambda x: (
            tf.string_split([x], delimiter='@').values[0],
            tf.string_split([x], delimiter='@').values[1],
            tf.string_split([x], delimiter='@').values[2]),
        num_parallel_calls=config['num_parallel_calls']
    ).prefetch(config['buff_size'])

    dataset = dataset.map(
        lambda q, d, l: (
            tf.string_split([q], delimiter=' ').values,
            tf.string_split([d], delimiter=' ').values,
            tf.cast(tf.equal(l, '0 1'), dtype=tf.int32)),
        num_parallel_calls=config['num_parallel_calls']
    ).prefetch(config['buff_size'])

    word2index = lookup_ops.index_table_from_file(config['vocab_file'], default_value=0)
    dataset = dataset.map(
        lambda q, d, l: (
            word2index.lookup(q),
            word2index.lookup(d),
            l),
        num_parallel_calls=config['num_parallel_calls']
    ).prefetch(config['buff_size'])

    dataset = dataset.padded_batch(
        batch_size=config['batch_size'],
        padded_shapes=(
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([])),
        padding_values=(
            tf.constant(0, dtype=tf.int64),
            tf.constant(0, dtype=tf.int64),
            0)
    )

    return dataset
