import tensorflow as tf

from tensorflow.python.ops import lookup_ops


def _build_dataset_from_files(files, config):
    if isinstance(files, str):
        dataset = tf.data.TextLineDataset(files).skip(config.get('skip_count', 0))
    elif isinstance(files, list):
        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.flat_map(lambda x: tf.data.TextLineDataset(x).skip(config.get('skip_count', 0)))
    else:
        raise ValueError("Argument `train_files` must be a `str` or `list`.")
    return dataset


@tf.function
def _normalize_fn(x):
    # return tf.where(tf.equal(x, '0 1') or tf.equal(x, '1'), 1, 0)
    if tf.equal(x, '0 1'):
        return 1
    if tf.equal(x, '1'):
        return 1
    return 0


def build_train_dataset(train_files, config):
    return _build_dataset(train_files, config, mode='train')


def _build_dataset(files, config, mode='train'):
    dataset = _build_dataset_from_files(files, config)
    dataset = dataset.shuffle(buffer_size=config['buffer_size'],
                              seed=config['seed'],
                              reshuffle_each_iteration=config['reshuffle_each_iteration'])
    dataset = dataset.filter(lambda x: tf.equal(tf.size(tf.strings.split([x], sep=config['sep']).values), 3))

    dataset = dataset.map(
        lambda x: (tf.strings.split([x], sep=config['sep']).values[0],
                   tf.strings.split([x], sep=config['sep']).values[1],
                   tf.strings.split([x], sep=config['sep']).values[2]),
        num_parallel_calls=config['num_parallel_calls']
    ).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(
        lambda q, d, l: (tf.strings.split([q]).values, tf.strings.split([d]).values, _normalize_fn(l)),
        num_parallel_calls=config['num_parallel_calls']
    ).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.filter(
        lambda q, d, l: tf.logical_and(tf.size(q) <= config['query_max_len'], tf.size(d) <= config['doc_max_len']))

    str2id = lookup_ops.index_table_from_file(config['vocab_file'], default_value=0)  # unk_id: 0
    unk_id = tf.constant(0, dtype=tf.int64)

    dataset = dataset.map(
        lambda q, d, l: (str2id.lookup(q), str2id.lookup(d), tf.cast(l, tf.int64)),
        num_parallel_calls=config['num_parallel_calls']
    ).prefetch(tf.data.experimental.AUTOTUNE)

    batch_size = config['eval_batch_size'] if mode == 'eval' else config['train_batch_size']

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(config['query_max_len'], config['doc_max_len'], []),
        padding_values=(unk_id, unk_id, unk_id)
    )
    dataset = dataset.map(
        lambda q, d, l: ((q, d), l),
        num_parallel_calls=config['num_parallel_calls']
    )
    return dataset


def build_eval_dataset(eval_files, config):
    return _build_dataset(eval_files, config)


def build_predict_dataset(predict_files, config):
    dataset = _build_dataset_from_files(predict_files, config)
    dataset = dataset.filter(lambda x: tf.equal(tf.size(tf.strings.split([x], sep=config['sep']).values), 3))
    dataset = dataset.map(
        lambda x: (tf.strings.split([x], sep=config['sep']).values[0],
                   tf.strings.split([x], sep=config['sep']).values[1],),
        num_parallel_calls=config['num_parallel_calls']
    ).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(
        lambda q, d: (tf.strings.split([q], sep=' ').values, tf.strings.split([d]).values),
        num_parallel_calls=config['num_parallel_calls']
    ).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.filter(
        lambda q, d: tf.logical_and(tf.size(q) <= config['query_max_len'], tf.size(d) <= config['doc_max_len']))

    str2id = lookup_ops.index_table_from_file(config['vocab_file'], default_value=0)  # unk_id: 0
    unk_id = tf.constant(0, dtype=tf.int64)

    dataset = dataset.map(
        lambda q, d: (str2id.lookup(q), str2id.lookup(d)),
        num_parallel_calls=config['num_parallel_calls']
    ).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.padded_batch(
        config['predict_batch_size'],
        padded_shapes=(config['query_max_len'], config['doc_max_len']),
        padding_values=(unk_id, unk_id)
    )
    dataset = dataset.map(
        lambda q, d: ((q, d),)
    )
    return dataset
