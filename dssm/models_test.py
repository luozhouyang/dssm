import tensorflow as tf
import tensorflow.contrib.eager as tfe
from dssm import datasets
from dssm import models

config = {
    'batch_size': 2,
    'buff_size': 1,
    'num_parallel_calls': 1,
    'train_files': '../data/train.txt',
    'vocab_file': '../data/vocab.txt',
    'vocab_size': 10,
    'embedding_size': 8,
    'lstm_units': 4
}

tf.enable_eager_execution()


class ModelsTest(tf.test.TestCase):

    def testSimpleModel(self):
        model = models.build_simple_model(config)
        dataset = datasets.build_train_dataset(config)
        dataset = dataset.map(lambda q, d, l: ((q, d), l))

        model.compile(loss='binary_crossentropy', optimizer='sgd')
        model.summary()
        model.fit(dataset, steps_per_epoch=2)


if __name__ == '__main__':
    tf.test.main()
