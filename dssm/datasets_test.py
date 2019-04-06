import tensorflow as tf

from dssm import datasets

config = {
    'batch_size': 2,
    'buff_size': 1,
    'num_parallel_calls': 1,
}

tf.enable_eager_execution()


class DatasetsTest(tf.test.TestCase):

    def testTrainDataset(self):
        config.update({
            'train_files': '../data/train.txt',
            'vocab_file': '../data/vocab.txt',
            'vocab_size': 10
        })
        dataset = datasets.build_train_dataset(config)
        # dataset = dataset.repeat(10)
        for (query, doc, label) in iter(dataset):
            print(query)
            print(doc)
            print(label)
            print('==================================')


if __name__ == '__main__':
    tf.test.main()
