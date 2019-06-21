import tensorflow as tf

from dssm import datasets

config = {
    'sep': '@',
    'num_parallel_calls': 1,
    'buffer_size': 1000,
    'seed': None,
    'reshuffle_each_iteration': True,
    'train_batch_size': 2,
    'eval_predict_size': 2,
    'predict_batch_size': 2,
    'query_max_len': 5,
    'doc_max_len': 5,
    'vocab_file': '../data/vocab.txt'
}


class DatasetTest(tf.test.TestCase):

    def decodeNumpyArray2D(self, t):
        return [[x.decode('utf8') for x in s] for s in t]

    def testBuildTrainDataset(self):
        train_files = '../data/train.txt'
        dataset = datasets.build_train_dataset(train_files, config)
        for v in dataset:
            qd = v[0]
            print(qd[0].numpy())
            print(qd[1].numpy())
            print(v[1])

    def testBuildEvalDataset(self):
        eval_files = '../data/train.txt'
        dataset = datasets.build_eval_dataset(eval_files, config)
        for v in dataset:
            qd = v[0]
            print(qd[0].numpy())
            print(qd[1].numpy())
            print(v[1])

    def testBuildPredictDataset(self):
        predict_files = '../data/train.txt'
        dataset = datasets.build_predict_dataset(predict_files, config)
        for v in dataset:
            print(v[0].numpy())
            print(v[1].numpy())


if __name__ == '__main__':
    tf.test.main()
