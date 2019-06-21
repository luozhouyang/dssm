import tensorflow as tf
from dssm import models

config = {
    'vocab_size': 100,
    'embedding_size': 300,
    'vec_dim': 256
}


class ModelsTest(tf.test.TestCase):

    def testMLPModel(self):
        m = models.build_mlp_model(config)
        m.summary()

    def testLSTMModel(self):
        m = models.build_lstm_model(config)
        m.summary()


if __name__ == '__main__':
    tf.test.main()
