import tensorflow as tf

from dssm.lstm_model import LSTMModel
from dssm import utils


class LSTMModelTest(tf.test.TestCase):

    def testTrain(self):
        m = LSTMModel()
        m.compile(loss={'output_4': 'binary_crossentropy'}, optimizer='sgd')
        d = utils.build_dummy_train_dataset()
        m.fit(d)


if __name__ == '__main__':
    tf.test.main()
