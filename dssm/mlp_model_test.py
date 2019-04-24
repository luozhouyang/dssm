import tensorflow as tf

from dssm.mlp_model import MLPModel
from dssm import utils


class MLPModelTest(tf.test.TestCase):

    def testTrain(self):
        m = MLPModel()
        m.compile(loss={'output_4': 'binary_crossentropy'}, optimizer='sgd')
        d = utils.build_dummy_train_dataset()
        for v in iter(d):
            print(v)
            print("------------")

        m.fit(d)


if __name__ == '__main__':
    tf.test.main()
