import tensorflow as tf 

from dssm.mlp_model import MLPModel
from datasets.dssm.query_doc_datasets import QueryDocSameFileDataset


class Runner:

    def __init__(self, model):
        self.model = model

    def train(self, train_files):
        dataset = QueryDocSameFileDataset(train_files=train_files, eval_files=None, predict_files=None, vocab_file=None)
        self.model.fit(dataset)


