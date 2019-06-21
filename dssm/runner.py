import argparse
import os
import tensorflow as tf
from easylib.dl import KerasModelDatasetRunner

from dssm import datasets
from dssm import models


def train(model, config):
    runner = KerasModelDatasetRunner(
        model,
        model_dir=config['model_dir'],
        model_name='dssm',
        configs=config,
        logger_name='dssm')
    train_dataset = datasets.build_train_dataset(config['train_files'], config)
    eval_dataset = None
    if config['eval_files']:
        eval_dataset = datasets.build_eval_dataset(config['eval_files'], config)
    runner.train(train_dataset, val_dataset=eval_dataset)


def evaluate(model, config):
    if not config.get('eval_files', None):
        raise ValueError('`eval_files` must provided.')
    runner = KerasModelDatasetRunner(
        model,
        model_dir=config['model_dir'],
        model_name='dssm',
        configs=config,
        logger_name='dssm')
    eval_dataset = datasets.build_eval_dataset(config['eval_files'], config)
    runner.eval(eval_dataset)


def predict(model, config):
    runner = KerasModelDatasetRunner(
        model,
        model_dir=config['model_dir'],
        model_name='dssm',
        configs=config,
        logger_name='dssm')
    predict_dataset = datasets.build_predict_dataset(config['predict_files'], config)
    res = runner.predict(predict_dataset)
    print(res[0])
    print(res[1])


# TODO(luozhouyang) Fix export
def export(model, config):
    runner = KerasModelDatasetRunner(
        model,
        model_dir=config['model_dir'],
        model_name='dssm',
        configs=config,
        logger_name='dssm')
    runner.export(os.path.join(config['model_dir'], 'export', '1'), ckpt=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'lstm'], help="""model type.""")
    parser.add_argument('--action', type=str, default='train', choices=['train', 'eval', 'predict', 'export'])

    config = {}
    dataset_config = {
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
        'vocab_file': 'data/vocab.txt',
        'train_files': ['data/train.txt'],
        'eval_files': ['data/train.txt'],
        'predict_files': ['data/train.txt']
    }
    model_config = {
        'vocab_size': 10,
        'embedding_size': 256,
        'vec_dim': 256,
    }
    runner_config = {
        'ckpt_period': 1,
        'model_dir': '/tmp/dssm'
    }
    config.update(dataset_config)
    config.update(model_config)
    config.update(runner_config)

    args, _ = parser.parse_known_args()
    if 'mlp' == args.model:
        model = models.build_mlp_model(config)
    elif 'lstm' == args.model:
        model = models.build_lstm_model(config)
    else:
        raise ValueError('Invalid model: %s' % args.model)

    if 'train' == args.action:
        train(model, config)
    elif 'eval' == args.action:
        evaluate(model, config)
    elif 'predict' == args.action:
        predict(model, config)
    elif 'export' == args.action:
        export(model, config)
    else:
        raise ValueError('Invalid action: %s' % args.action)
