import argparse
import logging
import os

import tensorflow as tf
from easylib.dl import KerasModelDatasetRunner
from nlp_datasets.tokenizers import SpaceTokenizer
from nlp_datasets.xyz_dataset import XYZSameFileDataset

from dssm import models

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

    tokenizer = SpaceTokenizer()
    tokenizer.build_from_vocab(config['vocab_file'])
    logging.info('Build tokenizer from vocab file: %s' % config['vocab_file'])
    logging.info('vocab size of tokenizer: %d' % tokenizer.vocab_size)
    dataset = XYZSameFileDataset(x_tokenizer=tokenizer, y_tokenizer=tokenizer, config=None)

    runner = KerasModelDatasetRunner(
        model=model,
        model_name='dssm',
        model_dir=config['model_dir'],
        configs=config)

    if 'train' == args.action:
        train_files = config['train_files']
        train_dataset = dataset.build_train_dataset(train_files=train_files)
        eval_dataset = dataset.build_eval_dataset(eval_files=config['eval_files']) if config['eval_files'] else None
        runner.train(dataset=train_dataset, val_dataset=eval_dataset, ckpt=None)

    elif 'eval' == args.action:
        if not config['eval_files']:
            raise ValueError('eval_files must not be None in eval mode.')
        eval_dataset = dataset.build_eval_dataset(eval_files=config['eval_files'])
        runner.eval(dataset=eval_dataset, ckpt=None)
        logging.info('Finished to evaluate model.')
    elif 'predict' == args.action:
        if not config['predict_files']:
            raise ValueError('predict_files must not be None in predict mode.')
        predict_dataset = dataset.build_predict_dataset(eval_files=config['eval_files'])
        history = runner.predict(dataset=eval_dataset, ckpt=None)
        logging.info('Finished ti predict files.')
    elif 'export' == args.action:
        runner.export(os.path.join(config['model_dir'], 'export'), ckpt=None)
        logging.info('Finished to export models.')
    else:
        raise ValueError('Invalid action: %s' % args.action)
