import os
import yaml
import random
import logging
logging.basicConfig(level=logging.INFO)

import mxnet as mx
from mxnet import nd, gluon, autograd, init

import numpy as np
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


import data.dataloader
import config
from config import PARAM_PATH
from helper.callback import Speedometer, Logger
from helper.metric import MAE, RMSE, IndexMAE, IndexRMSE
import model

class ModelTrainer:
	def __init__(self, net, trainer, clip_gradient, logger, ctx):
		self.net = net
		self.trainer = trainer
		self.clip_gradient = clip_gradient
		self.logger = logger
		self.ctx = ctx
		
	def step(self, batch_size): # train step with gradient clipping
		self.trainer.allreduce_grads()
		grads = []
		for param in self.trainer._params:
			if param.grad_req == 'write':
				grads += param.list_grad()
		
		import math
		gluon.utils.clip_global_norm(grads, self.clip_gradient * math.sqrt(len(self.ctx)))
		self.trainer.update(batch_size, ignore_stale_grad=True)

	def process_data(self, epoch, dataloader, metrics=None, is_training=True, title='[TRAIN]'):
		speedometer = Speedometer(title, epoch, frequent=50)
		speedometer.reset()
		if metrics is not None:
			for metric in metrics:
				metric.reset()
		
		for nbatch, batch_data in enumerate(dataloader):
			inputs = [gluon.utils.split_and_load(x, self.ctx) for x in batch_data]
			if is_training:
				self.net.decoder.global_steps += 1.0

				with autograd.record():
					outputs = [self.net(*x, is_training) for x in zip(*inputs)]

				for out in outputs:
					out[0].backward()

				self.step(batch_data[0].shape[0])
			else: outputs = [self.net(*x, False) for x in zip(*inputs)]

			if metrics is not None:
				for metric in metrics:
					for out in outputs:
						metric.update(*out[1])
			speedometer.log_metrics(nbatch + 1, metrics)

		speedometer.finish(metrics)

	def fit(self, begin_epoch, num_epochs, train, eval, test, metrics=None):
		for epoch in range(begin_epoch, begin_epoch + num_epochs):
			if train is not None:
				self.process_data(epoch, train, metrics)
			
			if eval is not None:
				self.process_data(epoch, eval, metrics, is_training=False, title='[EVAL]')
				if (train is not None) and (metrics is not None):
					self.logger.log(epoch, metrics)

			if test is not None:
				self.process_data(epoch, test, metrics, is_training=False, title='[TEST]')
			
			print('')

def main(args):
	with open(args.file, 'r') as f:
		settings = yaml.load(f)
	assert args.file[:-5].endswith(settings['model']['name']), \
		'The model name is not consistent! %s != %s' % (args.file[:-5], settings['model']['name'])

	mx.random.seed(settings['seed'])
	np.random.seed(settings['seed'])
	random.seed(settings['seed'])
	
	dataset_setting = settings['dataset']
	model_setting = settings['model']
	train_setting = settings['training']

	### set meta hiddens
	if 'meta_hiddens' in model_setting.keys():
		config.MODEL['meta_hiddens'] = model_setting['meta_hiddens']

	name = os.path.join(PARAM_PATH, model_setting['name'])
	model_type = getattr(model, model_setting['type'])
	net = model_type.net(settings)

	try:
		logger = Logger.load('%s.yaml' % name)
		net.load_parameters('%s-%04d.params' % (name, logger.best_epoch()), ctx=args.gpus)
		logger.set_net(net)
		print('Successfully loading the model %s [epoch: %d]' % (model_setting['name'], logger.best_epoch()))

		num_params = 0
		for v in net.collect_params().values():
			num_params += np.prod(v.shape)
		print(net.collect_params())
		print('NUMBER OF PARAMS:', num_params)
	except:
		logger = Logger(name, net, train_setting['early_stop_metric'], train_setting['early_stop_epoch'])
		net.initialize(init.Orthogonal(), ctx=args.gpus)
		print('Initialize the model')

	# net.hybridize()
	model_trainer = ModelTrainer(
		net = net,
		trainer = gluon.Trainer(
			net.collect_params(),
			mx.optimizer.Adam(
				learning_rate	= train_setting['lr'],
				multi_precision	= True,
				lr_scheduler	= mx.lr_scheduler.FactorScheduler(
					step			= train_setting['lr_decay_step'] * len(args.gpus),
					factor			= train_setting['lr_decay_factor'],
					stop_factor_lr	= 1e-6
				)
			),
			update_on_kvstore = False
		),
		clip_gradient = train_setting['clip_gradient'],
		logger = logger,
		ctx = args.gpus
	)

	train, eval, test, scaler = getattr(data.dataloader, dataset_setting['dataloader'])(settings)
	model_trainer.fit(
		begin_epoch = logger.best_epoch(),
		num_epochs	= args.epochs,
		train		= train,
		eval		= eval,
		test		= test,
		metrics		= [MAE(scaler), RMSE(scaler), IndexMAE(scaler, [0,1,2]), IndexRMSE(scaler, [0,1,2])],
	)

	net.load_parameters('%s-%04d.params' % (name, logger.best_epoch()), ctx=args.gpus)
	model_trainer.fit(
		begin_epoch	= 0,
		num_epochs	= 1,
		train		= None,
		eval		= eval,
		test		= test,
		metrics		= [MAE(scaler), RMSE(scaler), IndexMAE(scaler, [0,1,2]), IndexRMSE(scaler, [0,1,2])]
	)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', type=str)
	parser.add_argument('--epochs', type=int)
	parser.add_argument('--gpus', type=str)
	args = parser.parse_args()

	args.gpus = [mx.gpu(int(i)) for i in args.gpus.split(',')]
	main(args)