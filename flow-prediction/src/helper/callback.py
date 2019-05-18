import os
import mxnet as mx
import numpy as np
import logging
import yaml

import time

class Speedometer:
	def __init__(self, title, epoch, frequent):
		self.title = title
		self.epoch = epoch
		self.frequent = frequent
		self.start = self.end = None
	
	def reset(self):
		self.start = self.tic = time.time()

	def finish(self, metrics):
		output_str = ''
		if metrics is not None:
			for metric in metrics:
				result = metric.get_value()
				for k, v in result.items():
					v = np.array_str(v.asnumpy())
					output_str += '\t' + k + ': ' + v
		self.end = time.time()
		print('%s\tEpoch[%d]\tTime:%.2fs%s' % (self.title, self.epoch, self.end-self.start, output_str))
		self.reset()
	
	def log_metrics(self, nbatch, metrics):
		if nbatch % self.frequent == 0:
			output_str = ''
			if metrics is not None:
				for metric in metrics:
					result = metric.get_value()
					for k, v in result.items():
						v = np.array_str(v.asnumpy())
						output_str += '\t' + k + ': ' + v

			time_spent = time.time() - self.tic
			self.tic = time.time()
			speed = self.frequent / time_spent
			print('%s\tEpoch[%d]\tBatch[%d]\tTime spent:%.2fs\tSpeed: %.2fbatch/s%s' %
					(self.title, self.epoch, nbatch, time_spent, speed, output_str))

class Logger:
	def __init__(self, name, net, early_stop_metric, early_stop_epoch):
		self.name = name
		self.net = net
		self.early_stop_metric = early_stop_metric
		self.early_stop_epoch = early_stop_epoch
		self.best = 0
		self.eval = []
		self.cnt = 0

	def log(self, epoch, metrics):
		self.cnt += 1

		result = {'epoch': epoch + 1}
		for metric in metrics:
			for k, v in metric.get_value().items():
				result[k] = np.asscalar(v.mean().asnumpy())

		self.eval.append(result)

		if self.eval[self.best][self.early_stop_metric] > self.eval[-1][self.early_stop_metric] - 1e-5:
			if self.best >= 0:
				try:
					os.remove('%s-%04d.params' % (self.name, self.eval[self.best]['epoch']))
				except:
					pass

			self.cnt = 0
			self.best = len(self.eval) - 1
			self.net.save_parameters('%s-%04d.params' % (self.name, self.eval[self.best]['epoch']))
			print('save model to %s-%04d.params' % (self.name, self.eval[self.best]['epoch']))
			
		self.dump()

		if  self.cnt >= self.early_stop_epoch:
			logging.info('Early stopping!')
			exit()
		
	def best_epoch(self):
		if self.best >= len(self.eval):
			return 0
		return self.eval[self.best]['epoch']

	def set_net(self, net):
		self.net = net
	
	@staticmethod
	def load(filename):
		with open(filename, 'r') as f:
			history = yaml.load(f)

		logger = Logger(history['name'], None, history['early_stop_metric'], history['early_stop_epoch'])
		logger.best = history['best']
		logger.eval = history['eval']
		return logger

	def dump(self):
		history = {}
		history['name'] = self.name
		history['early_stop_metric'] = self.early_stop_metric
		history['early_stop_epoch'] = self.early_stop_epoch
		history['best'] = self.best
		history['eval'] = self.eval
		history['best_result'] = self.eval[self.best]
		with open('%s.yaml' % self.name, 'w') as f:
			yaml.dump(history, f)

	