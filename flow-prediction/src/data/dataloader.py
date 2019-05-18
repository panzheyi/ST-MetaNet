import os
import h5py
import logging
import mxnet as mx
import numpy as np
import pandas as pd
import math

from data import utils
from config import DATA_PATH, TRAIN_PROP, EVAL_PROP

def get_grids():
	x0, y0 = 116.25, 39.83
	x1, y1 = 116.64, 40.12
	rows, cols = 32, 32
	size_x, size_y = (x1 - x0) / rows, (y1 - y0) / cols

	grids = []
	for r in range(rows):
		for c in range(cols):
			_x0, _y0 = (rows - r - 1) * size_x + x0, c * size_y + y0
			_x1, _y1 = (rows - r) * size_x + x0, (c + 1) * size_y + y0
			grids += [ [[_y1, _x0], [_y1, _x1], [_y0, _x1], [_y0, _x0]] ]

	print(grids)

def get_graph():
	adj_feature = utils.load_h5(os.path.join(DATA_PATH, 'BJ_GRAPH.h5'), ['data'])
	src, dst = np.where(np.sum(adj_feature, axis=2) > 0)
	
	values = adj_feature[src, dst]
	adj_feature = (adj_feature - np.mean(values, axis=0)) / (np.std(values, axis=0) + 1e-8)

	return adj_feature, src, dst

def get_geo_feature(dataset):
	geo = utils.load_h5(os.path.join(DATA_PATH, 'BJ_FEATURE.h5'), ['embeddings'])
	row, col, _ = geo.shape
	geo = np.reshape(geo, (row * col, -1))

	geo = (geo - np.mean(geo, axis=0)) / (np.std(geo, axis=0) + 1e-8)	
	return geo

def dataloader(dataset):
	data = utils.load_h5(os.path.join(DATA_PATH, 'BJ_FLOW.h5'), ['data'])
	days, hours, rows, cols, _ = data.shape

	data = np.reshape(data, (days * hours, rows * cols, -1))

	n_timestamp = data.shape[0]
	num_train = int(n_timestamp * TRAIN_PROP)
	num_eval = int(n_timestamp * EVAL_PROP)
	num_test = n_timestamp - num_train - num_eval

	return data[:num_train], data[num_train: num_train + num_eval], data[-num_test:]

def dataiter_all_sensors_seq2seq(flow, scaler, setting, shuffle=True):
	dataset = setting['dataset']
	training = setting['training']

	mask = np.sum(flow, axis=(1,2)) > 5000

	flow = scaler.transform(flow)

	n_timestamp, num_nodes, _ = flow.shape

	timespan = (np.arange(n_timestamp) % 24) / 24
	timespan = np.tile(timespan, (1, num_nodes, 1)).T
	flow = np.concatenate((flow, timespan), axis=2)

	geo_feature = get_geo_feature(dataset)

	input_len = dataset['input_len']
	output_len = dataset['output_len']
	feature, data, label  = [], [], []
	for i in range(n_timestamp - input_len - output_len + 1):
		if mask[i + input_len: i + input_len + output_len].sum() != output_len:
			continue
			
		data.append(flow[i: i + input_len])
		label.append(flow[i + input_len: i + input_len + output_len])
		feature.append(geo_feature)

		if i % 1000 == 0:
			logging.info('Processing %d timestamps', i)
			# if i > 0: break

	data = mx.nd.array(np.stack(data)) # [B, T, N, D]
	label = mx.nd.array(np.stack(label)) # [B, T, N, D]
	feature = mx.nd.array(np.stack(feature)) # [B, N, D]

	logging.info('shape of feature: %s', feature.shape)
	logging.info('shape of data: %s', data.shape)
	logging.info('shape of label: %s', label.shape)

	from mxnet.gluon.data import ArrayDataset, DataLoader
	return DataLoader(
		ArrayDataset(feature, data, label),
		shuffle		= shuffle,
		batch_size	= training['batch_size'],
		num_workers	= 4,
		last_batch	= 'rollover',
	)

def dataloader_all_sensors_seq2seq(setting):
	train, eval, test = dataloader(setting['dataset'])
	scaler = utils.Scaler(train)

	return dataiter_all_sensors_seq2seq(train, scaler, setting), \
		   dataiter_all_sensors_seq2seq(eval, scaler, setting, shuffle=False), \
		   dataiter_all_sensors_seq2seq(test, scaler, setting, shuffle=False), \
		   scaler