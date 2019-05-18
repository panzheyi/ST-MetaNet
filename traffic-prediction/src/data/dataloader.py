import os
import h5py
import logging
import mxnet as mx
import numpy as np
import pandas as pd
import math

from data import utils
from config import DATA_PATH, NUM_NODES

def get_geo_feature(dataset):
	n_neighbors = dataset['n_neighbors']

	# get locations
	loc = utils.sensor_location()
	loc = (loc - np.mean(loc, axis=0)) / np.std(loc, axis=0)

	# get distance matrix
	dist, e_in, e_out = utils.distant_matrix(n_neighbors)

	# normalize distance matrix
	n = loc.shape[0]
	edge = np.zeros((n, n))
	for i in range(n):
		for j in range(n_neighbors):
			edge[e_in[i][j], i] = edge[i, e_out[i][j]] = 1
	dist[edge == 0] = np.inf

	values = dist.flatten()
	values = values[values != np.inf]
	dist_mean = np.mean(values)
	dist_std = np.std(values)
	dist = np.exp(-(dist - dist_mean) / dist_std)

	# merge features
	features = []
	for i in range(n):
		f = np.concatenate([loc[i], dist[e_in[i], i], dist[i, e_out[i]]])
		features.append(f)
	features = np.stack(features)
	return features, (dist, e_in, e_out)

def dataloader(dataset):
	data = pd.read_hdf(os.path.join(DATA_PATH, 'df_highway_2012_4mon_sample.h5'))
	
	n_timestamp = data.shape[0]

	num_train = int(n_timestamp * dataset['train_prop'])
	num_eval = int(n_timestamp * dataset['eval_prop'])
	num_test = n_timestamp - num_train - num_eval

	train = data[:num_train].copy()
	eval = data[num_train: num_train + num_eval].copy()
	test = data[-num_test:].copy()

	return train, eval, test


def dataiter_all_sensors_seq2seq(df, scaler, setting, shuffle=True):
	dataset = setting['dataset']
	training = setting['training']

	df_fill = utils.fill_missing(df)
	df_fill = scaler.transform(df_fill)

	n_timestamp = df_fill.shape[0]
	data_list = [np.expand_dims(df_fill.values, axis=-1)]

	# time in day
	time_idx = (df_fill.index.values - df_fill.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
	time_in_day = np.tile(time_idx, [1, NUM_NODES, 1]).transpose((2, 1, 0))
	data_list.append(time_in_day)

	# day in week
	day_in_week = np.zeros(shape=(n_timestamp, NUM_NODES, 7))
	day_in_week[np.arange(n_timestamp), :, df_fill.index.dayofweek] = 1
	data_list.append(day_in_week)

	# temporal feature
	temporal_feature = np.concatenate(data_list, axis=-1)

	geo_feature, _ = get_geo_feature(dataset)

	input_len = dataset['input_len']
	output_len = dataset['output_len']
	feature, data, mask, label  = [], [], [], []
	for i in range(n_timestamp - input_len - output_len + 1):
		data.append(temporal_feature[i: i + input_len])

		_mask = np.array(df.iloc[i + input_len: i + input_len + output_len] > 1e-5, dtype=np.float32)
		mask.append(_mask)

		label.append(temporal_feature[i + input_len: i + input_len + output_len])
		
		feature.append(geo_feature)

		if i % 1000 == 0:
			logging.info('Processing %d timestamps', i)
			# if i > 0: break

	data = mx.nd.array(np.stack(data))
	label = mx.nd.array(np.stack(label))
	mask = mx.nd.array(np.expand_dims(np.stack(mask), axis=3))
	feature = mx.nd.array(np.stack(feature))

	logging.info('shape of feature: %s', feature.shape)
	logging.info('shape of data: %s', data.shape)
	logging.info('shape of mask: %s', mask.shape)
	logging.info('shape of label: %s', label.shape)

	from mxnet.gluon.data import ArrayDataset, DataLoader
	return DataLoader(
		ArrayDataset(feature, data, label, mask),
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