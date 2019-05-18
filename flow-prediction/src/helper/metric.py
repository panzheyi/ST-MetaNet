import mxnet as mx
from mxnet import nd

class Metric:
	def __init__(self, name):
		self.name = name
		self.cnt = None
		self.loss = None
	
	def reset(self):
		self.cnt = None
		self.loss = None

class RMSE(Metric):
	def __init__(self, scaler, name='rmse'):
		super(RMSE, self).__init__(name)
		self.scaler = scaler
	
	def update(self, data, label):
		data = self.scaler.inverse_transform(data)
		label = self.scaler.inverse_transform(label)
		
		_cnt = data.size
		_loss = nd.sum((data - label) ** 2).as_in_context(mx.cpu())
		if self.cnt is None:
			self.cnt = 0
			self.loss = 0
		self.cnt += _cnt
		self.loss += _loss
	
	def get_value(self):
		return { self.name: nd.sqrt(self.loss / (self.cnt + 1e-8)) }

class MAE(Metric):
	def __init__(self, scaler, name='mae'):
		super(MAE, self).__init__(name)
		self.scaler = scaler
	
	def update(self, data, label):
		data = self.scaler.inverse_transform(data)
		label = self.scaler.inverse_transform(label)
		
		_cnt = data.size
		_loss = nd.sum(nd.abs(data - label)).as_in_context(mx.cpu())
		if self.cnt is None:
			self.cnt = _cnt
			self.loss = _loss
		else:
			self.cnt += _cnt
			self.loss += _loss
	
	def get_value(self):
		return { self.name: self.loss / (self.cnt + 1e-8) }

class IndexRMSE(Metric):
	def __init__(self, scaler, indices, name='rmse-index'):
		super(IndexRMSE, self).__init__(name)
		self.scaler = scaler
		self.indices = indices
	
	def update(self, data, label):
		data = self.scaler.inverse_transform(data)
		label = self.scaler.inverse_transform(label)
		
		_cnt = data[:,:,0].size
		_loss = nd.sum((data - label) ** 2, axis=2, exclude=True).as_in_context(mx.cpu())
		if self.cnt is None:
			self.cnt = _cnt
			self.loss = _loss
		else:
			self.cnt += _cnt
			self.loss += _loss
	
	def get_value(self):
		return { self.name: nd.sqrt((self.loss / (self.cnt + 1e-8))[self.indices]) }

class IndexMAE(Metric):
	def __init__(self, scaler, indices, name='mae-index'):
		super(IndexMAE, self).__init__(name)
		self.scaler = scaler
		self.indices = indices
	
	def update(self, data, label):
		data = self.scaler.inverse_transform(data)
		label = self.scaler.inverse_transform(label)
		
		_cnt = data[:,:,0].size
		_loss = nd.sum(nd.abs(data - label), axis=2, exclude=True).as_in_context(mx.cpu())
		if self.cnt is None:
			self.cnt = _cnt
			self.loss = _loss
		else:
			self.cnt += _cnt
			self.loss += _loss
	
	def get_value(self):
		return { self.name: (self.loss / (self.cnt + 1e-8))[self.indices] }