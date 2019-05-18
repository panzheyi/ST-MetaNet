import math
import random
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon import Block, nn

from model.cell import RNNCell
from model.graph import Graph
from model.basic_structure import MLP

class Encoder(Block):
    """ Seq2Seq encoder. """
    def __init__(self, cells, graphs, prefix=None):
        super(Encoder, self).__init__(prefix=prefix)

        self.cells = cells
        for cell in cells:
            self.register_child(cell)

        self.graphs = graphs
        for graph in graphs:
            if graph is not None:
                for g in graph:
                    if g is not None:
                        self.register_child(g)

    def forward(self, feature, data):
        """ Encode the temporal sequence sequence.

        Parameters
        ----------
        feature: a NDArray with shape [n, d].
        data: a NDArray with shape [n, b, t, d].        

        Returns
        -------
        states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.
        """

        _, batch, seq_len, _ = data.shape
        states = []
        for depth, cell in enumerate(self.cells):
            # rnn unroll
            data, state = cell(feature, data, None)
            states.append(state)

            # graph attention
            if self.graphs[depth] != None:
                _data = 0
                for g in self.graphs[depth]:
                    _data = _data + g(data, feature)
                data = _data

        return states

class Decoder(Block):
    """ Seq2Seq decoder. """
    def __init__(self, cells, graphs, input_dim, output_dim, use_sampling, cl_decay_steps, prefix=None):
        super(Decoder, self).__init__(prefix=prefix)
        self.cells = cells
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_sampling = use_sampling
        self.global_steps = 0.0
        self.cl_decay_steps = float(cl_decay_steps)

        for cell in cells:
            self.register_child(cell)

        self.graphs = graphs
        for graph in graphs:
            if graph is not None:
                for g in graph:
                    if g is not None:
                        self.register_child(g)
        
        # initialize projection layer for the output
        with self.name_scope():
            self.proj = nn.Dense(output_dim, prefix='proj_')

    def sampling(self):
        """ Schedule sampling: sampling the ground truth. """
        threshold = self.cl_decay_steps / (self.cl_decay_steps + math.exp(self.global_steps / self.cl_decay_steps))
        return float(random.random() < threshold)

    def forward(self, feature, label, begin_states, is_training):
        ''' Decode the hidden states to a temporal sequence.

        Parameters
        ----------
        feature: a NDArray with shape [n, d].
        label: a NDArray with shape [n, b, t, d].
        begin_states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.
        is_training: bool
        
        Returns
        -------
            outputs: the prediction, which is a NDArray with shape [n, b, t, d]
        '''
        ctx = label.context

        num_nodes, batch_size, seq_len, _ = label.shape 
        aux = label[:,:,:, self.output_dim:] # [n,b,t,d]
        label = label[:,:,:, :self.output_dim] # [n,b,t,d]
        
        go = nd.zeros(shape=(num_nodes, batch_size, self.input_dim), ctx=ctx)
        output, states = [], begin_states

        for i in range(seq_len):
            # get next input
            if i == 0: data = go
            else:
                prev = nd.concat(output[i - 1], aux[:,:,i - 1], dim=-1)
                truth = nd.concat(label[:,:,i - 1], aux[:,:,i - 1], dim=-1)
                if is_training and self.use_sampling: value = self.sampling()
                else: value = 0
                data = value * truth + (1 - value) * prev

            # unroll 1 step
            for depth, cell in enumerate(self.cells):
                data, states[depth] = cell.forward_single(feature, data, states[depth])
                if self.graphs[depth] is not None:
                    _data = 0
                    for g in self.graphs[depth]:
                        _data = _data + g(data, feature)
                    data = _data / len(self.graphs[depth])

            # append feature to output
            _feature = nd.expand_dims(feature, axis=1) # [n, 1, d]
            _feature = nd.broadcast_to(_feature, shape=(0, batch_size, 0)) # [n, b, d]
            data = nd.concat(data, _feature, dim=-1) # [n, b, t, d]

            # proj output to prediction
            data = nd.reshape(data, shape=(num_nodes * batch_size, -1))
            data = self.proj(data)
            data = nd.reshape(data, shape=(num_nodes, batch_size, -1))
            
            output.append(data)

        output = nd.stack(*output, axis=2)
        return output

class Seq2Seq(Block):
    def __init__(self,
                 geo_hiddens,
                 rnn_type, rnn_hiddens,
                 graph_type, graph,
                 input_dim, output_dim,
                 use_sampling,
                 cl_decay_steps,
                 prefix=None):
        """ Initializer.

        Parameters
        ----------
        geo_hiddens: list of int
            the hidden units of NMK-learner.
        rnn_type: list of str
            the types of rnn cells (denoting GRU or MetaGRU).
        rnn_hiddens: list of int
            the hidden units for each rnn layer.
        graph_type: list of str
            the types of graph attention (denoting GAT or MetaGAT).
        graph: the graph structure
        input_dim: int
        output_dim: int
        use_sampling: bool
            whether use schedule sampling during training process.
        cl_decay_steps: int
            decay steps in schedule sampling.
        """
        super(Seq2Seq, self).__init__(prefix=prefix)

        # initialize encoder
        with self.name_scope():
            encoder_cells = []
            encoder_graphs = []
            for i, hidden_size in enumerate(rnn_hiddens):
                pre_hidden_size = input_dim if i == 0 else rnn_hiddens[i - 1]
                c = RNNCell.create(rnn_type[i], pre_hidden_size, hidden_size, prefix='encoder_c%d_' % i)
                g = Graph.create_graphs('None' if i == len(rnn_hiddens) - 1 else graph_type[i], graph, hidden_size, prefix='encoder_g%d_' % i)
                encoder_cells.append(c)
                encoder_graphs.append(g)
        self.encoder = Encoder(encoder_cells, encoder_graphs)

        # initialize decoder
        with self.name_scope():
            decoder_cells = []
            decoder_graphs = []
            for i, hidden_size in enumerate(rnn_hiddens):
                pre_hidden_size = input_dim if i == 0 else rnn_hiddens[i - 1]
                c = RNNCell.create(rnn_type[i], pre_hidden_size, hidden_size, prefix='decoder_c%d_' % i)
                g = Graph.create_graphs(graph_type[i], graph, hidden_size, prefix='decoder_g%d_' % i)
                decoder_cells.append(c)
                decoder_graphs.append(g)
        self.decoder = Decoder(decoder_cells, decoder_graphs, input_dim, output_dim, use_sampling, cl_decay_steps)

        # initalize geo encoder network (node meta knowledge learner)
        self.geo_encoder = MLP(geo_hiddens, act_type='relu', out_act=True, prefix='geo_encoder_')
    
    def meta_knowledge(self, feature):
        return self.geo_encoder(nd.mean(feature, axis=0))
        
    def forward(self, feature, data, label, mask, is_training):
        """ Forward the seq2seq network.

        Parameters
        ----------
        feature: NDArray with shape [b, n, d].
            The features of each node. 
        data: NDArray with shape [b, t, n, d].
            The flow readings.
        label: NDArray with shape [b, t, n, d].
            The flow labels.
        is_training: bool.


        Returns
        -------
        loss: loss for gradient descent.
        (pred, label): each of them is a NDArray with shape [n, b, t, d].

        """
        data = nd.transpose(data, axes=(2, 0, 1, 3)) # [n, b, t, d]
        label = nd.transpose(label, axes=(2, 0, 1, 3)) # [n, b, t, d]
        mask = nd.transpose(mask, axes=(2, 0, 1, 3)) # [n, b, t, d]

        # geo-feature embedding (NMK Learner)
        feature = self.geo_encoder(nd.mean(feature, axis=0)) # shape=[n, d]

        # seq2seq encoding process
        states = self.encoder(feature, data)

        # seq2seq decoding process
        output = self.decoder(feature, label, states, is_training) # [n, b, t, d]
             
        # loss calculation
        label = label[:,:,:,:self.decoder.output_dim]
        output = output * mask
        label = label * mask

        loss = nd.mean(nd.abs(output - label), axis=1, exclude=True)
        return loss, [output, label, mask]

def net(settings):
    from data.dataloader import get_geo_feature
    _, graph = get_geo_feature(settings['dataset'])

    net = Seq2Seq(
        geo_hiddens = settings['model']['geo_hiddens'],
        rnn_type    = settings['model']['rnn_type'],
        rnn_hiddens = settings['model']['rnn_hiddens'],
        graph_type  = settings['model']['graph_type'],
        graph       = graph,
        input_dim   = settings['dataset']['input_dim'],
        output_dim  = settings['dataset']['output_dim'],
        use_sampling    = settings['training']['use_sampling'],
        cl_decay_steps  = settings['training']['cl_decay_steps'],
        prefix      = settings['model']['type'] + "_"
    )
    return net