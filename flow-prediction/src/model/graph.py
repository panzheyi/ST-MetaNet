import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn, Block

import dgl
from dgl import DGLGraph
from functools import partial
from config import MODEL

class Graph(Block):
    """ The base class of GAT and MetaGAT. We implement the methods based on DGL library. """

    @staticmethod
    def create(graph_type, dist, src, dst, hidden_size, prefix):
        """ create a graph. """
        if graph_type == 'None': return None
        elif graph_type == 'GAT': return GAT(dist, src, dst, hidden_size, prefix=prefix)
        elif graph_type == 'MetaGAT': return MetaGAT(dist, src, dst, hidden_size, prefix=prefix)
        else: raise Exception('Unknow graph: %s' % graph_type)

    @staticmethod
    def create_graphs(graph_type, graph, hidden_size, prefix):
        """ Create a list of graphs according to graph_type & graph. """
        if graph_type == 'None': return None
        dist, src, dst = graph
        return [
            Graph.create(graph_type, dist, src, dst, hidden_size, prefix + 'graph_'),
        ]

    def __init__(self, dist, src, dst, hidden_size, prefix=None):
        super(Graph, self).__init__(prefix=prefix)
        self.dist = dist
        self.src = src
        self.dst = dst
        self.hidden_size = hidden_size
        self.dist = mx.nd.array(dist[src, dst])

        # create graph
        self.num_nodes = n = dist.shape[0]

        self.ctx = []
        self.graph_on_ctx = []

        self.init_model()    

    def build_graph_on_ctx(self, ctx):
        g = DGLGraph()
        g.set_n_initializer(dgl.init.zero_initializer)
        g.add_nodes(self.num_nodes)
        g.add_edges(self.src, self.dst)
        g.edata['dist'] = self.dist.as_in_context(ctx)
        self.graph_on_ctx.append(g)
        self.ctx.append(ctx)
    
    def get_graph_on_ctx(self, ctx):
        if ctx not in self.ctx:
            self.build_graph_on_ctx(ctx)
        return self.graph_on_ctx[self.ctx.index(ctx)]

    def forward(self, state, feature): # first dimension of state & feature should be num_nodes
        g = self.get_graph_on_ctx(state.context)
        g.ndata['state'] = state
        g.ndata['feature'] = feature        
        g.update_all(self.msg_edge, self.msg_reduce)
        state = g.ndata.pop('new_state')
        return state

    def init_model(self):
        raise NotImplementedError("To be implemented")

    def msg_edge(self, edge):
        """ Messege passing across edge.
        More detail usage please refers to the manual of DGL library.

        Parameters
        ----------
        edge: a dictionary of edge data.
            edge.src['state'] and edge.dst['state']: hidden states of the nodes, which is NDArrays with shape [e, b, t, d] or [e, b, d]
            edge.src['feature'] and  edge.dst['state']: features of the nodes, which is NDArrays with shape [e, d]
            edge.data['dist']: distance matrix of the edges, which is a NDArray with shape [e, d]

        Returns
        -------
            A dictionray of messages
        """
        raise NotImplementedError("To be implemented")

    def msg_reduce(self, node):
        raise NotImplementedError("To be implemented")
        
class GAT(Graph):
    """ Graph attention. """
    def __init__(self, dist, src, dst, hidden_size, prefix=None):
        super(GAT, self).__init__(dist, src, dst, hidden_size, prefix)

    def init_model(self):
        with self.name_scope():
            self.dense = nn.Dense(self.hidden_size, flatten=False)
            self.weight = self.params.get('weight', shape=(1,1))
    
    def msg_edge(self, edge):
        dist = edge.data['dist']
        while len(dist.shape) < len(edge.src['state'].shape):
            dist = nd.expand_dims(dist, axis=1)
        dist = nd.broadcast_to(dist, shape=edge.src['state'].shape[:-1]+(0,))
        state = nd.concat(edge.src['state'], edge.dst['state'], dist, dim=-1)
        alpha = nd.LeakyReLU(self.dense(state))
        return { 'alpha': alpha, 'state': edge.src['state'] }

    def msg_reduce(self, node):
        state = node.mailbox['state']
        alpha = node.mailbox['alpha']
        alpha = nd.softmax(alpha, axis=1)

        new_state = nd.relu(nd.sum(alpha * state, axis=1)) * nd.sigmoid(self.weight.data(state.context))
        return { 'new_state': new_state }

class MetaGAT(Graph):
    """ Meta Graph Attention. """

    def __init__(self, dist, src, dst, hidden_size, prefix=None):
        super(MetaGAT, self).__init__(dist, src, dst, hidden_size, prefix)

    def init_model(self):
        from model.basic_structure import MLP
        with self.name_scope():
            self.w_mlp = MLP(MODEL['meta_hiddens'] + [self.hidden_size * self.hidden_size * 2,], 'sigmoid', False)
            self.weight = self.params.get('weight', shape=(1,1))
    
    def msg_edge(self, edge):
        state = nd.concat(edge.src['state'], edge.dst['state'], dim=-1)
        feature = nd.concat(edge.src['feature'], edge.dst['feature'], edge.data['dist'], dim=-1)

        # generate weight by meta-learner
        weight = self.w_mlp(feature)
        weight = nd.reshape(weight, shape=(-1, self.hidden_size * 2, self.hidden_size))

        # reshape state to [n, b * t, d] for batch_dot (currently mxnet only support batch_dot for 3D tensor)
        shape = state.shape
        state = nd.reshape(state, shape=(shape[0], -1, shape[-1]))

        alpha = nd.LeakyReLU(nd.batch_dot(state, weight))

        # reshape alpha to [n, b, t, d]
        alpha = nd.reshape(alpha, shape=shape[:-1] + (self.hidden_size,))
        return { 'alpha': alpha, 'state': edge.src['state'] }

    def msg_reduce(self, node):
        state = node.mailbox['state']
        alpha = node.mailbox['alpha']
        alpha = nd.softmax(alpha, axis=1)

        new_state = nd.relu(nd.sum(alpha * state, axis=1)) * nd.sigmoid(self.weight.data(state.context))
        return { 'new_state': new_state }
