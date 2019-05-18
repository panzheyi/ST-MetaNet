from mxnet import nd
from mxnet.gluon import nn, Block

class MLP(nn.HybridSequential):
    """ Multilayer perceptron. """
    def __init__(self, hiddens, act_type, out_act, weight_initializer=None, **kwargs):
        """
        The initializer.

        Parameters
        ----------
        hiddens: list
            The list of hidden units of each dense layer.
        act_type: str
            The activation function after each dense layer.
        out_act: bool
            Weather apply activation function after last dense layer.
        """
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            for i, h in enumerate(hiddens):
                activation = None if i == len(hiddens) - 1 and not out_act else act_type
                self.add(nn.Dense(h, activation=activation, weight_initializer=weight_initializer))

class MetaDense(Block):
    """ The meta-dense layer. """
    def __init__(self, input_hidden_size, output_hidden_size, meta_hiddens, prefix=None):
        """
        The initializer.

        Parameters
        ----------
        input_hidden_size: int
            The hidden size of the input.
        output_hidden_size: int
            The hidden size of the output.
        meta_hiddens: list of int
            The list of hidden units of meta learner (a MLP).
        """
        super(MetaDense, self).__init__(prefix=prefix)
        self.input_hidden_size = input_hidden_size
        self.output_hidden_size = output_hidden_size
        self.act_type = 'sigmoid'
        
        with self.name_scope():
            self.w_mlp = MLP(meta_hiddens + [self.input_hidden_size * self.output_hidden_size,], act_type=self.act_type, out_act=False, prefix='w_')
            self.b_mlp = MLP(meta_hiddens + [1,], act_type=self.act_type, out_act=False, prefix='b_')

    def forward(self, feature, data):
        """ Forward process of a MetaDense layer

        Parameters
        ----------
        feature: NDArray with shape [n, d]
        data: NDArray with shape [n, b, input_hidden_size]

        Returns
        -------
        output: NDArray with shape [n, b, output_hidden_size]
        """
        weight = self.w_mlp(feature) # [n, input_hidden_size * output_hidden_size]
        weight = nd.reshape(weight, (-1, self.input_hidden_size, self.output_hidden_size))
        bias = nd.reshape(self.b_mlp(feature), shape=(-1, 1, 1)) # [n, 1, 1]
        return nd.batch_dot(data, weight) + bias