import torch.nn as nn
import numpy as np


class MultilayerPerceptron(nn.Module):
    """
    Feed-forward neural network with `feature_size` input units, `num_targets`
    output units, and hidden layers given by the list `hidden_layer_sizes`.
    The input layer and all hidden layers share the following generic structure

    .. math::

        \\text{dropout} \\Big( f \\big( \\text{norm}(W x + b) \\big) \\Big) \\text{,}

    where

    - :math:`x` is the input to the layer,
    - :math:`W` and :math:`b` are learnable weights,
    - :math:`\\text{norm}` is a placeholder for a normalization layer (leave
      empty for no normalization),
    - :math:`f` is a placeholder for an activation function (leave empty for no
      non-linearity),
    - :math:`\\text{dropout}` is a placeholder for a dropout layer (leave empty
      for no dropout).

    The output layer is not followed by normalization, non-linearity (this will
    be included in the loss function), nor dropout.
    """

    def __init__(self, feature_size, hidden_layer_sizes, num_targets, dropout_input=.0, dropout_hidden=.0, nonlinearity='Identity'):
        super().__init__()

        # linear layers
        self.linear_input = nn.Linear(feature_size, hidden_layer_sizes[0])
        self.linear_hidden_l = nn.ModuleList(
            [nn.Linear(s, spp) for s, spp in zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])]
        )
        self.linear_output = nn.Linear(hidden_layer_sizes[-1], num_targets)

        # normalization layers (placeholders)
        self.normalization_input = nn.Identity()
        self.normalization_hidden_l = nn.ModuleList(
            [nn.Identity() for _ in hidden_layer_sizes[1:]]
        )
        assert len(self.linear_hidden_l) == len(self.normalization_hidden_l), 'Something went wrong initializing the hidden layers.'

        # non-linearity and dropout (placeholders)
        self.nonlinearity = getattr(nn, nonlinearity)()
        self.dropout_input = nn.Dropout(p=dropout_input)
        self.dropout_hidden = nn.Dropout(p=dropout_hidden)
        self.num_weight_matrices = len(hidden_layer_sizes) + 1

    def forward(self, x):
        x = self.linear_input(x)
        x = self.normalization_input(x)
        x = self.nonlinearity(x)
        x = self.dropout_input(x)
        if len(self.linear_hidden_l) > 0:
            for linear_hidden, normalization_hidden in zip(self.linear_hidden_l, self.normalization_hidden_l):
                x = linear_hidden(x)
                x = normalization_hidden(x)
                x = self.nonlinearity(x)
                x = self.dropout_hidden(x)
        x = self.linear_output(x)
        return x


class ReLUNetworkLayerNorm(MultilayerPerceptron):
    """
    Child class of :class:`MultilayerPerceptron` where

    - normalization layers are set to :class:`~torch.nn.LayerNorm`,
    - non-linearity is set to :class:`~torch.nn.ReLU`,
    - dropout layers are set to :class:`~torch.nn.Dropout`,

    """

    def __init__(self, feature_size, hidden_layer_sizes, num_targets, dropout_input, dropout_hidden):
        super().__init__(feature_size, hidden_layer_sizes, num_targets)
        self.normalization_input = nn.LayerNorm(
            normalized_shape=self.linear_input.out_features,
            elementwise_affine=False
        )
        for i, linear_hidden in enumerate(self.linear_hidden_l):
            self.normalization_hidden_l[i] = nn.LayerNorm(
                normalized_shape=linear_hidden.out_features,
                elementwise_affine=False
            )
        self.nonlinearity = nn.ReLU()
        self.dropout_input = nn.Dropout(p=dropout_input)
        self.dropout_hidden = nn.Dropout(p=dropout_hidden)
        
class NetworkLayerNorm(MultilayerPerceptron):
    """
    Child class of :class:`MultilayerPerceptron` where

    - normalization layers are set to :class:`~torch.nn.LayerNorm`,
    - non-linearity is set to :class:`~torch.nn.__` which can be set by the argument nonlinearity,
    - dropout layers are set to :class:`~torch.nn.Dropout`,

    and the weights are initialized using :meth:`msra_initialization`.
    """
    def __init__(self, feature_size, hidden_layer_sizes, num_targets, dropout_input, dropout_hidden, 
    nonlinearity='ReLU'):
        super().__init__(feature_size, hidden_layer_sizes, num_targets)
        self.normalization_input = nn.LayerNorm(
            normalized_shape=self.linear_input.out_features,
            elementwise_affine=False
        )
        for i, linear_hidden in enumerate(self.linear_hidden_l):
            self.normalization_hidden_l[i] = nn.LayerNorm(
                normalized_shape=linear_hidden.out_features,
                elementwise_affine=False
            )
        self.nonlinearity = getattr(nn, nonlinearity if nonlinearity else 'ReLU')()
        self.dropout_input = nn.Dropout(p=dropout_input)
        self.dropout_hidden = nn.Dropout(p=dropout_hidden)
