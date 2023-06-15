import torch.nn as nn
import numpy as np


def lecun_initialization(m):
    """
    LeCun initialization of the weights of a :class:`torch.nn.Module` (a layer),
    that is, the weights of the layer are :math:`\\mathbf{W} \\sim N(0, 1 / D)`,
    where :math:`D` is the incoming dimension. For more details see this
    blogpost_ and `LeCun's paper`_.

    .. _blogpost: https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/weight_initialization_activation_functions/#lecun-initialization-normalize-variance
    .. _`LeCun's paper`: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

    Parameters
    ----------
    m: :class:`torch.nn.Module`
        Module (layer) whose weights should be normalized.
    """
    nn.init.normal_(m.weight, mean=0., std=np.sqrt(1. / m.in_features))
    nn.init.zeros_(m.bias)


def msra_initialization(m):
    """
    MSRA initialization of the weights of a :class:`torch.nn.Module` (a layer),
    that is, the weights of the layer are :math:`\\mathbf{W} \\sim N(0, 2 / D)`,
    where :math:`D` is the incoming dimension. For more details see
    `He's paper`_.

    .. _`He's paper`: https://arxiv.org/abs/1502.01852

    Parameters
    ----------
    m: :class:`torch.nn.Module`
        Module (layer) whose weights should be normalized.
    """
    nn.init.normal_(m.weight, mean=0., std=np.sqrt(2. / m.in_features))
    nn.init.zeros_(m.bias)


def kaiming_normal_initialization(m):
    """
    Equivalent to :meth:`msra_initialization`, but using
    :meth:`pytorch.nn.init.kaiming_normal_`, which can be more general.
    How it is used here, it should be the same. For testing.

    Parameters
    ----------
    m: :class:`torch.nn.Module`
        Module (layer) whose weights should be normalized.
    """
    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    nn.init.zeros_(m.bias)


class LogisticRegression(nn.Module):
    """
    Logistic regression with `feature_size` input units and `num_targets`
    output units. It does not have a non-linearity because this will be
    included in the loss function.
    """

    def __init__(self, feature_size, num_targets):
        super().__init__()
        self.linear = nn.Linear(feature_size, num_targets)

    def forward(self, x):
        x = self.linear(x)
        return x


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

    def initialize_weights(self, init):
        """
        Initialize all the weights using the method `init`.
        """
        init(self.linear_input)
        if len(self.linear_hidden_l) > 0:
            for i, _ in enumerate(self.linear_hidden_l):
                init(self.linear_hidden_l[i])
        init(self.linear_output)


class SELUNetwork(MultilayerPerceptron):
    """
    Child class of :class:`MultilayerPerceptron` where

    - normalization layers are left empty (no explicit normalization),
    - non-linearity is set to :class:`~torch.nn.SELU`,
    - dropout layers are set to :class:`~torch.nn.AlphaDropout`,

    and the weights are initialized using :meth:`lecun_initialization`.
    """

    def __init__(self, feature_size, hidden_layer_sizes, num_targets, dropout_input, dropout_hidden):
        super().__init__(feature_size, hidden_layer_sizes, num_targets)
        self.nonlinearity = nn.SELU()
        self.dropout_input = nn.AlphaDropout(p=dropout_input)
        self.dropout_hidden = nn.AlphaDropout(p=dropout_hidden)
        self.initialize_weights(init=lecun_initialization)


class ReLUNetworkBatchNorm(MultilayerPerceptron):
    """
    Child class of :class:`MultilayerPerceptron` where

    - normalization layers are set to :class:`~torch.nn.BatchNorm1d`,
    - non-linearity is set to :class:`~torch.nn.ReLU`,
    - dropout layers are set to :class:`~torch.nn.Dropout`,

    and the weights are initialized using :meth:`msra_initialization`.
    """

    def __init__(self, feature_size, hidden_layer_sizes, num_targets, dropout_input, dropout_hidden):
        super().__init__(feature_size, hidden_layer_sizes, num_targets)
        self.normalization_input = nn.BatchNorm1d(num_features=self.linear_input.out_features)
        for i, linear_hidden in enumerate(self.linear_hidden_l):
            self.normalization_hidden_l[i] = nn.BatchNorm1d(num_features=linear_hidden.out_features)
        self.nonlinearity = nn.ReLU()
        self.dropout_input = nn.Dropout(p=dropout_input)
        self.dropout_hidden = nn.Dropout(p=dropout_hidden)
        self.initialize_weights(init=msra_initialization)

class ReLUNetworkLayerNorm(MultilayerPerceptron):
    """
    Child class of :class:`MultilayerPerceptron` where

    - normalization layers are set to :class:`~torch.nn.LayerNorm`,
    - non-linearity is set to :class:`~torch.nn.ReLU`,
    - dropout layers are set to :class:`~torch.nn.Dropout`,

    and the weights are initialized using :meth:`msra_initialization`.
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
        self.initialize_weights(init=msra_initialization)

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
        self.initialize_weights(init=msra_initialization)