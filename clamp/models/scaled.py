from .models import DotProduct, MLPLayerNorm, TransformerLN, LSTMAssayEncoderLN, Linear
from clamp.models.perceptron import NetworkLayerNorm, MultilayerPerceptron
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class Scaled(DotProduct):
    """
    Subclass of :class:`~biobert.models.models.DotProduct` with a learnable
    temperature parameter which scales the preactivactions.
    """

    def __init__(
            self,
            compound_features_size: int,
            assay_features_size: int,
            embedding_size: int,
            **kwargs
    ) -> None:
        """
        Initialize class with an additional scale parameter.

        Parameters
        ----------
        compound_features_size: int
            Input size of the compound encoder.
        assay_features_size: int
            Input size of the assay encoder.
        embedding_size: int
            Size of the association space.
        """
        super().__init__(
            compound_features_size=compound_features_size,
            assay_features_size=assay_features_size,
            embedding_size=embedding_size,
            **kwargs
        )

        # clip scales by 1/0.07 = 14.2
        # clip: self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # clipped to prevent scaling the logits by more than a hundred --> tau = max(4.6, tau)
        # this parameter is clipped max / min ...
        # l2_normed = layer_normed * 1/sqrt(embedding_size)
        # so additionally we can scale by e^...

        # we scale by e^0.04 
        # clip scales by 14

        self.scale = nn.Parameter(
           torch.ones([1]) * -np.log(np.sqrt(self.embedding_size)*(1/30)) # e^scale = beta = 1/np.sqrt(embedding_size)
        )

        #self.scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 

    def forward(
            self,
            compound_features: torch.Tensor,
            assay_features: torch.Tensor
    ) -> torch.Tensor:
        """
        In the end scale the preactivations.
        """
        preactivations = super().forward(
            compound_features,
            assay_features
        )
        scaled_preactivations = self.scale.exp() * preactivations
        return scaled_preactivations

    def forward_dense(
            self,
            compound_features: torch.Tensor,
            assay_features: torch.Tensor
    ) -> torch.Tensor:
        """
        In the end scale the preactivations.
        """
        preactivations = super().forward_dense(
            compound_features,
            assay_features
        )
        scaled_preactivations = self.scale.exp() * preactivations
        return scaled_preactivations

class ScaledLinear(Scaled, Linear):
    pass


class ScaledMLPLayerNorm(Scaled, MLPLayerNorm):
    pass

class ScaledTransformerLN(Scaled, TransformerLN):
    pass

class ScaledLSTMAssayEncoderLN(Scaled, LSTMAssayEncoderLN):
    pass
