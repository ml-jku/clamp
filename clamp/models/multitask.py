from .models import DotProduct, MLPLayerNorm

from typing import List, Tuple


class Multitask(DotProduct):
    """
    Subclass of :class:`~biobert.models.models.DotProduct` to implement
    multitask models using the contrastive set up. This is achieved in the
    following way:

    - the embedding size is the number of training assays
    - the training assays are represented as 1-hot feature vectors
    - the assay encoder is the identity function
    - the compound encoder is then a multitask network

    Instead of iterating over compounds, predicting the activity for all
    training assays at once, and masking those for which activity is available,
    we iterate over <compound, assay, activity> triplets, and the dot product
    with the 1-hot assay vector works as the mask.
    """
    def __init__(
            self,
            compound_features_size: int,
            num_assays: int,
            **kwargs
    ):
        """
        Initialize class.

        Parameters
        ----------
        compound_features_size: int
            Input size of the compound encoder.
        num_assays: int
            Number of training assays.
        """
        super().__init__(
            compound_features_size=compound_features_size,
            assay_features_size=num_assays,
            embedding_size=num_assays,
            **kwargs
        )

    def _define_encoders(
            self,
            compound_layer_sizes: List[int],
            dropout_input: float,
            dropout_hidden: float, **kwargs
    ) -> Tuple[callable, callable]:
        """
        Define compound and assay encoders using the
        :meth:`~biobert.models.models.DotProduct._define_encoders` method of a
        meaningful parent class. However, the assay encoder is just initialized
        to dummy values and then replaced by the identity function.

        Parameters
        ----------
        compound_layer_sizes: list of int
            Sizes of the hidden layers of the compound encoder.
        dropout_input: float
            Dropout rate at the input layer.
        dropout_hidden: float
            Dropout rate at the hidden layers.

        Returns
        -------
        tuple of callable
            Compound encoder and identity function.
        """
        compound_encoder, _ = super()._define_encoders(
            compound_layer_sizes=compound_layer_sizes,
            assay_layer_sizes=[1],  # any dummy sizes
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden
        )

        return compound_encoder, lambda x: x



class MultitaskMLPLayerNorm(Multitask, MLPLayerNorm):
    pass
