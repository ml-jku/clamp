from biobert.models import multitask
import torch


compound_features_size, assay_features_size = 2, 3
num_samples = 5


def test_onesided_mlplayernorm():
    mt_mlp = multitask.MultitaskMLPLayerNorm(
        compound_features_size=compound_features_size,
        num_assays=assay_features_size,
        compound_layer_sizes=[3, 2],
        dropout_input=0.,
        dropout_hidden=0.
    )

    # sparse forward pass
    compound_features = torch.rand(num_samples, compound_features_size)
    assay_features = torch.rand(num_samples, assay_features_size)

    probs = mt_mlp.forward(compound_features, assay_features)
    assert probs.shape == torch.Size([num_samples])

    # dense forward pass
    num_compounds, num_assays = 3, 5
    compound_features = torch.rand(num_compounds, compound_features_size)
    assay_features = torch.rand(num_assays, assay_features_size)

    probs = mt_mlp.forward_dense(compound_features, assay_features)
    assert probs.shape == torch.Size([num_compounds, num_assays])

    # assay encoder is the identity
    assay_embeddings = mt_mlp.assay_encoder(assay_features)
    assert assay_features.equal(assay_embeddings)
