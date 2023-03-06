from biobert.models import scaled
import torch


compound_features_size, assay_features_size, embedding_size = 2, 3, 4
num_samples = 5


def test_scaled_mlplayernorm():
    scaled_mlp = scaled.ScaledMLPLayerNorm(
        compound_features_size=compound_features_size,
        assay_features_size=assay_features_size,
        embedding_size=embedding_size,
        compound_layer_sizes=[3, 2],
        assay_layer_sizes=[2, 3],
        dropout_input=0.,
        dropout_hidden=0.
    )

    # sparse forward pass
    compound_features = torch.rand(num_samples, compound_features_size)
    assay_features = torch.rand(num_samples, assay_features_size)

    probs = scaled_mlp.forward(compound_features, assay_features)
    assert probs.shape == torch.Size([num_samples])

    # dense forward pass
    num_compounds, num_assays = 3, 5
    compound_features = torch.rand(num_compounds, compound_features_size)
    assay_features = torch.rand(num_assays, assay_features_size)

    probs = scaled_mlp.forward_dense(compound_features, assay_features)
    assert probs.shape == torch.Size([num_compounds, num_assays])
