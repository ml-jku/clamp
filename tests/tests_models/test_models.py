from biobert.models import models
import torch
import pytest


compound_features_size, assay_features_size, embedding_size = 2, 3, 4
num_samples = 5


def test_dot_product():
    with pytest.raises(NotImplementedError):
        models.DotProduct(
            compound_features_size=compound_features_size,
            assay_features_size=assay_features_size,
            embedding_size=embedding_size
        )


def test_dummy():
    dummy = models.Dummy(
        compound_features_size=compound_features_size,
        assay_features_size=assay_features_size,
        embedding_size=embedding_size
    )

    compound_features = torch.rand(num_samples, compound_features_size)
    assay_features = torch.rand(num_samples, assay_features_size)

    output = dummy(compound_features, assay_features)
    assert output.shape == torch.Size([num_samples])


def test_linear():
    linear = models.Linear(
        compound_features_size=compound_features_size,
        assay_features_size=assay_features_size,
        embedding_size=6
    )

    # sparse forward pass
    compound_features = torch.rand(num_samples, compound_features_size)
    assay_features = torch.rand(num_samples, assay_features_size)

    probs = linear(compound_features, assay_features)
    assert probs.shape == torch.Size([num_samples])

    # dense forward pass
    num_compounds, num_assays = 3, 5
    compound_features = torch.rand(num_compounds, compound_features_size)
    assay_features = torch.rand(num_assays, assay_features_size)

    probs = linear.forward_dense(compound_features, assay_features)
    assert probs.shape == torch.Size([num_compounds, num_assays])


def test_mlp_batchnorm():
    mlp = models.MLPBatchNorm(
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

    probs = mlp(compound_features, assay_features)
    assert probs.shape == torch.Size([num_samples])

    # dense forward pass
    num_compounds, num_assays = 3, 5
    compound_features = torch.rand(num_compounds, compound_features_size)
    assay_features = torch.rand(num_assays, assay_features_size)

    probs = mlp.forward_dense(compound_features, assay_features)
    assert probs.shape == torch.Size([num_compounds, num_assays])


def test_mlp_layernorm():
    mlp = models.MLPLayerNorm(
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

    probs = mlp.forward(compound_features, assay_features)
    assert probs.shape == torch.Size([num_samples])

    # dense forward pass
    num_compounds, num_assays = 3, 5
    compound_features = torch.rand(num_compounds, compound_features_size)
    assay_features = torch.rand(num_assays, assay_features_size)

    probs = mlp.forward_dense(compound_features, assay_features)
    assert probs.shape == torch.Size([num_compounds, num_assays])
