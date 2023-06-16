from .models import DotProduct, MLPLayerNorm, TransformerLN, LSTMAssayEncoderLN, Linear
from clamp.models.perceptron import ReLUNetworkBatchNorm, NetworkLayerNorm, MultilayerPerceptron

from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np

try:
    import dgl
    from dgllife.model import GCNPredictor, GAT, GraphSAGE, MPNNGNN, GCN, GATPredictor, GNNOGBPredictor
    import dgllife.model as dglmodel
    from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
    from dgllife.utils import WeaveAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
except:
    pass


class GNNEncoder(nn.Module):
    """

    """
    def __init__(
        self, feature_size,  num_targets, dropout_input, dropout_hidden, 
        gnn_hidden_feats=128,
        hidden_layer_sizes=[128],
        gnn_type='gcn', #or gin
        # in_edge_feats=27, = feature_size
        n_layers=10,
        readout='mean', #'mean', 'sum', 'max'
        **args):
        """
        gnn_type: (**'gcn'**, 'gin')
        readout: (**'mean'**, 'sum', 'max')
        
        """
        super().__init__()

        asso_dim = hidden_layer_sizes[-1]

        self.gnn_enc = GAT(in_feats=27)
        
        gnn_out_feats = self.gnn_enc.hidden_feats[-1] * self.gnn_enc.num_heads[-1] if (self.gnn_enc.agg_modes[-1] == 'flatten') else self.gnn_enc.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.gnn_last = nn.Linear(gnn_out_feats*2, asso_dim)

        #self.gnn_enc = GNNOGBPredictor(in_edge_feats=feature_size, hidden_feats=gnn_hidden_feats, 
        #                                      n_tasks=num_targets, n_layers=n_layers, readout=readout, 
        #                                      gnn_type=gnn_type) #gin also possible

        
        self.W_0 = nn.Linear(hidden_layer_sizes[-1], num_targets)
        self.ln = nn.LayerNorm(normalized_shape=num_targets, elementwise_affine=False)

        self.args = args
        self.assay2description = None

    def forward_smiles(self, X, **kwargs):
        smi = [cid2smiles[xi[0]] for xi in X.cpu().detach().numpy()]
        bg = batch_smiles_to_bigraph(smi).to(X.device)
        #bg = process_map(batch_smiles_to_bigraph, smi, max_workers=20, chunksize=1, mininterval=1).to(X.device)
        
        node_feats = bg.ndata['h']#.to(args['device'])
        edge_feats = bg.edata['e']#.to(args['device'])
        node_feats = self.gnn_enc.forward(bg, node_feats)
        graph_feats = self.readout(bg, node_feats)
        return self.gnn_last(graph_feats)

    def forward(self, batch):
        """
        
        """
        node_feats = batch.ndata['h']#.long()
        edge_feats = batch.edata['e']#.long()

        #self.gnn_enc.to(batch.device)

        out = self.gnn_enc.forward(batch, node_feats) #edge_feats
        node_feats = self.gnn_enc.forward(bg, node_feats)
        graph_feats = self.readout(bg, node_feats)

        return self.ln(self.gnn_last(graph_feats))

class GNNLayerNorm(MLPLayerNorm):
    """
    Subclass of :class:`MLPLayerNorm` where compound is GNN and assay encoder is
    a multilayer perceptron (MLP) with `Layer Normalization`_.

    .. _`Layer Normalization`: https://arxiv.org/abs/1607.06450
    """
    def _define_encoders(
            self,
            compound_layer_sizes: List[int],
            assay_layer_sizes: List[int],
            dropout_input: float,
            dropout_hidden: float, **kwargs
    ) -> Tuple[callable, callable]:
        """
        Define encoders as multilayer perceptrons with layer normalization.

        Parameters
        ----------
        compound_layer_sizes: list of int
            Sizes of the hidden layers of the compound encoder.
        assay_layer_sizes: list int
            Sizes of the hidden layers of the assay encoder.
        dropout_input: float
            Dropout rate at the input layer.
        dropout_hidden: float
            Dropout rate at the hidden layers.

        Returns
        -------
        tuple of callable
        - Compound encoder
        - Assay encoder
        """

        # compound multilayer perceptron
        compound_encoder = GNNEncoder(
            feature_size=self.compound_features_size,
            hidden_layer_sizes=compound_layer_sizes,
            num_targets=self.embedding_size,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden,
            gnn_hidden_feats=kwargs.get('gnn_hidden_feats', 128),
            gnn_type=kwargs.get('gnn_type', 'gcn'),
            n_layers=kwargs.get('n_layers', 10),
            nonlinearity=kwargs.get('nonlinearity','ReLU')
        )

        # assay multilayer perceptron
        assay_encoder = NetworkLayerNorm(
            feature_size=self.assay_features_size,
            hidden_layer_sizes=assay_layer_sizes,
            num_targets=self.embedding_size,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden,
            nonlinearity=kwargs.get('nonlinearity','ReLU')
        )

        return compound_encoder, assay_encoder

