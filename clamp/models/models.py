from clamp.models.perceptron import ReLUNetworkBatchNorm, NetworkLayerNorm, MultilayerPerceptron

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

llm_cache_dir = '/system/user/publicdata/llm/'

class DotProduct(nn.Module):
    """
    Class for :class:`DotProduct` models.

    This family of models projects compound and assay feature vectors to
    embeddings of size `embedding_size`, typically by means of separate
    but similar compound and assay network encoders. Then all the pairwise
    similarities between compound and assay representations are computed with
    their dot products.

    The default :meth:`forward` method processes compound-assay interactions in
    COO-like format, while the :meth:`forward_dense` method does it in a
    matrix-factorization-like fashion.

    All subclasses of :class:`DotProduct` must implement the `_define_encoders`
    method, which has to return the compound and assay network encoders.
    """

    def __init__(
            self,
            compound_features_size: int,
            assay_features_size: int,
            embedding_size: int,
            **kwargs
    ) -> None:
        """
        Initialize class.

        Parameters
        ----------
        compound_features_size: int
            Input size of the compound encoder.
        assay_features_size: int
            Input size of the assay encoder.
        embedding_size: int
            Size of the association space.
        """
        super().__init__()

        self.compound_features_size = compound_features_size
        self.assay_features_size = assay_features_size
        self.embedding_size = embedding_size
        self.norm = kwargs.get('norm', None) # l2 norm of the output

        self.compound_encoder, self.assay_encoder = self._define_encoders(**kwargs)
        self.hps = kwargs
        self._check_encoders()

    def _define_encoders(self, **kwargs):
        """
        All subclasses of :class:`DotProduct` must implement this method, which
        has to return the compound and the assay encoders. The encoders can be
        any callables yielding the compound and assay embeddings. Typically
        though, the encoders will be two instances of :class:`torch.nn.Module`,
        whose :meth:`torch.nn.Module.forward` methods provide the embeddings.
        """
        raise NotImplementedError('"_define_encoders" must be implemented to provide the compound and assay encoders.')

    def _check_encoders(self):
        """
        Run minimal consistency checks.
        """
        assert callable(self.compound_encoder)
        assert callable(self.assay_encoder)

    def forward(
            self,
            compound_features: torch.Tensor,
            assay_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Take `compound_features` :math:`\\in \\mathbb{R}^{N \\times C}` and
        `assay_features` :math:`\\in \\mathbb{R}^{N \\times A}`, both with
        :math:`N` rows. Project both sets of features to :math:`D` dimensions,
        that is, `compound_embeddings` :math:`\\in \\mathbb{R}^{N \\times D}`
        and `assay_embeddings` :math:`\\in \\mathbb{R}^{N \\times D}`. Compute
        the row-wise dot products, thus obtaining `preactivations`
        :math:`\\in \\mathbb{R}^N`.

        Parameters
        ----------
        compound_features: :class:`torch.Tensor`, shape (N, compound_features_size)
            Array of compound features.
        assay_features: :class:`torch.Tensor`, shape (N, assay_features_size)
            Array of assay features.

        Returns
        -------
        :class:`torch.Tensor`, shape (N, )
            Row-wise dot products of the compound and assay projections.
        """
        #assert compound_features.shape[0] == assay_features.shape[0], 'Dimension mismatch'

        compound_embeddings = self.compound_encoder(compound_features)
        assay_embeddings = self.assay_encoder(assay_features)

        if self.norm:
            compound_embeddings = compound_embeddings / (torch.norm(compound_embeddings, dim=1, keepdim=True) +1e-13)
            assay_embeddings = assay_embeddings / (torch.norm(assay_embeddings, dim=1, keepdim=True) +1e-13)
        

        preactivations = (compound_embeddings * assay_embeddings).sum(axis=1)

        return preactivations

    def forward_dense(
            self,
            compound_features: torch.Tensor,
            assay_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Take `compound_features` :math:`\\in \\mathbb{R}^{N \\times C}` and
        `assay_features` :math:`\\in \\mathbb{R}^{M \\times A}`, where the
        number of rows :math:`N` and :math:`M` must not be be the same.
        Project both sets of features to :math:`D` dimensions, that is,
        `compound_embeddings` :math:`\\in \\mathbb{R}^{N \\times D}` and
        `assay_embeddings` :math:`\\in \\mathbb{R}^{M \\times D}`. Compute all
        the pairwise dot products by means of a matrix multiplication, thus
        obtaining `preactivations` :math:`\\in \\mathbb{R}^{N \\times M}`.

        Parameters
        ----------
        compound_features: :class:`torch.Tensor`, shape (N, compound_features_size)
            Array of compound features.
        assay_features: :class:`torch.Tensor`, shape (M, assay_features_size)
            Array of assay features.

        Returns
        -------
        :class:`torch.Tensor`, shape (N, M)
            All pairwise dot products of the compound and assay projections.
        """
        compound_embeddings = self.compound_encoder(compound_features)
        assay_embeddings = self.assay_encoder(assay_features)
        
        if self.norm:
            compound_embeddings = compound_embeddings / (torch.norm(compound_embeddings, dim=1, keepdim=True) +1e-13)
            assay_embeddings = assay_embeddings / (torch.norm(assay_embeddings, dim=1, keepdim=True) +1e-13)

        preactivations = compound_embeddings @ assay_embeddings.T
        return preactivations


class Dummy(DotProduct):
    """
    Dummy subclass of :class:`DotProduct` for testing purposes.
    """

    def _define_encoders(self, **kwargs):
        """
        Define random compound and assay embeddings.
        """

        def compound_encoder(x):
            return torch.rand(x.shape[0], 3)

        def assay_encoder(y):
            return torch.rand(y.shape[0], 3)

        return compound_encoder, assay_encoder


class Linear(DotProduct):
    """
    Subclass of :class:`DotProduct` where compound and assay encoders are each
    a linear layer.
    """

    def _define_encoders(self, **kwargs):
        """
        Define linear layers.
        """

        # linear layer yielding the compound embeddings
        compound_encoder = nn.Linear(
            in_features=self.compound_features_size,
            out_features=self.embedding_size
        )
        nn.init.kaiming_normal_(
            compound_encoder.weight,
            mode='fan_in',
            nonlinearity='linear'
        )
        nn.init.zeros_(compound_encoder.bias)

        # linear layer yielding the assay embeddings
        assay_encoder = nn.Linear(
            in_features=self.assay_features_size,
            out_features=self.embedding_size
        )
        nn.init.kaiming_normal_(
            assay_encoder.weight,
            mode='fan_in',
            nonlinearity='linear'
        )
        nn.init.zeros_(assay_encoder.bias)

        return compound_encoder, assay_encoder


class MLPBatchNorm(DotProduct):
    """
    Subclass of :class:`DotProduct` where compound and assay encoders are each
    a multilayer perceptron (MLP) with `Batch Normalization`_.

    .. _`Batch Normalization`: https://arxiv.org/abs/1502.03167
    """

    def _define_encoders(
            self,
            compound_layer_sizes: List[int],
            assay_layer_sizes: List[int],
            dropout_input: float,
            dropout_hidden: float, **kwargs
    ) -> Tuple[callable, callable]:
        """
        Define encoders as multilayer perceptrons with batch normalization.

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
        compound_encoder = ReLUNetworkBatchNorm(
            feature_size=self.compound_features_size,
            hidden_layer_sizes=compound_layer_sizes,
            num_targets=self.embedding_size,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden
        )

        # assay multilayer perceptron
        assay_encoder = ReLUNetworkBatchNorm(
            feature_size=self.assay_features_size,
            hidden_layer_sizes=assay_layer_sizes,
            num_targets=self.embedding_size,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden
        )

        return compound_encoder, assay_encoder


class MLPLayerNorm(DotProduct):
    """
    Subclass of :class:`DotProduct` where compound and assay encoders are each
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
        compound_encoder = NetworkLayerNorm(
            feature_size=self.compound_features_size,
            hidden_layer_sizes=compound_layer_sizes,
            num_targets=self.embedding_size,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden,
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

class Global(DotProduct):
    """
    Global Model that doesn't use the assay-encoder

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
        compound_encoder = MultilayerPerceptron(
            feature_size=self.compound_features_size,
            hidden_layer_sizes=compound_layer_sizes,
            num_targets=1,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden,
            nonlinearity=kwargs.get('nonlinearity','ReLU')
        )
        return compound_encoder, lambda x: x
    
    def forward(
            self,
            compound_features: torch.Tensor,
            assay_features = None,
    ) -> torch.Tensor:
        """
        Take `compound_features` :math:`\\in \\mathbb{R}^{N \\times C}`.
        Projects sets of features to :math:`D` dimensions, to 1

        Parameters
        ----------
        compound_features: :class:`torch.Tensor`, shape (N, compound_features_size)
            Array of compound features.

        Returns
        -------
        :class:`torch.Tensor`, shape (N, )
        """

        compound_embeddings = self.compound_encoder(compound_features)
        preactivations = (compound_embeddings ).view(-1)
        return preactivations


class MLPLayerNormConcat(MLPLayerNorm):
    "Instead of making a dot-product we concatenate and continue with ff-layers (with the same design as the encoders)"
    def __init__(
            self,
            compound_features_size: int,
            assay_features_size: int,
            embedding_size: int,
            **kwargs
    ) -> None:
        """
        Initialize class.

        Parameters
        ----------

        """
        super().__init__(compound_features_size, assay_features_size, embedding_size, **kwargs)
        self.ffnn = NetworkLayerNorm(
            feature_size=self.embedding_size*2,
            hidden_layer_sizes=kwargs['hidden_layers'],
            num_targets = 1,
            dropout_input=kwargs[('dropout_input')],
            dropout_hidden=kwargs[('dropout_hidden')]
        )

    def forward(
            self,
            compound_features: torch.Tensor,
            assay_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Take `compound_features` :math:`\\in \\mathbb{R}^{N \\times C}` and
        `assay_features` :math:`\\in \\mathbb{R}^{N \\times A}`, both with
        :math:`N` rows. Project both sets of features to :math:`D` dimensions,
        that is, `compound_embeddings` :math:`\\in \\mathbb{R}^{N \\times D}`
        and `assay_embeddings` :math:`\\in \\mathbb{R}^{N \\times D}`. Compute
        the FFNN of the embedded features, returning
        :math:`\\in \\mathbb{R}^N`.

        Parameters
        ----------
        compound_features: :class:`torch.Tensor`, shape (N, compound_features_size)
            Array of compound features.
        assay_features: :class:`torch.Tensor`, shape (N, assay_features_size)
            Array of assay features.

        Returns
        -------
        :class:`torch.Tensor`, shape (N, )
            FFNN of the compound and assay projections.
        """
        assert compound_features.shape[0] == assay_features.shape[0], 'Dimension mismatch'

        compound_embeddings = self.compound_encoder(compound_features)
        assay_embeddings = self.assay_encoder(assay_features)

        concat_embeddings =  torch.cat((compound_embeddings, assay_embeddings), dim=1)

        activations = self.ffnn(concat_embeddings ).view(-1)

        return activations

class LSTMAssayEncoderLN(DotProduct):
    """
    LSTM Assay Enoder with Layer Norm (LN) at the end
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
        compound_encoder = NetworkLayerNorm(
            feature_size=self.compound_features_size,
            hidden_layer_sizes=compound_layer_sizes,
            num_targets=self.embedding_size,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden
        )

        # assay multilayer perceptron
        assay_encoder = LSTMEncoder(
            feature_size=self.assay_features_size,
            hidden_layer_sizes=assay_layer_sizes,
            num_targets=self.embedding_size,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden,
            **kwargs
        )

        return compound_encoder, assay_encoder

class LSTMEncoder(nn.Module):
    """
    LSTM Encoder with params:
        same as clamp.models.perceptron feature_size, hidden_layer_sizes, num_targets, dropout_input, dropout_hidden
    additionally:
        embedding_dim: default=512
        hidden_dim: of LSTM is set to hidden_layer_sizes[1] by default
        num_layers: default=3 number of LSTM layers
        vocab_size: of tokenizer, depends on assay_mode file provided with tokenized and zero-padded array
        bidirectional: parameter of LSTM, default=True
    """
    def __init__(self, feature_size, hidden_layer_sizes, num_targets, dropout_input, dropout_hidden, embedding_dim=512, num_layers=3, vocab_size=58996, bidirectional=True, **args):
        super().__init__()
        hidden_dim = hidden_layer_sizes[1]
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_hidden, bidirectional=bidirectional) 

        self.W_0 = nn.Linear(hidden_dim, num_targets)
        
        self.ln = nn.LayerNorm(normalized_shape=num_targets, elementwise_affine=False)
        torch.nn.init.kaiming_normal_(self.W_0.weight, mode='fan_in', nonlinearity='linear')


    def forward(self, batch, lengths=None):
        if lengths is None:
            # get first zero index of column = length of the batch #returns last 0 index so all zeros is fine
            lengths = torch.argmin((batch==0)*torch.arange(batch.shape[1], device=batch.device), axis=1) + 1
            lengths = lengths.cpu().detach().numpy().tolist()
            
        embeds = self.embedding(batch.long()) # batch x seq_len x embedding
        #embeds = embeds.transpose(1,0) #seq. len x batch x _
        packed_input = pack_padded_sequence(embeds, lengths, enforce_sorted=False, batch_first=True)
        out, (ht, ct) = self.lstm(packed_input)
        return self.ln(self.W_0(ht[-1])) #last hidden state projected out

class MFBERTEncoder(nn.Module):
    def __init__(self, feature_size, hidden_layer_sizes, num_targets, dropout_input, 
            dropout_hidden, embedding_dim=512, num_layers=3, **args):
        super().__init__()
        sys.path.append('/publicwork/seidl/projects/MFBERT/')
        from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
        from Model.model import MFBERT
        self.tokenizer = MFBERTTokenizer.from_pretrained(TOKENIZER_DIR+'Model/',
                                            dict_file = TOKENIZER_DIR+'Model/dict.txt')
        self.transformer = MFBERT()

        self.W_0 = nn.Linear(hidden_dim, num_targets)
        
        self.ln = nn.LayerNorm(normalized_shape=num_targets, elementwise_affine=False)
        torch.nn.init.kaiming_normal_(self.W_0.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, batch):
        """
        if batch is batch_size x 1, than it is assumed it has to be tokenized, and done so (provide either a string, or )
        """
        if len(batch.shape)==1:
            tokens = self.tokenizer.tokenize(batch) 
        else:
            tokens = batch #TODO convert to dict, that is readable by transformer

        out = self.transformer(**tokens)
        return self.ln(self.W_0(out))


class TransformerEncoder(nn.Module):
    """
    
    """
    def __init__(
        self, feature_size, hidden_layer_sizes, num_targets, 
        dropout_input, dropout_hidden, 
        tokenizer='dmis-lab/biobert-v1.1',#'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',#'dmis-lab/biobert-large-cased-v1.1', 
        transformer='dmis-lab/biobert-v1.1',#'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',#'dmis-lab/biobert-large-cased-v1.1',
        augment_prob=0.0,
        pooling_mode='max', 
        bitfit=False,
        **args):
        """
        pooling_modes: 'last', 'first' (=CLS token), 'mean', 'max', 'min', or another torch.* attribute (default: 'last')
        """
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = 'cuda:1' #TODO: make device configurable

        if tokenizer == 'clip':
            assert tokenizer == transformer, 'tokenizer and transformer must be the same for clip'
            import clip

            model, preprocess = clip.load("RN50",  device='cpu') 
            #del model.visual
            model.transformer.to(self.device)
            def bep_wrapper(text, return_tensors=False,padding=True,truncation=True,max_length=77):
                return clip.tokenize(text, truncate=True)#.to(self.device)
            
            model.batch_encode_plus = bep_wrapper
            self.tokenizer = model
            self.transformer = model
        elif tokenizer=='regex':
            import re
            SMARTSREGEX = r"""(\[|\]|Br?|Cl?|Au?|Fe?|Zn?|Mg?|Li?|Ga?|As?|Ru?|Eu?|Ta?|Ga?|Yb?|Dy?|N|O|S|P|F|I|H|K|U|W|V|Y|b|c|n|o|s|i|p|D\d+|[a-z]|\(|\)|\.|=|#|-|\+|\;|\%|\\|\/|:|~|@|\?|>>?|\*|\$|\d+|)"""
            class MySmartsTokenizer(object):
                """adapted from https://github.com/deepchem/deepchem/blob/2.4.0/deepchem/feat/smiles_tokenizer.py#L285-L323
                """
                def __init__(self, regex_pattern: str = SMARTSREGEX):
                    self.regex_pattern = SMARTSREGEX
                    self.regex = re.compile(self.regex_pattern)

                def tokenize(self, text):
                    tokens = [token for token in self.regex.findall(text)]
                    return tokens
            self.tokenizer = MySmartsTokenizer()
            self.transformer = AutoModelForSequenceClassification.from_pretrained(transformer, output_hidden_states=True)
            self.transformer.config.hidden_dropout_prob = dropout_hidden
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.transformer = AutoModelForSequenceClassification.from_pretrained(
                transformer, output_hidden_states=True
                )

            self.transformer.config.hidden_dropout_prob = dropout_hidden
        self.args = args
        

        #(Zaken et. al.)[2106.10199] BitFit: Simple Parameter-efficient Fine-tuning for ...
        if bitfit: 
            for name, param in self.transformer.named_parameters():
                if 'bias' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.W_0 = nn.Linear(512 if tokenizer=='clip' else self.transformer.config.hidden_size, num_targets)
        self.ln = nn.LayerNorm(normalized_shape=num_targets, elementwise_affine=False)

        self.augment_prob = augment_prob
        self.pooling_mode = pooling_mode
        self.args = args
        self.assay2description = None

    def load_assay2description(self, clms_to_use=list(('assay_cell_type','assay_organism','assay_strain','assay_subcellular_fraction','assay_test_type','assay_tissue', 
               'assay_strain','assay_test_type', 'confidence_description', 'relationship_description', 'description', 'target_chembl_id'))):
        from pathlib import Path
        import pandas as pd
        root = Path(self.args.get('dataset'))
        with open(root.joinpath('assay_names.parquet'), 'rb') as f:
            df = pd.read_parquet(f)

        adi = df[(clms_to_use)].values.tolist()
        bert_corpus = []
        for row in adi:
            ad = ''
            for i, clm in enumerate(clms_to_use):
                if row[i] != None:
                    ad = (ad+' '+clm+': '+row[i])
            bert_corpus.append(ad)

        self.assay2description = {i:bert_corpus[i] for i in range(len(bert_corpus))}

    def augment(self, txt):
        raise NotImplementedError()

    def tokenize(self, idx_or_str_list, return_tensors='pt', max_length=512):
        """
        tokenize batch/list of strings or list of ints with index to corresponding assay-description
        return_tensor is default 'pt' but can also be 'np' for numpy
        max_length is by default 512
        """
        if self.assay2description is None:
            self.load_assay2description()

        text = []
        
        for i in idx_or_str_list:
            ti = self.assay2description[i] if isinstance(i, int) else i

            # Further augmentation might be done here
            if self.augment_prob>np.random.rand():
                ti = self.augment(ti)
            text.append(ti)

        tokens = self.tokenizer.batch_encode_plus(
            text,
            return_tensors=return_tensors,
            padding=True,
            truncation=True,
            max_length=max_length,
            # padding only applies to batches of several data points
        )

        if isinstance(tokens, torch.Tensor):
            return tokens

        return {k:v.to(self.transformer.device) for k,v in tokens.items()}

    def forward(self, batch):
        """
        if batch is batch_size x 1, than it is assumed it has to be tokenized, and done so (provide either a string, or )
        """
        if len(batch.shape)==1:
            tokens = self.tokenize(batch).to(self.device) #TODO: make device configurable
        else:
            tokens = batch #TODO convert to dict, that is readable by transformer

        if hasattr(self.transformer, 'encode_text'): # it's a CLIP model
            out = self.transformer.encode_text(tokens).float()
            #return out
        else:
            features_pt = self.transformer(**tokens, return_dict=True) #returns logits and hidden_states
        
            #features_pt['hidden_states'] is layers x ( [batch_size, tokens, hidden_dim] )
            if self.pooling_mode=='last':
                out = features_pt['hidden_states'][-1][:,-1,:] #last layer last token
            elif self.pooling_mode=='first':
                out = features_pt['hidden_states'][-1][:,0,:] #last layer first token
            elif self.pooling_mode=='max':
                out = features_pt['hidden_states'][-1].max(dim=1)[0] #last layer max token
            else:
                out = getattr(torch, self.pooling_mode)(features_pt['hidden_states'][-1], axis=1)
        
        return self.ln(self.W_0(out))


class TransformerLN(DotProduct):
    """
    Transformer Assay Enoder with Layer Norm (LN) at the end
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
        """

        # compound multilayer perceptron
        compound_encoder = NetworkLayerNorm(
            feature_size=self.compound_features_size,
            hidden_layer_sizes=compound_layer_sizes,
            num_targets=self.embedding_size,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden
        )

        # assay multilayer perceptron
        assay_encoder = TransformerEncoder(
            feature_size=self.assay_features_size, #doesn't do anything here ;)
            hidden_layer_sizes=assay_layer_sizes,
            num_targets=self.embedding_size,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden,
            **kwargs,
        )

        return compound_encoder, assay_encoder


class BothTransformerLN(DotProduct):
    """
    Transformer Assay and Molecule Enoder with Layer Norm (LN) at the end
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
        """

        # TODO input has to be tokenized smiles

        # compound multilayer perceptron
        compound_encoder = TransformerEncoder(
            feature_size=self.compound_features_size,
            hidden_layer_sizes=compound_layer_sizes,
            num_targets=self.embedding_size,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden,
            tokenizer="seyonec/ChemBERTa-zinc-base-v1",
            transformer="seyonec/ChemBERTa-zinc-base-v1"
        )

        # assay multilayer perceptron
        assay_encoder = TransformerEncoder(
            feature_size=self.assay_features_size, #doesn't do anything here ;)
            hidden_layer_sizes=assay_layer_sizes,
            num_targets=self.embedding_size,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden,
            **kwargs,
        )

        return compound_encoder, assay_encoder




class KVPLMEncoder(nn.Module):
    def __init__(
        self, feature_size, hidden_layer_sizes, num_targets, 
        dropout_input, dropout_hidden, 
        tokenizer='allenai/scibert_scivocab_uncased',
        transformer='allenai/scibert_scivocab_uncased',
        device='cuda:3',
        bitfit=True,
        **args):
        super().__init__()
        from transformers import BertTokenizer, BertForPreTraining
        import sys
        import torch
        import torch.nn as nn
        import numpy as np
        self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_folder=llm_cache_dir)

        class BigModel(nn.Module):
            def __init__(self, main_model):
                super(BigModel, self).__init__()
                self.main_model = main_model
                self.dropout = nn.Dropout(dropout_hidden)

            def get_device(self):
                return next(self.main_model.parameters()).device

            def forward(self, tok, att):
                # get device of model
                typ = torch.zeros(tok.shape, device=self.get_device()).long()
                pooled_output = self.main_model(tok, token_type_ids=typ, attention_mask=att)['pooler_output']
                logits = self.dropout(pooled_output)
                return logits

        bert_model0 = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = BigModel(bert_model0.bert)
        self.device = device
        self.model.to(device)
        self.model.load_state_dict(torch.load('./model_zoo/kvplm/KV-PLM/save_model/ckpt_ret01.pt', map_location=torch.device(device) ))
        if bitfit:
            for name, param in self.model.named_parameters():
                if 'bias' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False


    def forward(self, batch):
        inp_SM = self.tokenizer.batch_encode_plus(batch, max_length=128, padding=True, truncation=True, return_tensors='pt')
        logits_smi = self.model(inp_SM['input_ids'].to(self.device), inp_SM['attention_mask'].to(self.device))
        return logits_smi


class KVPLM(DotProduct):
    """
    KV-PLM Assay and Molecule Enoder shared weights; as well as l2-normalization afterwards
    """
    def _define_encoders(
            self,
            compound_layer_sizes: List[int],
            assay_layer_sizes: List[int],
            dropout_input: float,
            dropout_hidden: float, **kwargs
    ) -> Tuple[callable, callable]:
        self.norm = True
        # assay multilayer perceptron
        assay_encoder = KVPLMEncoder(
            feature_size=self.assay_features_size, #doesn't do anything here ;)
            hidden_layer_sizes=assay_layer_sizes,
            num_targets=self.embedding_size,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden,
            **kwargs,
        )

        return assay_encoder, assay_encoder

class KVPLMsep(DotProduct):
    """
    2 seperate encoders for compound and assay
    """
    def _define_encoders(
            self,
            compound_layer_sizes: List[int],
            assay_layer_sizes: List[int],
            dropout_input: float,
            dropout_hidden: float, **kwargs
    ) -> Tuple[callable, callable]:

        # activate normalization after encoding
        self.norm = True

        # assay multilayer perceptron
        compound_encoder = KVPLMEncoder(
            feature_size=self.assay_features_size, #doesn't do anything here ;)
            hidden_layer_sizes=assay_layer_sizes,
            num_targets=self.embedding_size,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden,
            **kwargs,
        )
        assay_encoder = KVPLMEncoder(
            feature_size=self.assay_features_size, #doesn't do anything here ;)
            hidden_layer_sizes=assay_layer_sizes,
            num_targets=self.embedding_size,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden,
            **kwargs,
        )
        return compound_encoder, assay_encoder

class GalacticaEncoder(nn.Module):
    """
    Galactica encoder
    """
    def __init__(
        self, feature_size, hidden_layer_sizes, num_targets, 
        dropout_input, dropout_hidden, 
        tokenizer="facebook/galactica-125m",
        transformer="facebook/galactica-125m",
        device='cuda:3',
        fp16=True,
        bitfit=True, **args
    ):
        super().__init__()
        from transformers import AutoTokenizer, OPTForCausalLM
        if tokenizer=='':
            tokenizer = "facebook/galactica-125m"
        if transformer=='':
            transformer = "facebook/galactica-125m"

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, cache_folder=llm_cache_dir)
        self.tokenizer.pad_token = 0
        self.model = OPTForCausalLM.from_pretrained(transformer, cache_dir=llm_cache_dir, torch_dtype=torch.float16 if fp16 else torch.float32)
        self.model.to(device)
        self.device = device
        self.model.eval()

        if bitfit:
            for name, param in self.model.named_parameters():
                if 'bias' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def tokenize_smiles(self, smis, prefix='Here is a SMILES formula: [START_SMILES]', postfix='[END_SMILES]'):
        """tokenizes a text with smiles in it, adjust the pre and postfix to your liking also accepts lists of pre/postfixes"""
        # second option is way faster; we need to tokenize it by character
        #inp = torch.tensor([[tokenizer.vocab.get(i) for i in ['[START_SMILES]',*smis,'[END_SMILES]']]]).to(device)
        #prepost_len = len(self.tokenizer(prefix+postfix).input_ids)
        #len_idx = list(map(lambda k: len(k)+prepost_len-1, smis))
        #longest_smi = max(len_idx)+1
        #inps = np.zeros((len(smis), longest_smi+2))

        inps = []
        max_len = 0
        for ii, smi in enumerate(smis): # I think tokenization takes longer than going through it twice
            prefix_i = prefix[ii] if isinstance(prefix, list) else prefix
            postfix_i = postfix[ii] if isinstance(postfix, list) else postfix

            out = np.hstack( self.tokenizer([prefix_i,*smi,postfix_i]).input_ids )
            #inps[ii,:len(out)] = (out)
            #len_idx[ii] = len(out)
            inps.append(out)
            max_len = max(max_len, len(out))
        
        inps_np = np.zeros((len(smis), max_len))
        len_idx = np.zeros(len(smis))
        for ii, inp in enumerate(inps):
            len_idx[ii] = len(inp)-1
            inps_np[ii, :len(inp)] = inp

        return torch.tensor(inps_np, dtype=torch.long), len_idx

    def encode_smiles(self, smis):
        inps, lin_idx = self.tokenize_smiles(smis)

        out = self.model.forward(inps.to(self.device))['logits'][:,len_idx,:]
        return out
    
    def tokenize_txt(self, txt):
        if isinstance(txt, str):
            txt = [txt]
        if isinstance(txt, np.ndarray):
            txt = txt.tolist()
        inp = self.tokenizer(txt, padding=True, return_length=True)
        length = np.array(inp.length)-1
        inp = torch.tensor(inp.input_ids)
        return inp, length
        
    def encode_txt(self, txt):
        inp, length = self.tokenize_txt(txt)
        out = self.model.forward(inp.to(self.device),  )['logits'][:,length,:][:,0,:] #vocab prediction for ith element ; last one is "choice" token which is len of batch_size here we take the first
        return out

    def forward(self, batch):
        return self.encode_txt(batch) #returns vocab logits 50k

class Galactica(DotProduct):
    """
    Galactica encodes assay and compound in one encoder
    """
    
    # init method
    def __init__(self, assay_features_size, compound_features_size, embedding_size, **kwargs):
        super().__init__(assay_features_size, compound_features_size, embedding_size, **kwargs)
        self.kwargs = kwargs 

        self.pos_yes = None

    def _define_encoders(
            self,
            compound_layer_sizes: List[int],
            assay_layer_sizes: List[int],
            dropout_input: float,
            dropout_hidden: float, **kwargs
    ) -> Tuple[callable, callable]:
        self.norm = True
        # assay multilayer perceptron
        assay_encoder = GalacticaEncoder(
            feature_size=self.assay_features_size, 
            hidden_layer_sizes=assay_layer_sizes,
            num_targets=self.embedding_size,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden,
            **kwargs,
        )

        return assay_encoder, assay_encoder

    def forward(
            self,
            compound_features: torch.Tensor,
            assay_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Take `compound_features` :math:`\\in \\mathbb{R}^{N \\times C}` and
        `assay_features` :math:`\\in \\mathbb{R}^{N \\times A}`, both with
        :math:`N` rows. Project both sets of features to :math:`D` dimensions,
        that is, `compound_embeddings` :math:`\\in \\mathbb{R}^{N \\times D}`
        and `assay_embeddings` :math:`\\in \\mathbb{R}^{N \\times D}`. Compute
        the row-wise dot products, thus obtaining `preactivations`
        :math:`\\in \\mathbb{R}^N`.

        Parameters
        ----------
        compound_features: :class:`torch.Tensor`, shape (N, compound_features_size)
            Array of compound features.
        assay_features: :class:`torch.Tensor`, shape (N, assay_features_size)
            Array of assay features.

        Returns
        -------
        :class:`torch.Tensor`, shape (N, )
            Row-wise dot products of the compound and assay projections.
        """
        #assert compound_features.shape[0] == assay_features.shape[0], 'Dimension mismatch'

        model = self.compound_encoder

        if self.pos_yes is None:
            #self.pos_yes = model.tokenizer(' Yes').input_ids #make this configurable TODO
            self.pos_yes = model.tokenizer('Yes').input_ids #make this configurable TODO

        use_dot_prod_sim = False #TODO maybe implement ;)

        if use_dot_prod_sim:
            # if cosine sim between the two embeddings
            compound_embeddings = self.compound_encoder.encode_smiles(compound_features)
            assay_embeddings = self.assay_encoder.encode_txt(assay_features)

            if self.norm:
                compound_embeddings = compound_embeddings / (torch.norm(compound_embeddings, dim=1, keepdim=True) +1e-13)
                assay_embeddings = assay_embeddings / (torch.norm(assay_embeddings, dim=1, keepdim=True) +1e-13)
            

            preactivations = (compound_embeddings * assay_embeddings).sum(axis=1)

        else:
            if self.kwargs.get('mode', None)=='tok_smi': #tokenize smiles
                out = model.encode_txt(compound_features)
                preactivations = out[:,pos_yes].mean(1)
            else:
                smiles = compound_features[0]
                f"""Here is a SMILES formula: 

                [START_I_SMILES]{smiles}[END_I_SMILES] 

                Question: Will the chemical compound penetrate the blood-brain barrier? 

                Answer:"""

                assay_features = list(assay_features)
                prefix = 'Here is a SMILES formula:\n\n[START_SMILES]'
                # SMILES COMES HERE
                postfix= ['[END_SMILES] \n\nQuestion: Will the chemical compound be active in the assay: '+af+' \n\nAnswer (Yes or No): ' for af in assay_features]
                
                both, lens = model.tokenize_smiles(compound_features, prefix=prefix, postfix=postfix)
                
                out = model.model.forward(both.to(model.device))['logits']

                # take logit of pos_yes
                preactivations = out[:,lens,:][:,-1,self.pos_yes].mean(1)                

        return preactivations
