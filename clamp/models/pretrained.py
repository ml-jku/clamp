
from .scaled import ScaledMLPLayerNorm
from .models import MLPLayerNorm
from .models import DotProduct

import torch
import torch.nn as nn
import numpy as np
import os
from typing import List, Tuple
from loguru import logger
from pathlib import Path
import json

class PretrainedCLAMP(MLPLayerNorm):
    """
    A CLAMP model that uses a CLIP model as the text encoder pretrained on PubChem data. db4d
    """
    # init method
    def __init__(self, path_dir='./data/models/clamp_clip/', device='cuda:0', **kwargs):
        self.path_dir = Path(path_dir)
        self.checkpoint = self.path_dir/"checkpoint.pt"

        self.load_or_download()
        self.kwargs = kwargs 
        self.compound_features_size = 8192
        self.assay_features_size = 512
        self.device = device

        self.text_encoder = None

        # override forward function of compound encoder
        self.compound_encoder.old_forward = self.compound_encoder.forward
        self.compound_encoder.forward = self.compound_forward
        self.assay_encoder.old_forward = self.assay_encoder.forward
        self.assay_encoder.forward = self.assay_forward

    def load_or_download(self, device='cpu'):
        if not os.path.exists(self.path_dir):
            url =    "https://cloud.ml.jku.at/s/7nxgpAQrTr69Rp2/download/checkpoint.pt"
            url_hp = "https://cloud.ml.jku.at/s/dRX9TWPrF7WqnHd/download/hp.json"
            # download from url via wget
            # create path if not exists
            os.makedirs(self.path_dir, exist_ok=True)
            logger.info(f"Downloading checkpoint.pt from {url} to {self.path_dir}")
            os.system(f"wget {url} -O {self.checkpoint}")
            # download hp.json
            os.system(f"wget {url_hp} -O {Path(self.path_dir)/'hp.json'}")
        # load in the hyperparameters
        hp = json.load(open(self.path_dir/'hp.json', 'r'))
        self.hparams = hp
        super().__init__(**hp) #this calls _define_encoders

        # load in the model and generate hidden
        cp = torch.load(self.checkpoint, map_location=device)#, map_location=self.device)
        self.load_state_dict(cp['model_state_dict'], strict=False)

    def load_clip_text_encoder(self):
        import clip
        from PIL import Image
        model_clip, preprocess = clip.load("ViT-B/32", device=self.device)
        #del model_clip.visual
        self.text_encoder = model_clip
    
    def compound_forward(self, x):
        """compound_encoder forward function, takes smiles or features as tensor as input"""
        if isinstance(x[0], str):
            x = self.prepro_smiles(x)
        return self.compound_encoder.old_forward(x)
    
    def assay_forward(self, x):
        """assay_encoder forward function, takes list of text str or features tensor as input"""
        if isinstance(x[0], str):
            x = self.prepro_text(x)
        return self.assay_encoder.old_forward(x)

    def prepro_smiles(self, smi):
        """preprocess smiles for compound encoder"""
        from mhnreact.molutils import convert_smiles_to_fp
        fp_size = self.compound_encoder.linear_input.weight.shape[1]
        fp_inp = convert_smiles_to_fp(smi, 
                              which=self.hparams['compound_mode'], fp_size=fp_size, njobs=1).astype(np.float32)
        compound_features = torch.tensor(fp_inp).to(self.device)
        return compound_features

    def prepro_text(self, txt):
        """preprocess text for assay encoder"""
        import clip
        if not self.text_encoder:
            self.load_clip_text_encoder()
        tokenized_text = clip.tokenize(txt, truncate=True).to(self.device) 
        assay_features = self.text_encoder.encode_text(tokenized_text).float().to(self.device)

        assay_features = assay_features.detach()
        return assay_features

    def encode_smiles(self, smis, no_grad=True):
        """encode smiles"""
        compound_features = self.prepro_smiles(smis)
        with torch.no_grad() if no_grad else torch.enable_grad():
            compound_features = self.compound_encoder(compound_features)
        return compound_features

    def encode_text(self, txt, no_grad=True):
        """encode text"""
        assay_features = self.prepro_text(txt)
        with torch.no_grad() if no_grad else torch.enable_grad():
            assay_features = self.assay_encoder(assay_features)
        return assay_features


class PretrainedFH(MLPLayerNorm):
    """
    Frequent Hitter Baseline
    """
    
    # init method
    def __init__(self, assay_features_size, compound_features_size, embedding_size, **kwargs):
        super().__init__(assay_features_size, compound_features_size, embedding_size, **kwargs)
        self.kwargs = kwargs 
        self.compound_features_size = 8192
        self.assay_features_size = 512

    def _define_encoders(
            self,
            compound_layer_sizes: List[int],
            assay_layer_sizes: List[int],
            dropout_input: float,
            dropout_hidden: float, **kwargs
    ) -> Tuple[callable, callable]:

        from biobert.utils import load_model

        self.path_dir = './mlruns/54/6f38bba6caed41ed9c93336f27310df3/'
        self.device = 'cuda:0'

        model, hparams = load_model(mlrun_path=self.path_dir, compound_features_size=8192, 
                    assay_features_size=512,
                    device=self.device, ret_hparams=True)
        model.to(self.device)
        self.hparams = hparams
        # for param in model unpack and set to this mdoel
        self.model = model

        return model.compound_encoder, model.assay_encoder

    def forward(
            self,
            compound_features: torch.Tensor,
            assay_features: torch.Tensor
    ) -> torch.Tensor:
        """
        In the end scale the preactivations.
        """

        if isinstance(compound_features[0], str):
            from mhnreact.molutils import convert_smiles_to_fp
            fp_size = self.compound_encoder.linear_input.weight.shape[1]
            compound_features = compound_features.detach().cpu().numpy().tolist()
            fp_inp = convert_smiles_to_fp(compound_features, 
                                  which=self.hparams['compound_mode'], fp_size=fp_size, njobs=1).astype(np.float32)
            compound_features = torch.tensor(fp_inp).to(self.device)

        activ = self.model.forward(compound_features, assay_features)
        return activ


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
        
    def encode_text(self, txt):
        inp, length = self.tokenize_txt(txt)
        out = self.model.forward(inp.to(self.device),  )['logits'][:,length,:][:,0,:] #vocab prediction for ith element ; last one is "choice" token which is len of batch_size here we take the first
        return out

    def forward(self, batch):
        return self.encode_text(batch) #returns vocab logits 50k

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
            assay_embeddings = self.assay_encoder.encode_text(assay_features)

            if self.norm:
                compound_embeddings = compound_embeddings / (torch.norm(compound_embeddings, dim=1, keepdim=True) +1e-13)
                assay_embeddings = assay_embeddings / (torch.norm(assay_embeddings, dim=1, keepdim=True) +1e-13)
            

            preactivations = (compound_embeddings * assay_embeddings).sum(axis=1)

        else:
            if self.kwargs.get('mode', None)=='tok_smi': #tokenize smiles
                out = model.encode_text(compound_features)
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
