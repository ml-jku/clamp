from pathlib import Path
from loguru import logger

import numpy as np
import pandas as pd
import argparse
import joblib
import tqdm
import os

"""
Extract assay features like LSA. A list of PubChem assay ids
is provided, and a Numpy dense matrix is created such that its i-th row 
contains the features of the i-th assay in the initial list.

Example call:
```python clamp/dataset/encode_assay.py --assay_path=./data/pubchem23/assay_names.parquet --encoding=clip --gpu=0 --columns title```
Or with subtitle:
```python clamp/dataset/encode_assay.py --assay_path=./data/pubchem23/assay_names.parquet --encoding=clip --gpu=0 --columns title subtitle --suffix=all```

For FS-Mol we use the following columns:
python clamp/dataset/encode_assay.py --assay_path=./data/fsmol/assay_names.parquet --encoding=clip --gpu=0 --columns \
assay_type_description description assay_category assay_cell_type assay_chembl_id assay_classifications assay_organism assay_parameters assay_strain assay_subcellular_fraction assay_tax_id assay_test_type assay_tissue assay_type bao_format bao_label cell_chembl_id confidence_description confidence_score document_chembl_id relationship_description relationship_type src_assay_id src_id target_chembl_id tissue_chembl_id variant_sequence \
--suffix=all

"""

def clip_encode(list_of_assay_descriptions, gpu=0, batch_size=2048, truncate=True, verbose=True):
    """Encode a list of assay descriptions using a fitted Model.
    supposed to be called once
    Parameters
    ----------
    list_of_assay_descriptions : list of str
        List of assay descriptions.
    gpu : int
        Device to use for the CLIP model.   
    batch_size : int
        Batch size to use for the CLIP model.
    truncate : bool 
        deault True
        Whether to truncate the assay descriptions to 77 tokens.
    Returns
    -------
    numpy.ndarray
        Numpy dense matrix with shape (n_assays, n_components).
    """
    import torch
    import clip
    from PIL import Image
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Load CLIP model on {device}.')
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    logger.info('Encode assay descriptions using CLIP.')
    with torch.no_grad():
        text_features = []
        for b in tqdm.tqdm(range(0,len(list_of_assay_descriptions), batch_size), desc='Encode assay descriptions', disable=not verbose):
            tokenized_text = clip.tokenize(list_of_assay_descriptions[b:min(b+batch_size,len(list_of_assay_descriptions))], truncate=truncate).to(device) 
            tf = model.encode_text(tokenized_text)
            text_features.append(tf.cpu().detach().numpy())
    text_features = np.concatenate(text_features, axis=0)
    return text_features.astype(np.float32)



def lsa_encode(list_of_assay_descriptions, lsa_path='', verbose=True):
    """Encode a list of assay descriptions using a fitted Model.

    Parameters
    ----------
    list_of_assay_descriptions : list of str
        List of assay descriptions.
    lsa_path : str
        Path to a fitted, joblib-ed sklearn model.
    n_components : int
        Number of components to use for the TruncatedSVD model.

    Returns
    -------
    numpy.ndarray
        Numpy dense matrix with shape (n_assays, n_components).
    """
    if verbose:
        logger.info('Load a fitted sklearn model.')
    model = joblib.load(lsa_path)

    if verbose:
        logger.info('Encode assay descriptions.')
    features = model.transform(list_of_assay_descriptions)

    return features

def lsa_fit(list_of_assay_descriptions, model_save_path='./data/models/lsa.joblib', n_components=355, verbose=True):
    """Fit a sklearn TruncatedSVD model using a list of assay descriptions.

    Parameters
    ----------
    list_of_assay_descriptions : list of str
        List of assay descriptions.
    model_save_path : str
        Path to where the fitted model will be saved.
    n_components : int
        Number of components to use for the TruncatedSVD model.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline
    logger.info('Set up and fit-transform a sklearn TfidfVectorizer.')
    tok = Tokenizer()
    tfidf = TfidfVectorizer(
        strip_accents='unicode',
        analyzer='word',
        #tokenizer=tok, # TODO use this
        stop_words='english',
        max_df=0.95,
        min_df=1/10000,
        dtype=np.float32
    )

    features = tfidf.fit_transform(list_of_assay_descriptions)
    logger.info(f'tfidf vocabulary size: {len(tfidf.vocabulary_)}')
    if verbose:
        logger.info(f'Fit a sklearn TruncatedSVD model with {n_components} n_components.')
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(features)

    model = Pipeline([('tfidf', tfidf), ('svd', svd)])

    if verbose:
        logger.info('Save the fitted model.')
    joblib.dump(model, model_save_path)

    return model



class Tokenizer:
    """
    Custom tokenizer combining ideas from `sklearn documentation`_ and from
    `this post`_. Requires the `nltk` package. Using part of speech (POS) tags
    makes it quite slow, but results seem cleaner.

    .. _`sklearn documentation`: https://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
    .. _`this post`: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    """

    def __init__(self):
        import nltk
        self.nltk = nltk
        # check if resources are available
        try:
            nltk.word_tokenize('Hello world.')
        except LookupError:
            logger.info('Download nltk data.')
            nltk.download('punkt')

        try:
            from nltk.corpus import wordnet
            self.wordnet = wordnet
            wordnet.ADJ
            # check punkt
            self.nltk.word_tokenize('Hello world.')
        except LookupError:
            logger.info('Download wordnet corpus.')
            nltk.download('wordnet')
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            from nltk.corpus import wordnet
            self.wordnet = wordnet

        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.tag2pos = {
            'J': self.wordnet.ADJ,
            'N': self.wordnet.NOUN,
            'V': self.wordnet.VERB,
            'R': self.wordnet.ADV
        }

    def _get_wordnet_pos(self, word):
        """
        Map part of speech (POS) tag to first character lemmatize() accepts.
        If POS tag does not exist in `tag2pos`, return `wordnet.NOUN`.
        """
        tag = self.nltk.pos_tag([word])[0][1][0].upper()
        return self.tag2pos.get(tag, self.wordnet.NOUN)

    def __call__(self, doc):
        for word in self.nltk.word_tokenize(doc):
            pos = self._get_wordnet_pos(word)
            yield self.lemmatizer.lemmatize(word, pos)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Compute features for a collection of PubChem assay descriptions.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--assay_path', default='./data/pubchem23/assay_names.parquet', help='Path to a parquet file with assay index to AID for which to extract features.')
    parser.add_argument('-c','--columns', nargs='+', help='Columns to use for the assay descriptions. default: title and subtitle', default=['title', 'subtitle'])
    # example call for columns --columns=
    parser.add_argument('--suffix', help='Suffix to add to the output file. default: None', default=None)
   # encodngs
    parser.add_argument('--encoding', help='Encoding-type to use for the assay descriptions. Available are text, clamp, lsa, default: lsa', default='lsa')
    parser.add_argument('--lsa_path', help='Path to a fitted, joblib-ed sklearn TfidfVectorizer+LSA model, or where to save it if not present', default='./data/models/lsa.joblib')
    parser.add_argument('--train_set_size', help='Number of assay descriptions to use for training the model. default: first 80%', default=0.8, type=float)
    parser.add_argument('--gpu', help='GPU number to use for a GPU-based encoding, if any. default: 0', default=0)
    parser.add_argument('--batch_size', help='Batch size to use for a GPU-based encoding. default: 2048', default=2048, type=int)
    parser.add_argument('--n_components', help='Number of components to use for the TruncatedSVD model. default: 355', default=355, type=int)
    args = parser.parse_args()

    df = pd.read_parquet(args.assay_path)

    path = Path(args.assay_path)

    # check if all columns are present
    if not all([c in df.columns for c in args.columns]):
        raise ValueError(f'Columns {args.columns} not found in the assay dataframe. Available columns: {df.columns}')
    
    df[args.columns] = df[args.columns].fillna('')
    df[args.columns] = df[args.columns].astype(str)

    list_of_assay_descriptions = df[args.columns].apply(lambda x: ' '.join(x), axis=1).tolist()

    logger.info(f'example assay description: {list_of_assay_descriptions[0]}')

    if args.encoding == 'text':
        features = np.array(list_of_assay_descriptions)

    elif args.encoding == 'lsa':
        logger.info('Encode assay descriptions using LSA.')
        # load model if the file exists
        if not Path(args.lsa_path).is_file():
            logger.info('Fit a sklearn TfidfVectorizer model on training data.')
            #lsa_save_path = path.with_name(f'assay_lsa_enc{"_"+args.suffix if args.suffix else ""}.joblib')
            logger.info(f'Save the fitted LSA-model to {args.lsa_path}, load it later using the argument --lsa_path')

            # TODO custom fit depending on training-set size
            train_set_size = args.train_set_size
            if train_set_size < 1:
                train_set_size = int(len(list_of_assay_descriptions)*train_set_size)

            logger.info(f'Fit on {train_set_size} train assay descriptions, {train_set_size/len(list_of_assay_descriptions)*100:2.2f}% of the total dataset.')
            model = lsa_fit(list_of_assay_descriptions[:int(train_set_size)], model_save_path=args.lsa_path)

        features = lsa_encode(list_of_assay_descriptions, args.lsa_path)

    elif args.encoding == 'clip':
        features = clip_encode(list_of_assay_descriptions, gpu=args.gpu, verbose=True, batch_size=args.batch_size)
    elif args.encoding == 'biobert':
        raise NotImplementedError('BioBERT encoding is not yet implemented.')
    else:
        raise ValueError(f'Encoding {args.encoding} not implemented.')
    

    fn = path.with_name(f'assay_features_{args.encoding}{"_"+args.suffix if args.suffix else ""}.npy')
    np.save(fn, features)
    logger.info(f'Save assay features to {fn}')