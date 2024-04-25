from pathlib import Path
from loguru import logger

import numpy as np
import pandas as pd
import argparse
import joblib
import tqdm
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

"""
Text augmentaiton for assay names. We use the following augmentations:

Example calls:
```python clamp/dataset/assay_augment.py --assay_path=./data/fsmol/assay_names.parquet --augmentation_type=transformers_paraphrase --columns description --gpu=4```
```python -m clamp.dataset.assay_augment --assay_path=./data/pubchem18/assay_names.parquet --augmentation_type=openchat --columns title subtitle```
```python -m clamp.dataset.assay_augment --assay_path=./data/pubchem18/assay_names.parquet --augmentation_type=openchat --columns title subtitle --num_augmentations=3```

python -m clamp.dataset.assay_augment --assay_path=./data/fsmol/assay_names.parquet --augmentation_type=openchat --columns description assay_type_description assay_category assay_cell_type assay_chembl_id assay_classifications assay_organism assay_parameters assay_strain assay_subcellular_fraction assay_tax_id assay_test_type assay_tissue assay_type --num_augmentations=3

"""

def openchat_paraphrase(list_of_assay_descriptions, url="http://localhost:18888/v1/chat/completions", verbose=True, num_augmentations=5):
    """
    before run the following command:
    ```conda activate openchat; python -m ochat.serving.openai_api_server --model-type openchat_v3.2 --model openchat/openchat_v3.2 --max-num-batched-tokens 5120  --tensor-parallel-size 2```
    """
    import requests
    headers = {"Content-Type": "application/json"}

    def get_response(text):
        data = {
            "model": "openchat_v3.2",
            "messages": [
                {"role": "user", "content": text}
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        if "choices" in response.json().keys():
            return response.json()["choices"][0]["message"]["content"]
        else:
            # try once more
            response = requests.post(url, headers=headers, json=data)
            if "choices" in response.json().keys():
                return response.json()["choices"][0]["message"]["content"]
            else:
                print("ERR: ",response.json())
                return ""
        return ""


    paraphrases = []
    for i, assay_description in tqdm.tqdm(enumerate(list_of_assay_descriptions), total=len(list_of_assay_descriptions)):
        content = f"""Reformulate and summarize the following assay description in 100 words or less:
        Be as concise as possible. The summary should contain relevant information to compair over assay similarity.
        ---
        {assay_description}
        ---
        """
        res = [get_response(content) for _ in range(num_augmentations)]
        paraphrases.append(res)

        if verbose and i % 100 == 0:
            logger.info(f'batch {i}, assay_description: {assay_description} paraphrases: {res}')
    
    return paraphrases

def transformers_paraphrase(list_of_assay_descriptions, model_name="humarin/chatgpt_paraphraser_on_T5_base", gpu=0, batch_size=128, truncate=True, verbose=True):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def paraphrase(
        question,
        num_beams=5,
        device=0,
        num_beam_groups=5,
        num_return_sequences=5,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        temperature=0.8,
        max_length=512
    ):
        input_ids = tokenizer(
            f'paraphrase: {question}',
            return_tensors="pt", padding="longest",
            max_length=max_length,
            truncation=True,
        ).input_ids.to(device)
        
        outputs = model.generate(
            input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams, num_beam_groups=num_beam_groups,
            max_length=max_length, diversity_penalty=diversity_penalty
        )

        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return res

    paraphrases = []
    for i in tqdm.tqdm(range(0,len(list_of_assay_descriptions), 1)):
        batch = list_of_assay_descriptions[i:i+batch_size]
        paras = paraphrase(batch, device=device)
        paraphrases.append(paras)
        if verbose and i % 100 == 0:
            logger.info(f'batch {i} paraphrases: {paras}')
    return paraphrases



if __name__ == '__main__':

    parser = argparse.ArgumentParser('Compute features for a collection of PubChem assay descriptions.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--assay_path', default='./data/pubchem23/assay_names.parquet', help='Path to a parquet file with assay index to AID for which to extract features.')
    parser.add_argument('--augmentation_type', default='', help='Augmentation type to use. Options: ')
    parser.add_argument('--num_augmentations', type=int, default=5, help='Number of augmentations to generate for each assay description.')
    parser.add_argument('-c','--columns', nargs='+', help='Columns to use for the assay descriptions. default: title and subtitle', default=['title', 'subtitle'])
    parser.add_argument('-g', "--gpu", type=int, default=0, help="GPU to use for augmentation.")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add to the output file name.")
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

    logger.info("RUNNING AUGMENTATION")

    if args.augmentation_type == 'transformers_paraphrase':
        paraphrases = transformers_paraphrase(list_of_assay_descriptions)
    if args.augmentation_type == 'openchat':
        paraphrases = openchat_paraphrase(list_of_assay_descriptions, num_augmentations=args.num_augmentations, verbose=True)
    else:
        raise ValueError(f'Augmentation type {args.augmentation_type} not supported.')
    
    for i in range(len(paraphrases[0])):
        df[f"augmented_{args.columns}_{i}"] = [p[i] for p in paraphrases]

    suffix = "_"+args.suffix if args.suffix else ""
    new_fn = args.assay_path.replace('.parquet', f'_augmented_{args.augmentation_type}{suffix}.parquet')
    df.to_parquet(new_fn)
    
    logger.info(f'Save assay parquet file to {new_fn}')
