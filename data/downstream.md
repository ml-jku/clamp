# Downstream datasets

To download the preprocessed downstream datasets call
```bash
wget -N -r https://cloud.ml.jku.at/s/pyJMm4yQeWFM2gG/download -O downstream.zip
unzip downstream.zip; rm downstream.zip
```

To download an preprocess the downstream datasets from source call.
```python clamp/dataset/prep_moleculenet.py```
(Doesn't include Tox21-10k)

# check for overlap between two datasets

```bash
python clamp/dataset/overlap.py \
    --dset_path=./data/pubchem23/ \
    --standardize_smiles=True \
    --cids_overlap_path=./data/pubchem23/cidx_overlap_moleculenet.npy \
    --downstream_dsets ./data/moleculenet/tox21/ ./data/moleculenet/toxcast ./data/moleculenet/bace_c ./data/moleculenet/hiv \
                       ./data/moleculenet/bbbp ./data/moleculenet/sider ./data/moleculenet/clintox ./data/moleculenet/tox21_10k 
```