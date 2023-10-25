# PubChem

## PubChem18 (used in the paper)
For the PubChem dataset used in the paper run the following commands, which downloads, unzips and deletes the zip-file:
```bash
wget -N -r https://cloud.ml.jku.at/s/2ybfLRXWSYb4DZN/download -O pubchem18.zip
unzip pubchem18.zip; rm pubchem18.zip
```

## PubChem23 / up-to-date PubChem 
To setup the PubChem pretraining dataset with the latest version call:
```
python clamp/dataset/prep_pubchem.py --data_dir=./data/pubchem23/
```
This downloads the PubChem-database and preprocesses it.
You may manually delete the raw PubChem Folder (data_dir/`ftp.ncbi.nlm.nih.gov`) after running this command.
Get some coffee, this will take quite some time ;)

For reproducibility, data is also available through
```bash
wget -N -r https://cloud.ml.jku.at/s/fi83oGMN2KTbsNQ/download -O pubchem23.zip
unzip pubchem23.zip
rm pubchem23.zip
```

## Satistics

|                                 | PubChem23 (2023-02)   | PubChem18   | PubChem HTS   | FSMOL (v1)   |
|:--------------------------------|:------------------------|:-----------------|:--------------|:---------|
| # measurements                  | 245,234,081             | 223,219,241      | 143,886,653   | 501,366  |
| # compounds                     | 3,475,417               | 2,120,811        | 715,231       | 240,465  |
| # assays                        | 521,603                 | 21,002           | 582           | 5,135    |
| Mean # compounds / assay        | 462.69                  | 10,628.48        | 247,227.93    | 96.32    |
| Median # compounds / assay      | 2.00                    | 35.00            | 304,804.00    | 46.00    |
| % active                        | 2.58                    | 1.51             | 0.70          | 46.48    |
| Mean % active per assay         | 80.57                   | 79.46            | 1.04          | 47.17    |
| Median % active per assay       | 100.00                  | 100.00           | 0.42          | 48.84    |
| Source                          | PubChem 2023                | PubChem 2018     | PubChem 2018      | ChEMBL27 |
| % of assays with only one class | 95.54                   | 74.54            | 1.37          | 0.00     |
| % density                       | 0.01                    | 0.50             | 34.57         | 0.04     |

# Compound and Assay Encodings

To compute the compound encodings as input for your model run
```bash
python clamp/dataset/encode_compound.py \
--compounds=./data/pubchem23/compound_names.parquet \
--compound2smiles=./data/pubchem23/compound_smiles.parquet \
--fp_type=morganc+rdkc --fp_size=8096
```

To compute the assay encodings as input for your model run
```bash
python clamp/dataset/encode_assay.py --encoding=clip
```
or use ```--encoding=lsa```.


