from clamp.dataset import InMemoryClamp
from clamp import utils
from clamp.utils import set_device
import os
import pandas as pd
from loguru import logger
import random
import torch
import numpy as np

import mlflow
import argparse
import wandb
from time import time

from loguru import logger

from pathlib import Path
import mlflow
import argparse
import wandb
from time import time


"""example call:
python clamp/train.py \
    --dataset=./data/fsmol \
    --split=FSMOL_split \
    --assay_mode=clip \
    --compound_mode=morganc+rdkc 
"""

""" training pubchem23 without downstream datasets
python clamp/train.py --dataset=./data/pubchem23/ --split=time_a --assay_mode=clip --batch_size=8192 --dropout_hidde=0.3 --drop_cidx_path=./data/pubchem23/cidx_overlap_moleculenet.npy --train_subsample=10e6 --wandb --experiment=pretrain

"""


def parse_args_override(override_hpjson=True):
    parser = argparse.ArgumentParser('Train and test a single run of clamp-Activity model. Overrides arguments from hyperparam-file')
    parser.add_argument('-f', type=str)
    parser.add_argument('--dataset', type=str, default='./data/fsmol/', help='Path to a prepared dataset directory.')
    parser.add_argument('--assay_mode', type=str, default='lsa', help='Type of assay features ("clip", "biobert" or "lsa").')
    parser.add_argument('--compound_mode', type=str, default='morganc+rdkc', help='Type of compound features (default: morganc+rdkc)')
    parser.add_argument('--hyperparams',type=str, default='./hparams/default.json', help='Path to hyperparameters to use in training (json, Hyperparams, or logs).')
    
    parser.add_argument('--checkpoint', help='Path to a model-optimizer PyTorch checkpoint from which to resume training.', metavar='')
    parser.add_argument('--experiment', type=str, default='debug', help='Name of MLflow experiment where to assign this run.', metavar='')
    parser.add_argument('--random', action='store_true', help='Forget about the specified model and run a random baseline.')

    #parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use for training (default: AdamW).')
    #parser.add_argument('--l2', type=float, default=0.01, help='Weight decay to use for training (default: 0.01).')
    #parser.add_argument('--loss_fun', type=str, default='BCE', help='Loss function to use for training (default: BCE).')
    #parser.add_argument('--epoch_max', type=int, default=50, help='Maximum number of epochs to train for (default: 100).')
    #parser.add_argument('--lr_ini', type=float, default=1e-5, help='Initial learning rate (default: 1e-5).')
    
    parser.add_argument('--gpu', type=str, default="0", help='GPU number to use. Default: 0', metavar='')
    parser.add_argument('--seed', type=int, default=None, help='seed everything with provided seed, default no seed')

    parser.add_argument('--split',type=str, default='time_a_c',help="split-type default: time_a_c for time based assay and compound split, other options time_a, time_c, random_{seed}, or column of activity.parquet triplet")
    parser.add_argument('--support_set_size',type=int, default='0',help="per task how many to add from test- as well as valid- to the train-set default=0 = zero-shot")
    parser.add_argument('--train_only_actives', action='store_true', help='train only with active molecules')
    parser.add_argument('--drop_cidx_path', type=str, default=None, help='Path to a file containing a np of cidx (NOT CIDs) to drop from the dataset.')

    parser.add_argument('--verbose','-v', type=int, default=0, help='verbosity level default=0')
    parser.add_argument('--wandb','-w', action='store_true', help='wandb logging on')
    parser.add_argument('--bf16', action='store_true', help='use bfloat16 for training')
    
    args, unknown = parser.parse_known_args()
    keypairs = dict([unknown[i:i+2] for i in range(0, len(unknown), 1) if unknown[i].startswith("--") and not (unknown[i+1:i+2]+["--"])[0].startswith("--")])

    hparams = utils.get_hparams(path=args.hyperparams, mode='json', verbose=args.verbose)
    
    if override_hpjson:
        from clamp.utils import NAME2FORMATTER
        for k,v in NAME2FORMATTER.items():
            if (k not in args):
                default = hparams.get(k, None)
                parser.add_argument('--'+k, type=v, default=default)
                if (k in keypairs):
                    logger.info(f'{k} from hparam file will be overwritten')

        args = parser.parse_args()

    if args.nonlinearity is None:
        args.nonlinearity = 'ReLU'
    if args.compound_layer_sizes is None:
        logger.info('no compound_layer_sizes provided, setting to hidden_layers')
        args.compound_layer_sizes = args.hidden_layers
    if args.assay_layer_sizes is None:
        logger.info('no assay_layer_sizes provided, setting to hidden_layers')
        args.assay_layer_sizes =  args.hidden_layers

    return args


def setup_dataset(dataset='./data/pubchem23/', assay_mode='lsa', compound_mode='',
        split='time_a_c', verbose=False, support_set_size=0, drop_cidx_path=None, train_only_actives=False, **kwargs):
    """
    Setup the dataset given a dataset-path
    Loads an InMemoryclamp object containing 
    support_set_size ... 0, adding {support_set_size} samples from test and from valid to train (per assay/task)
    train_only_actives ... False,  only uses the active compounds
    drop_cidx_path ... None, path to a npy file containing cidx (NOT CIDs) to drop from the dataset
    """
    dataset = Path(dataset)
    clamp_dl = InMemoryClamp(
        root=dataset,
        assay_mode=assay_mode,
        compound_mode=compound_mode,
        verbose=verbose,
    )
    if split=='time_a_c': # requiers that assay and compound ids are sorted by time
        train_idx = clamp_dl.subset(
            c_high=clamp_dl.compound_cut['train'] - 1,
            a_high=clamp_dl.assay_cut['train'] - 1
        )
        valid_idx = clamp_dl.subset(
            c_low=clamp_dl.compound_cut['train'],
            c_high=clamp_dl.compound_cut['valid'] - 1,
            a_low=clamp_dl.assay_cut['train'],
            a_high=clamp_dl.assay_cut['valid'] - 1
        )
        test_idx = clamp_dl.subset(
            c_low=clamp_dl.compound_cut['valid'],
            a_low=clamp_dl.assay_cut['valid'],
        )
    elif split=='time_a':
        train_idx = clamp_dl.subset(
            a_high=clamp_dl.assay_cut['train'] - 1
        )
        valid_idx = clamp_dl.subset(
            a_low=clamp_dl.assay_cut['train'],
            a_high=clamp_dl.assay_cut['valid'] - 1
        )
        test_idx = clamp_dl.subset(
            a_low=clamp_dl.assay_cut['valid'],
        )
    elif split=='dense':
        train_idx, valid_idx, test_idx = utils.get_dense_split(clamp_dl.activity_df)
    elif split=='all':
        train_idx = np.arange(len(clamp_dl.activity_df))
        valid_idx = np.array([])
        test_idx = np.array([])
    elif 'random' in split:
        seed = int(split.split('_')[-1])
        if 'pubchem' in str(dataset):
            val_size=0.1
            test_size=0.05
        else:
            val_size=0.0
            test_size=0.3
        train_idx, valid_idx, test_idx = utils.get_random_split(clamp_dl.activity_df, seed=seed, val_size=val_size, test_size=test_size)
    #elif 'scaffold' in split:
    #    from rdkit.Chem.Scaffolds import MurckoScaffold
    #    def get_scaffold(smiles, includeChirality=False):
    #        return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles, includeChirality=includeChirality)
    #    print('computing scaffolds')
    #    clamp_dl.compound_df['scaffold'] = clamp_dl.compound_df.CanonicalSMILES.apply(get_scaffold)
    #    # consider using indices instead of scaffold strings
    #    cid2scaffold = clamp_dl.compound_df.set_index('CID')['scaffold'].to_dict()
    #    cidx2cid = clamp_dl.compound_names.CID.to_dict()
    #    clamp_dl.activity_df['scaffold'] = clamp_dl.activity_df.compound_idx.apply(lambda x: cid2scaffold[cidx2cid[cidx]])
    #    from utils import scaffold_split
    #    train_idx, valid_idx, test_idx = utils.get_scaffold_split(clamp_dl.activity_df.scaffold, )

    #    train_idx, valid_idx, test_idx = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
    elif 'fewshot' in split:
        balanced=False
        sub_idx = clamp_dl.activity_df.index
        if 'test' in split:
            # only use the test set
            sub_idx = clamp_dl.activity_df[split.split('_test')[0]]=='test'
        if split[-1]=='b':
            balanced=True
            split = split[:-1]
        sss = int(split.split('_')[-1]) #support_set_size
        train_idx, valid_idx, test_idx = utils.get_fewshot_split(clamp_dl.activity_df.loc[sub_idx], support_set_size=sss, seed=kwargs.get('seed', 0), balanced=balanced)
    elif 'hts' in split:
        if split=='hts_helal':
            logger.info(f'loading hts split (PubChem HTS) from Helal et. al - Public Domain HTS Fingerprints: Design and Evaluation of Compound Bioactivity Profiles from PubChem Bioassay Repository')
            # helal is a subset of laufkoetter
            # does also remove compounds
            hts_AIDs = {2563, 2052, 488965, 504326, 2057, 602123, 588814, 1554, 2066, 588819, 651800, 602141, 493087, 624125, 493091, 504357, 2599, 624168, 624169, 493098, 651819, 651821, 2094, 2606, 588334, 2097, 2098, 2099, 504884, 588850, 588852, 602162, 602163, 588352, 588354, 2629, 489030, 489031, 588358, 493131, 624204, 2129, 2130, 504406, 504408, 2650, 504411, 504414, 2661, 504423, 493160, 588391, 540267, 540275, 588405, 602229, 504441, 720508, 588413, 1662, 1663, 720509, 2690, 493187, 602244, 504454, 540295, 624267, 624268, 504462, 540308, 588436, 602261, 2716, 2717, 720543, 602274, 2216, 602281, 504490, 588458, 2221, 540336, 624304, 651957, 651958, 588473, 588475, 493244, 2751, 720582, 588489, 504523, 540364, 588492, 588493, 588497, 588499, 720596, 588501, 743126, 651999, 624352, 624354, 449763, 2280, 463082, 602346, 2796, 652010, 504558, 652017, 2805, 2806, 624377, 2300, 623870, 463104, 504577, 588549, 504582, 652039, 720648, 2825, 2314, 652048, 434962, 652051, 1813, 1814, 652054, 602396, 434973, 624414, 602399, 624415, 624416, 651550, 652067, 463141, 1832, 651560, 434989, 504621, 651572, 435003, 720700, 435005, 602429, 720702, 720704, 602438, 602440, 504651, 2380, 504652, 435022, 588621, 602449, 624466, 1875, 504660, 624467, 435030, 463190, 651602, 463195, 686940, 1885, 652126, 743269, 1899, 463212, 743279, 504690, 651636, 686964, 1910, 588664, 504700, 651647, 588674, 2435, 504707, 488839, 2445, 488847, 504720, 588692, 485270, 485272, 485273, 492953, 485275, 492956, 1950, 2462, 488862, 504734, 624037, 624038, 687014, 624040, 1962, 492972, 651699, 588726, 651702, 1979, 504766, 488895, 488896, 1987, 488899, 651718, 504775, 651719, 651724, 651725, 493008, 493011, 493012, 2520, 2521, 2524, 2016, 652257, 485346, 504803, 743397, 1511, 2023, 2025, 504810, 2540, 2029, 485358, 493036, 2544, 504812, 2557, 624126, 624127}
            hts_AIDs_test = {488965, 602123, 504462, 504720, 602261, 485275, 488862, 493091, 624040, 602281, 504621, 624304, 588852, 435003, 435005, 588354, 588358, 602438, 493131, 435022, 588499, 493012, 435030, 504406, 2521, 2650, 624352, 485346, 504803, 504690, 588405, 2557, 624126}
        else:
            # from Laufkoetter et. al - Combining structural and bioactivity-based fingerprints improves prediction performance and scaffold hopping capability, JCIM
            logger.info(f'loading hts split (PubChem HTS) from Laufkoetter et. al - Combining structural and bioactivity-based fingerprints improves prediction performance and scaffold hopping capability')
            hts_AIDs = {2052,2057,602123,2066,743445,2071,602141,2094,2097,602162,2099,2098,602163,2129,2130,2156,602229,2174,2177,602244,602247,602248,602250,602252,602261,2205,602274,2216,602281,2221,2227,2234,2235,2237,2239,2247,602329,463075,602340,602342,463079,2280,602346,463082,2291,602363,2300,463104,463111,2314,463115,2315,602393,602396,651550,602399,602405,463141,651560,602410,651572,602429,651582,463165,602438,602440,2380,602449,651602,463187,463190,463193,651610,463195,463210,463212,368,602481,2417,651636,373,374,651640,651647,2435,651654,651658,2445,651661,2462,422,651687,2472,429,651699,436,651702,438,651704,441,440,651710,651711,449,453,651718,651719,651723,651724,651725,1053133,463,460,465,2520,2521,2524,485,487,488,2540,2544,501,2550,2553,2557,2563,504326,522,524,525,1053197,527,528,529,651800,538,539,504357,2599,552,555,556,651821,588334,588335,560,561,2606,651819,559,565,567,570,571,573,574,588352,577,588354,581,588358,2629,2642,598,504406,504408,2648,2650,504411,604,602,504414,612,2661,614,588391,615,504423,619,622,623,625,626,628,629,630,631,588405,633,504441,588413,639,640,641,2690,645,504454,648,504462,588436,2716,2717,680,504490,588458,684,2732,685,687,688,693,651958,651957,588473,697,588475,701,2751,704,707,708,709,710,588489,504523,588492,717,588493,719,720,588497,588499,588501,727,729,731,588511,651999,736,738,739,740,745,652010,746,2796,2797,504558,751,750,652017,757,2806,2805,760,761,758,759,764,652031,504577,588549,504582,775,652039,777,2825,778,781,782,784,652048,434962,652051,652054,793,434970,434971,796,434973,798,799,795,797,802,803,652067,804,434982,800,808,811,434989,504621,813,818,819,504634,435003,828,435005,827,833,834,836,841,504651,504652,588621,435022,847,652115,504660,853,435030,588627,686940,861,652126,862,868,871,873,686954,878,504690,504692,686964,588664,652154,504700,652162,652163,588674,588675,504707,898,588676,686992,504720,588692,686996,920,504734,932,652197,687014,687016,940,950,588726,952,951,504766,504775,652257,504803,504810,504812,1006,1007,1008,1009,1010,1012,1016,1019,1020,1021,1022,1024,1025,1027,1029,1032,588814,1040,588819,1044,1046,1063,1066,588850,588852,504884,1085,1117264,1117267,1135,1136,1203,1214,1216,449728,1217,449739,1229,1230,1235,1236,1239,1240,1242,1246,1251,449763,449768,1273,1274,1276,623870,623877,1285,1296,1304,623901,1321,1325,1326,1359,1362,1377,1381,1385,1415,488839,1416,1422,488847,1423,1424,1430,492953,492956,488862,1439,1440,1441,1443,624037,1446,624038,1448,624040,492972,488895,488896,488899,1481,488910,1486,493008,493011,493012,1496,1497,488922,1499,1509,1510,1511,1515,493036,1527,1529,1530,1531,1532,1533,624125,624127,493056,624126,488965,488969,488975,488977,1554,1556,624151,493084,493087,493091,624168,624169,493098,489028,489030,489031,493131,624204,493160,540267,540275,540277,1656,720508,720509,1662,1663,624256,720511,624255,493187,540295,1672,624267,624268,540308,720543,1700,1706,540333,624304,540336,743093,493244,720574,720582,540364,720596,743126,624352,624354,624377,1789,1792,720647,720648,1800,1813,1814,624414,624415,624416,1825,1822,1823,1832,1845,720700,720702,720704,720706,1861,743238,743247,624466,624467,1875,1885,624483,743269,1899,743279,1906,1910,743287,1918,485270,485272,485273,485275,1947,1950,1962,1974,1979,1987,485317,1992,485344,2016,485346,485347,743397,743398,2023,2025,2029,485358}
            # laufkoetter removed assays with less than 20k compounds #TODO not done here
            hts_remove = {1792, 488969, 1044, 434970, 434971, 1823, 1440, 434982, 552, 811, 540333, 438, 833, 489028, 488910, 487, 622, 623, 625, 2291, 652031}
            # for each assay he removed the test assay from the fingerprint
            # this is not practival in this setting 
            hts_AIDs_test = {463104, 1273, 504454, 522, 527, 623901, 798, 624414, 555, 2606, 560, 720700, 2129, 588497, 504406, 2280, 746, 1515, 2540, 1006, 2544, 686964, 2553, 602363}
            
        
        train_set_prop = 0.8 # do train-val split
        hts_AIDs_train = set(list(hts_AIDs-hts_AIDs_test)[:int(len(hts_AIDs-hts_AIDs_test)*train_set_prop)])
        hts_AIDs_valid = set(list(hts_AIDs-hts_AIDs_test)[int(len(hts_AIDs-hts_AIDs_test)*train_set_prop):])

        # no intersection
        assert len(hts_AIDs_valid.intersection(hts_AIDs_test))==0
        assert len(hts_AIDs_train.intersection(hts_AIDs_valid))==0
        assert len(hts_AIDs_train.intersection(hts_AIDs_test))==0

        # map from aid to idx
        idx2aid = clamp_dl.assay_names.AID.to_dict()
        aid2idx = {v:k for k,v in idx2aid.items()}
        hts_AIDs_train_idx = set(map(lambda k: aid2idx[k], hts_AIDs_train))
        hts_AIDs_valid_idx = set(map(lambda k: aid2idx[k], hts_AIDs_valid))
        hts_AIDs_test_idx = set(map(lambda k: aid2idx[k], hts_AIDs_test))

        train_idx = clamp_dl.activity_df.assay_idx[clamp_dl.activity_df.assay_idx.apply(lambda k: k in hts_AIDs_train_idx)].index.values
        valid_idx = clamp_dl.activity_df.assay_idx[clamp_dl.activity_df.assay_idx.apply(lambda k: k in hts_AIDs_valid_idx)].index.values
        test_idx = clamp_dl.activity_df.assay_idx[clamp_dl.activity_df.assay_idx.apply(lambda k: k in hts_AIDs_test_idx)].index.values

    else:
        logger.info(f'loading split info from activity.parquet triplet-list under the column split={split}')
        try:
            splis = pd.read_parquet(dataset/'activity.parquet')[split]
        except KeyError:
            raise ValueError(f'no split column {split} in activity.parquet', pd.read_parquet(dataset/'activity.parquet').columns, 'columns available')
        train_idx, valid_idx, test_idx = [splis[splis==sp].index.values for sp in ['train','valid','test']]

    if support_set_size>0:
        logger.warning('ONLY SUPPORTED FOR FSMOL')
        logger.info(f'{support_set_size}-shot option: adding {support_set_size} samples from test and from valid to train (per task)')
        triplet_df = clamp_dl.activity_df
        #triplet_df = pd.read_parquet('./data/FSMOL/'+'activity.parquet')

        def sample_with_max(x, support_set_size=support_set_size):
            return x.sample(min(support_set_size, len(x)))
        val2train = triplet_df.loc[valid_idx].groupby('assay_idx').apply(sample_with_max).droplevel([0]).index.values
        test2train = triplet_df.loc[test_idx].groupby('assay_idx').apply(sample_with_max).droplevel([0]).index.values
        # compute new indices
        train_idx = np.concatenate([train_idx, val2train, test2train]) #add to train
        valid_idx = np.array(list(set(valid_idx)-set(val2train))) # remove from valid
        test_idx = np.array(list(set(test_idx)-set(test2train))) #remove from test
        np.random.shuffle(train_idx)
        np.random.shuffle(valid_idx)
        np.random.shuffle(test_idx)

    if train_only_actives:
        logger.info('only training with active molecules')
        # remove inactive from train_idx
        train_idx = train_idx[ clamp_dl.activity_df.loc[train_idx]['activity'] == 1 ]

    if drop_cidx_path:
        logger.info(f"drop_cidx_path: {drop_cidx_path}")
        dropidx = set(list(np.load(drop_cidx_path, allow_pickle=True).tolist()))

        for spl_idx, spl in enumerate([train_idx, valid_idx]):
            trtmp = clamp_dl.activity_df.loc[spl].compound_idx.apply(lambda k: k not in dropidx)
            logger.info(f"dropped {(~trtmp).mean()*100:2.2f}% {(~trtmp).sum()} from {['train','valid'][spl_idx]}")
            spl = trtmp.loc[trtmp].index.values # new train_idx

            # override train_idx, valid_idx
            if spl_idx==0:
                train_idx = spl
            else:
                valid_idx = spl

    if verbose:
        logger.info(f'split: train: {len(train_idx)} valid: {len(valid_idx)} test {len(test_idx)}')
        len_all = len(train_idx)+len(valid_idx)+len(test_idx)
        logger.info(f'ratio train/valid/test: {len(train_idx)/len_all*100:2.1f}/{len(valid_idx)/len_all*100:2.1f}/{len(test_idx)/len_all*100:2.1f}')

    return clamp_dl, train_idx, valid_idx, test_idx

def main(args):
    hparams = args.__dict__

    mlflow.set_experiment(args.experiment)
    
    if args.seed:
        from clamp.utils import seed_everything
        seed_everything(args.seed)
        logger.info(f'seeded everything with seed {args.seed}')

    clamp_dl, train_idx, valid_idx, test_idx = setup_dataset(**args.__dict__)
    assert set(train_idx).intersection(set(valid_idx)) == set()
    assert set(train_idx).intersection(set(test_idx)) == set()

    if args.wandb:
        #runname = args.experiment+'_'+args.dataset.split('pubchem_')[-1]+'_'+args.split+'_'+args.assay_mode+'_'
        runname = args.experiment+args.split[-1]+args.assay_mode[-1]
        if args.random: 
            runname += 'random'
        else:
            runname = str(runname)+''
            runname += str(args.model)
        runname += ''.join([chr(random.randrange(97, 97 + 26)) for _ in range(3)])
        wandb.init(project='clamp', entity='phseidl', name=runname, config=args.__dict__)

    device = set_device(gpu=args.gpu, verbose=args.verbose)

    metrics_df = pd.DataFrame()

    try:
        with mlflow.start_run():
            mlflowi = mlflow.active_run().info
    
            if args.checkpoint is not None:
                mlflow.set_tag(
                    'mlflow.note.content',
                    f'Resumed training from {args.checkpoint}.'
                )

            if 'assay_mode' in hparams:
                if hparams['assay_mode'] != args.assay_mode:
                    # assay features were already loaded in `InMemoryclamp` using `args.assay_mode`
                    logger.warning(f'Assay features are "{args.assay_mode}" in command line but \"{hparams["assay_mode"]}\" in hyperparameter combination.')
                    logger.warning(f'Command line "{args.assay_mode}" is the prevailing option.')
                    hparams['assay_mode'] = args.assay_mode
            else:
                mlflow.log_param('assay_mode', args.assay_mode)
            mlflow.log_params(hparams)

            if args.random:
                mlflow.set_tag(
                    'mlflow.note.content',
                    'Ignore the displayed parameters. Metrics correspond to predictions randomly drawn from U(0, 1).'
                )
                utils.random(
                    clamp_dl,
                    test_idx=test_idx,
                    run_info=mlflowi,
                    verbose=args.verbose
                )

            else:
                metrics_df = utils.train_and_test(
                    clamp_dl,
                    train_idx=train_idx,
                    valid_idx=valid_idx,
                    test_idx=test_idx,
                    hparams=hparams,
                    run_info=mlflowi,
                    checkpoint_file=args.checkpoint,
                    device=device,
                    bf16=args.bf16,
                    verbose=args.verbose,
                )

    except KeyboardInterrupt:
        logger.error('Training manually interrupted. Trying to test with last checkpoint')
        metrics_df = utils.test(
            clamp_dl,
            train_idx=train_idx,
            test_idx=test_idx,
            hparams=hparams,
            run_info=mlflowi,
            device=device,
            verbose=args.verbose
        )
    
    #logger.info('Model-Checkpoint saved to: ', mlflowi.get_artifact_uri())
    #print(mlflowi)
        
if __name__ == '__main__':
    args = parse_args_override()

    run_id = str(time()).split('.')[0]
    fn_postfix = f'{args.experiment}_{run_id}' 

    if args.verbose>=1:
        logger.info('Run args: ', os.getcwd()+__file__, args.__dict__)

    main(args)