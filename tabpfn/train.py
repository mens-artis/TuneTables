import os
import itertools
import argparse
import time
import datetime
import yaml
import json
from contextlib import nullcontext
import copy
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import autograd
from torch.utils.data import Subset

import tabpfn.utils as utils
from transformer import TransformerModel
from tabpfn.scripts.tabular_evaluation import predict_wrapper
from tabpfn.utils import get_cosine_schedule_with_warmup, get_openai_lr, StoreDictKeyPair, get_weighted_single_eval_pos_sampler, get_uniform_single_eval_pos_sampler
import tabpfn.priors as priors
import sys
script_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(script_path))
sys.path.insert(0, os.path.join(project_path + "/tabpfn/priors"))

import real

import tabpfn.encoders as encoders
import tabpfn.positional_encodings as positional_encodings
from utils import init_dist, seed_all, EmbeddingConcatenator

from torch.cuda.amp import autocast, GradScaler
from torch import nn
from torch.utils.data import Dataset

import numpy as np

import uncertainty_metrics.numpy as um
#from priors.real import process_data
from tabpfn.utils import normalize_data, to_ranking_low_mem, remove_outliers, NOP, normalize_by_used_features_f

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, RobustScaler, PowerTransformer


def process_data(
    dataset,
    train_index,
    val_index,
    test_index,
    verbose=False,
    scaler="None",
    one_hot_encode=False,
    impute=True,
    args=None,
):
    

    X_train, y_train = dataset.X[train_index], dataset.y[train_index]
    X_val, y_val = dataset.X[val_index], dataset.y[val_index]
    X_test, y_test = dataset.X[test_index], dataset.y[test_index]

    # validate the scaler
    assert scaler in ["None"], f"scaler not recognized: {scaler}"


    num_mask = np.ones(dataset.X.shape[1], dtype=int)
    num_mask[dataset.cat_idx] = 0

    # Impute numerical features
    if impute:
        num_idx = np.where(num_mask)[0]

        # The imputer drops columns that are fully NaN. So, we first identify columns that are fully NaN and set them to
        # zero. This will effectively drop the columns without changing the column indexing and ordering that many of
        # the functions in this repository rely upon.
        fully_nan_num_idcs = np.nonzero(
            (~np.isnan(X_train[:, num_idx].astype("float"))).sum(axis=0) == 0
        )[0]
        if fully_nan_num_idcs.size > 0:
            X_train[:, num_idx[fully_nan_num_idcs]] = 0
            X_val[:, num_idx[fully_nan_num_idcs]] = 0
            X_test[:, num_idx[fully_nan_num_idcs]] = 0

        # Impute numerical features, and pass through the rest
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer())])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_idx),
                ("pass", "passthrough", dataset.cat_idx),
                # ("cat", categorical_transformer, categorical_features),
            ],
            # remainder="passthrough",
        )
        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)
        X_test = preprocessor.transform(X_test)

        # Re-order columns (ColumnTransformer permutes them)
        perm_idx = []
        running_num_idx = 0
        running_cat_idx = 0
        for is_num in num_mask:
            if is_num > 0:
                perm_idx.append(running_num_idx)
                running_num_idx += 1
            else:
                perm_idx.append(running_cat_idx + len(num_idx))
                running_cat_idx += 1
        assert running_num_idx == len(num_idx)
        assert running_cat_idx == len(dataset.cat_idx)
        X_train = X_train[:, perm_idx]
        X_val = X_val[:, perm_idx]
        X_test = X_test[:, perm_idx]

    if scaler != "None":
        if verbose:
            print(f"Scaling the data using {scaler}...")
        X_train[:, num_mask] = scaler_function.fit_transform(X_train[:, num_mask])
        X_val[:, num_mask] = scaler_function.transform(X_val[:, num_mask])
        X_test[:, num_mask] = scaler_function.transform(X_test[:, num_mask])

    if one_hot_encode:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        new_x1 = ohe.fit_transform(X_train[:, dataset.cat_idx])
        X_train = np.concatenate([new_x1, X_train[:, num_mask]], axis=1)
        new_x1_test = ohe.transform(X_test[:, dataset.cat_idx])
        X_test = np.concatenate([new_x1_test, X_test[:, num_mask]], axis=1)
        new_x1_val = ohe.transform(X_val[:, dataset.cat_idx])
        X_val = np.concatenate([new_x1_val, X_val[:, num_mask]], axis=1)
        if verbose:
            print("New Shape:", X_train.shape)

    args.num_features = X_train.shape[1]
    # create subset of dataset if needed
    if (
        args is not None
        and (args.subset_features > 0 or args.subset_rows > 0)
        and (
            args.subset_features < args.num_features or args.subset_rows < len(X_train)
        )
    ):
            
        if getattr(dataset, "ssm", None) is None:
            subset_maker = real.SubsetMaker(
                args.subset_features,
                args.subset_rows,
                args.subset_features_method,
                args.subset_rows_method,
                give_full_features = args.summerize_after_prep, #if we summerize after prep, we don't want to summerize here
            )
        X_train, y_train = subset_maker.make_subset(
            X_train,
            y_train,
            split="train",
            seed=args.rand_seed,
        )
        if args.subset_features < args.num_features:
            X_val, y_val = subset_maker.make_subset(
                X_val,
                y_val,
                split="val",
                seed=args.rand_seed,
            )
            X_test, y_test = subset_maker.make_subset(
                X_test,
                y_test,
                split="test",
                seed=args.rand_seed,
            )
    return {
        "data_train": (X_train, y_train),
        "data_val": (X_val, y_val),
        "data_test": (X_test, y_test),
    }

class Losses():
    gaussian = nn.GaussianNLLLoss(full=True, reduction='none')
    mse = nn.MSELoss(reduction='none')
    def ce(num_classes):
        num_classes = num_classes.shape[0] if torch.is_tensor(num_classes) else num_classes
        return nn.CrossEntropyLoss(reduction='none', weight=torch.ones(num_classes))
    bce = nn.BCEWithLogitsLoss(reduction='none')


class TabDS(Dataset):
    def __init__(self, X, y):
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X.copy().astype(np.float32))
        else:
            self.X = X

        self.y_float = torch.from_numpy(y.copy().astype(np.float32))
        self.y = torch.from_numpy(y.copy().astype(np.int64))

        print(f"TabDS: X.shape = {self.X.shape}, y.shape = {self.y.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        #(X,y) data, y target, single_eval_pos
        ret_item = tuple([self.X[idx], self.y_float[idx]]), self.y[idx], torch.tensor([])
        # print("ret item:", ret_item[0][0], ret_item[0][1], ret_item[1], ret_item[2])
        return ret_item


def preprocess_input(eval_xs, preprocess_transform, summerize_after_prep):
    import warnings

    if preprocess_transform != 'none':
        if preprocess_transform == 'power' or preprocess_transform == 'power_all':
            pt = PowerTransformer(standardize=True)
        elif preprocess_transform == 'quantile' or preprocess_transform == 'quantile_all':
            pt = QuantileTransformer(output_distribution='normal')
        elif preprocess_transform == 'robust' or preprocess_transform == 'robust_all':
            pt = RobustScaler(unit_variance=True)
    eval_position = eval_xs.shape[0]
    eval_xs = normalize_data(eval_xs, normalize_positions=eval_position)

    warnings.simplefilter('error')
    if preprocess_transform != 'none':
        eval_xs = eval_xs.cpu().numpy()
        feats = set(range(eval_xs.shape[1]))
        for col in feats:
            try:
                pt.fit(eval_xs[0:eval_position, col:col + 1])
                trans = pt.transform(eval_xs[:, col:col + 1])
                # print(scipy.stats.spearmanr(trans[~np.isnan(eval_xs[:, col:col+1])], eval_xs[:, col:col+1][~np.isnan(eval_xs[:, col:col+1])]))
                eval_xs[:, col:col + 1] = trans
            except:
                pass
        eval_xs = torch.tensor(eval_xs).float()
    warnings.simplefilter('default')

    eval_xs = eval_xs.unsqueeze(1)

    eval_xs = remove_outliers(eval_xs, normalize_positions=eval_position)
    # Rescale X
    #hard-coded
    max_features = 100

    if summerize_after_prep:
        eval_xs = normalize_by_used_features_f(eval_xs, min(eval_xs.shape[-1],max_features), max_features,
                                            normalize_with_sqrt=False)        
    else:
        eval_xs = normalize_by_used_features_f(eval_xs, eval_xs.shape[-1], max_features,
                                                normalize_with_sqrt=False)

    eval_xs = eval_xs.squeeze(1)
    return eval_xs

def get_train_dataloader(ds, bptt=1000, shuffle=True, num_workers=1, drop_last=True, agg_k_grads=1):
        # old_bptt = bptt
        dl = DataLoader(
            ds, batch_size=bptt, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
        )
        if len(dl) == 0:
            ds_len = len(ds)
            n_batches = 10
            bptt = int(ds_len // n_batches)
            # bptt = int(bptt // 2)
            dl = DataLoader(
                ds, batch_size=bptt, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
            )
        while len(dl) % agg_k_grads != 0:
            bptt += 1
            dl = DataLoader(
                ds, batch_size=bptt, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
            )
            # raise ValueError(f'Number of batches {len(dl)} not divisible by {agg_k_grads}, please modify aggregation factor.')
        # if old_bptt != bptt:
        #     print(f'Batch size changed from {old_bptt} to {bptt} to be divisible by {agg_k_grads} (with last batch dropped).')
        return dl, bptt

def SummarizeAfter(X, X_val, X_test, y, y_val, y_test, num_features, subset_features_method):

        SM = real.SubsetMaker(
                num_features,
                10e8, #subset_rows = 10^8 ablate this part here
                subset_features_method,
                "first", #args.subset_rows_method is not used anyhow
            )

        X, y = SM.make_subset(
            X,
            y,
            split="train",
        )

        X_val, y = SM.make_subset(
            X_val, 
            y,
            split="val",
        )
            
        X_test, y = SM.make_subset(
            X_test, 
            y,
            split="test",
        )
        if isinstance(X, torch.Tensor):
            return X.to(torch.float32), X_val.to(torch.float32), X_test.to(torch.float32)
        elif isinstance(X, np.ndarray):
            return torch.from_numpy(X).to(torch.float32), torch.from_numpy(X_val).to(torch.float32), torch.from_numpy(X_test).to(torch.float32)
        else:
            raise Exception(f"X is {type(X)}, not a tensor or numpy array")

#def train(args, priordataloader_class, criterion, encoder_generator, emsize=200, nhid=200, nlayers=6, nhead=2, dropout=0.0,
def train(args, dataset, criterion, encoder_generator, emsize=200, nhid=200, nlayers=6, nhead=2, dropout=0.0,
          epochs=10, steps_per_epoch=100, batch_size=200, bptt=10, lr=None, weight_decay=0.0, warmup_epochs=10, input_normalization=False,
          y_encoder_generator=None, pos_encoder_generator=None, decoder=None, extra_prior_kwargs_dict={}, scheduler=get_cosine_schedule_with_warmup,
          load_weights_from_this_state_dict=None, validation_period=10, single_eval_pos_gen=None, bptt_extra_samples=None, gpu_device='cuda:0',
          aggregate_k_gradients=1, verbose=True, style_encoder_generator=None, epoch_callback=None,
          initializer=None, initialize_with_model=None, train_mixed_precision=False, efficient_eval_masking=True, 
          boosting=False, boosting_lr=1e-3, boosting_n_iters=10, rand_init_ensemble=False, do_concat="", **model_extra_args
          ):



        
    device = gpu_device if torch.cuda.is_available() else 'cpu:0'
    print(f'Using {device} device')
    using_dist, rank, device = init_dist(device)
    start_time = time.time()
    max_time = extra_prior_kwargs_dict.get('max_time', 0)

    if extra_prior_kwargs_dict.get('pad_features', None):
        num_features = 100
    else:
        num_features = extra_prior_kwargs_dict.get('num_features', 100)

    if extra_prior_kwargs_dict.get('prior_type') == 'real':
        real_prior = True
    else:
        real_prior = False

    if extra_prior_kwargs_dict.get('prompt_tuning'):
        do_prompt_tuning = True
        prefix_size = extra_prior_kwargs_dict.get('tuned_prompt_size', 100)
    else:
        do_prompt_tuning = False
        prefix_size = 0



    single_eval_pos_gen = single_eval_pos_gen if callable(single_eval_pos_gen) else lambda: single_eval_pos_gen
    real_data_qty = extra_prior_kwargs_dict.get('real_data_qty', 0)
    if real_data_qty <= 0:
        real_data_qty = bptt

    print("Real data qty (for fitting): ", real_data_qty)

    def eval_pos_seq_len_sampler():
        single_eval_pos = single_eval_pos_gen()
        if bptt_extra_samples:
            return single_eval_pos, single_eval_pos + bptt_extra_samples
        else:
            return single_eval_pos, bptt

    def loop_translate(a, my_dict):
        new_a = np.empty(a.shape)
        if a.ndim == 1:
            for i,elem in enumerate(a):
                new_a[i] = my_dict.get(elem)
        elif a.ndim == 2:
            # print("In loop translate: ")
            # print("a shape: ", a.shape)
            # print("a: ", a[:5, ...])
            new_a = []
            for val in list(my_dict.keys()):
                new_a.append(a[:, val])
            if isinstance(new_a[0], np.ndarray):
                new_a = np.stack(new_a, axis=1)
            else:
                #torch tensor
                new_a = torch.stack(new_a, axis=1)
            # print("new_a shape: ", new_a.shape)
            # print("new_a: ", new_a[:5, ...])
        return new_a




    def make_datasets(extra_prior_kwargs_dict, do_permute=True, bptt = 0, steps_per_epoch=None):
        """  """
        dataset_built = False
        for i, split_dictionary in enumerate(dataset.split_indeces):
            # TODO: make stopping index a hyperparameter
            if i != extra_prior_kwargs_dict.get('split'): # config['split']:
                continue
            train_index = split_dictionary["train"]
            val_index = split_dictionary["val"]
            test_index = split_dictionary["test"]
            
            # run pre-processing & split data (list of numpy arrays of length num_ensembles)
            processed_data = process_data(
                dataset,
                train_index,
                val_index,
                test_index,
                verbose= extra_prior_kwargs_dict.get('verbose'), #config['verbose'],
                scaler="None",
                one_hot_encode=False,
                args=args,
            )
            X_train, y_train = processed_data["data_train"]
            X_val, y_val = processed_data["data_val"]
            X_test, y_test = processed_data["data_test"]
            n_features = X_train.shape[1]
            n_samples = X_train.shape[0]
            #config['num_classes'] = len(set(y_train))
            num_classes = len(set(y_train))
            #config['num_steps'] = len(X_train) // config['bptt']
            steps_per_epoch = len(X_train) // bptt

            if bptt > n_samples:
                print(f"WARNING: bptt {bptt} is larger than the number of samples in the training set, {n_samples}. Setting bptt=128.")
                bptt = 128

            priordataloader_class = [[X_train, y_train], [X_val, y_val], [X_test, y_test]]
            dataset_built = True
            break
        
        seed_all(extra_prior_kwargs_dict.get('rand_seed'))

        X, y = X_train, y_train
        ##X, y = priordataloader_class[0][0], priordataloader_class[0][1]
        # print("In make datasets: ")
        #print("unique y: ", np.unique(y))
        ##X_val, y_val = priordataloader_class[1][0], priordataloader_class[1][1]
        ##X_test, y_test = priordataloader_class[2][0], priordataloader_class[2][1]
        #shuffle data
        if do_permute:
            label_perm = np.random.permutation(num_classes)
        else:
            label_perm = np.arange(num_classes)

        invert_perm_map = {
            label_perm[i]: i for i in range(num_classes)
        }
        rev_invert_perm_map = {
            i: label_perm[i] for i in range(num_classes)
        }
        if do_permute:
            feat_idx = np.random.permutation(X.shape[1])
        else:
            feat_idx = np.arange(X.shape[1])
        
        idx = np.random.permutation(X.shape[0])
        X = X[idx, ...]
        y = y[idx, ...]
        # print("y: ", y[:20, ...])
        # print("Label perm: ", label_perm)
        y = loop_translate(y, rev_invert_perm_map)
        # for i in range(num_classes):
        #     new_y[i] = y[rev_invert_perm_map[i]]
        # print("New y: ", new_y[:20, ...])
        # y = label_perm[y[idx, ...]]
        X = X[:, feat_idx, ...]
        X_val = X_val[:, feat_idx, ...]
        X_test = X_test[:, feat_idx, ...]

        # label balancing
        num_classes = len(np.unique(np.unique(y)))
        if do_prompt_tuning and extra_prior_kwargs_dict.get('tuned_prompt_label_balance', 'equal') == 'proportional':
            label_weights = np.bincount(y) / len(y)
            label_weights = torch.from_numpy(label_weights).float().to(device)
        else:
            label_weights = None

        if extra_prior_kwargs_dict.get("do_preprocess", False):
            preprocess_type=extra_prior_kwargs_dict.get("preprocess_type", "none")
            summerize_after_prep=extra_prior_kwargs_dict.get("summerize_after_prep", "False")

            X = preprocess_input(torch.from_numpy(X.copy().astype(np.float32)), preprocess_type, summerize_after_prep)    
            X_val = preprocess_input(torch.from_numpy(X_val.copy().astype(np.float32)), preprocess_type, summerize_after_prep)  
            X_test = preprocess_input(torch.from_numpy(X_test.copy().astype(np.float32)), preprocess_type, summerize_after_prep)  


            print("X",X.dtype)
            ## SummarizeAfter
            if args.summerize_after_prep:
                X, X_val, X_test = SummarizeAfter(X, X_val, X_test, y, y_val, y_test, num_features, args.subset_features_method)            
            print("X",X.dtype)

        if X.shape[1] < num_features and extra_prior_kwargs_dict.get("pad_features", True):
            
            def pad_data(data):
                return torch.cat([data, torch.zeros(data.shape[0], num_features - data.shape[1])], dim=1)
            
            X = pad_data(X)
            X_val = pad_data(X_val)
            X_test = pad_data(X_test)


        print('aggregate_k_gradients',aggregate_k_gradients)


        train_ds = TabDS(X, y)
        val_ds = TabDS(X_val, y_val)
        test_ds = TabDS(X_test, y_test)


        return X, y, X_val, y_val, X_test, y_test, invert_perm_map, steps_per_epoch, num_classes, label_weights, train_ds, val_ds, test_ds
    


    def make_dataloaders(bptt=bptt):

        dl, bptt = get_train_dataloader(train_ds, 
                                  bptt=bptt, 
                                  shuffle=False, 
                                  num_workers=1, 
                                  drop_last=True, 
                                  agg_k_grads=aggregate_k_gradients
                                )

        val_dl = DataLoader(
            val_ds, batch_size=min(128, y_val.shape[0] // 2), shuffle=False, num_workers=1,
        )

        test_dl = DataLoader(
            test_ds, batch_size=min(128, y_test.shape[0] // 2), shuffle=False, num_workers=1,
        )
        # Fix the prior data TabPFN will use for fitting when including real data points
        X_data_for_fitting = []
        y_data_for_fitting = []
        #td is a list of tensors
        for idx, (td, _, _) in enumerate(dl):
            X_data_for_fitting.append(td[0])
            y_data_for_fitting.append(td[1])
            if idx == 10:
                break
        X_data_concat = torch.cat(X_data_for_fitting, dim=0)
        y_data_concat = torch.cat(y_data_for_fitting, dim=0)
        data_for_fitting = [X_data_concat, y_data_concat]
        return dl, val_dl, test_dl, bptt, data_for_fitting



    if real_prior:


        #load data
        not_zs = extra_prior_kwargs_dict.get('zs_eval_ensemble', 0) == 0
        seed_all(extra_prior_kwargs_dict.get('rand_seed'))


        data_for_fitting = None

        X, y, X_val, y_val, X_test, y_test, invert_perm_map, steps_per_epoch, num_classes, label_weights, train_ds, val_ds, test_ds = make_datasets(extra_prior_kwargs_dict, do_permute=not_zs, bptt=bptt, steps_per_epoch=steps_per_epoch)
        old_bptt = bptt
        dl, val_dl, test_dl, bptt, data_for_fitting  = make_dataloaders(bptt=bptt)
        if old_bptt != bptt:
            print("bptt changed from {} to {}".format(old_bptt, bptt))
            max_pos = int((len(data_for_fitting[0]) // 10) * (.8))
            if extra_prior_kwargs_dict.get('uniform_bptt', False):
                single_eval_pos_gen = lambda: np.random.randint(0, max_pos)
            else:
                single_eval_pos_gen = max_pos
        # print("Dataloader size: ", len(dl))
        if extra_prior_kwargs_dict.get('zs_eval_ensemble', 0) > 0:

            def tpc_data_eval(cl=1000, X=None, y=None, X_val=None, y_val=None, ens_size=1):
                    print("Num classes: ", num_classes)
                    from scripts.transformer_prediction_interface import TabPFNClassifier
                    start_time = time.time()
                    results = dict()
                    if cl > len(X):
                        cl = len(X) - 1
                    eval_model = TabPFNClassifier(device='cuda', 
                                                N_ensemble_configurations=ens_size, 
                                                base_path="/home/benfeuer/TabPFN-pt/tabpfn",
                                                # seed=None,
                                                seed=extra_prior_kwargs_dict.get('rand_seed', 0),
                                                batch_size_inference=1,
                                                )
                    eval_model.fit(X[:cl, ...], y[:cl, ...], overwrite_warning=True)
                    predictions = eval_model.predict(X_val).astype(np.int64)
                    outputs = eval_model.predict_proba(X_val)
                    #invert permutation of labels
                    new_output = np.zeros_like(outputs)
                    for i in range(num_classes):
                        new_output[:, invert_perm_map[i]] = outputs[:, i]
                    targets = y_val
                    # print("Predictions shape: ", predictions.shape)
                    # print("predictions numpy type: ", predictions.dtype)
                    # print("Predictions: ", predictions[:20, ...])
                    # print("Targets shape: ", targets.shape)
                    # print("Targets numpy type: ", targets.dtype)
                    # print("Targets: ", targets[:20, ...])
                    warnings.filterwarnings("ignore")
                    end_time = time.time()
                    results['Eval_Time'] = np.round(end_time - start_time, 3).item()
                    results['Accuracy'] = np.round(accuracy_score(targets, predictions), 3).item()
                    results['Log_Loss'] = np.round(log_loss(targets, outputs, labels=np.arange(num_classes)), 3).item()
                    results['F1_Weighted'] = np.round(f1_score(targets, predictions, average='weighted'), 3).item()
                    results['F1_Macro'] = np.round(f1_score(targets, predictions, average='macro'), 3).item()
                    try:
                        if num_classes == 2:
                            results['ROC_AUC'] = np.round(roc_auc_score(targets, outputs[:, 1], labels=np.arange(num_classes)), 3).item()
                        else:
                            results['ROC_AUC'] = np.round(roc_auc_score(targets, outputs, labels=np.arange(num_classes), multi_class='ovr'), 3).item()
                    except Exception as e:
                        print("Error calculating ROC AUC: ", e)
                        results['ROC_AUC'] = 0.0
                    results['ECE'] = np.round(um.ece(targets, outputs, num_bins=30), 3).item()
                    results['TACE'] = np.round(um.tace(targets, outputs, num_bins=30), 3).item()
                    warnings.filterwarnings("default")
                    return results
            res_dict = dict()
            val_results = tpc_data_eval(cl=real_data_qty, X=data_for_fitting[0], y=data_for_fitting[1], X_val=X_val, y_val=y_val, ens_size=extra_prior_kwargs_dict.get('zs_eval_ensemble', 0))
            res_dict = dict(res_dict, **{"Val_" + k : v for k, v in val_results.items()})
            test_results = tpc_data_eval(cl=real_data_qty, X=data_for_fitting[0], y=data_for_fitting[1], X_val=X_test, y_val=y_test, ens_size=extra_prior_kwargs_dict.get('zs_eval_ensemble', 0))
            res_dict = dict(res_dict, **{"Test_" + k : v for k, v in test_results.items()})
            print("Results: ", res_dict)
            with open(os.path.join(extra_prior_kwargs_dict.get('save_path'), 'zs_eval_ensemble.json'), 'w') as f:
                json.dump(res_dict, f)
            if extra_prior_kwargs_dict.get('wandb_log', False):
                import wandb
                wandb.log(res_dict, step=1, commit=True)
            exit(0)

    else:
        raise Exception("Excepted a real dataset")


    encoder = encoder_generator(num_features, emsize)
    #style_def = dl.get_test_batch()[0][0] # the style in batch of the form ((style, x, y), target, single_eval_pos)
    style_def = None
    #print(f'Style definition of first 3 examples: {style_def[:3] if style_def is not None else None}')
    style_encoder = style_encoder_generator(style_def.shape[1], emsize) if (style_def is not None) else None
    if isinstance(criterion, nn.GaussianNLLLoss):
        n_out = 2
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out = criterion.weight.shape[0]
    else:
        n_out = 1
    model = TransformerModel(encoder, n_out, emsize, nhead, nhid, nlayers, dropout, style_encoder=style_encoder,
                             y_encoder=y_encoder_generator(1, emsize), input_normalization=input_normalization,
                             pos_encoder=(pos_encoder_generator or positional_encodings.NoPositionalEncoding)(emsize, bptt*2),
                             decoder=decoder, init_method=initializer, efficient_eval_masking=efficient_eval_masking, prefix_size=prefix_size,
                             n_classes=num_classes, prefix_label_probs=label_weights, num_features=extra_prior_kwargs_dict.get("num_features", 100), **model_extra_args
                             )
    model.criterion = criterion    
    if load_weights_from_this_state_dict is not None:
        encoder_mismatch = False
        decoder_mismatch = False
        if num_classes > 10:
            #initialize a new decoder head
            decoder_mismatch = True
            # decoder_weight = load_weights_from_this_state_dict['decoder.2.weight'].cpu().numpy()
            # target_size = model.state_dict()['decoder.2.weight'].shape
            # # project decoder_weight to target_size by copying the first 10 classes
            # new_decoder_weight = np.zeros(target_size)
            # new_decoder_weight[:10, ...] = decoder_weight[:10, ...]
            # new_decoder_weight[10:, ...] = decoder_weight[0, ...]
            # load_weights_from_this_state_dict['decoder.2.weight'] = torch.from_numpy(new_decoder_weight).to(device)

            load_weights_from_this_state_dict['decoder.2.weight'] = model.state_dict()['decoder.2.weight']
            load_weights_from_this_state_dict['decoder.2.bias'] = model.state_dict()['decoder.2.bias']
            load_weights_from_this_state_dict['criterion.weight'] = model.state_dict()['criterion.weight']
        if load_weights_from_this_state_dict.get('prefix_embedding.weight', None) is None and model.state_dict().get('prefix_embedding.weight', None) is not None:
            load_weights_from_this_state_dict['prefix_embedding.weight'] = model.state_dict()['prefix_embedding.weight']
        if load_weights_from_this_state_dict.get('encoder.weight', None) is not None:
            load_shape = load_weights_from_this_state_dict.get('encoder.weight', None).shape
            model_shape = model.state_dict().get('encoder.weight', None).shape
            if load_shape != model_shape:
                encoder_mismatch = True
                print("Encoder weight shape mismatch: ", load_shape, model_shape, "Using randomly initialized encoder weights from model instead")
                load_weights_from_this_state_dict['encoder.weight'] = model.state_dict()['encoder.weight']
        model.load_state_dict(load_weights_from_this_state_dict)
    if initialize_with_model is not None:
        model.init_from_small_model(initialize_with_model)

    params_to_optimize = []
    if do_prompt_tuning:
        params_to_optimize.append("prefix_embedding")
    if encoder_mismatch:
        params_to_optimize.append("encoder")
    if decoder_mismatch:
        params_to_optimize.append("decoder.2")
        params_to_optimize.append("criterion")
    print("Params to optimize: ", params_to_optimize)

    print(f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters")

    try:
        for (k, v), (k2, v2) in zip(model.state_dict().items(), initialize_with_model.state_dict().items()):
            print(k, ((v - v2) / v).abs().mean(), v.shape)
    except Exception:
        pass

    model.to(device)
    if using_dist:
        print("Distributed training")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    
    if not real_prior:
        dl.model = model

    # learning rate
    if lr is None:
        lr = get_openai_lr(model)
        print(f"Using OpenAI max lr of {lr}.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched_obj = scheduler(optimizer, warmup_epochs, epochs if epochs is not None else 100) # when training for fixed time lr schedule takes 100 steps

    scaler = GradScaler() if train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    eval_model = None

    master_epoch_count = []
    
    def real_data_eval(r_model, cl=1000, train_data=None, val_dl=None, softmax_temperature = torch.log(torch.tensor([0.8]))):
        td = copy.deepcopy(train_data)
        #
        td[0] = td[0][:cl, ...]
        td[1] = td[1][:cl, ...]
        single_eval_pos = len(td[0])
        softmax_temperature = softmax_temperature.to(device)
        with torch.no_grad():
            # correct = 0
            # total = len(val_dl.dataset)
            prediction_list = []
            target_list = []
            output_list = []
            for batch, (data, targets, _) in enumerate(val_dl):
                batch_data = tuple([torch.cat((td[0], data[0]), dim=0), torch.cat((td[1], data[1]), dim=0)])
                output = r_model(tuple(e.to(device) if torch.is_tensor(e) else e for e in batch_data) if isinstance(batch_data, tuple) else batch_data.to(device)
                    , single_eval_pos=single_eval_pos)
                #invert permutation of labels
                _, predicted = torch.max(output.cpu().data, 1)
                # print("In real data eval: ")
                # print("Original preds: ", predicted[:20, ...])
                new_output = loop_translate(output, invert_perm_map)
                # new_output = torch.zeros_like(output)
                # for i in range(num_classes):
                #     new_output[:, invert_perm_map[i]] = output[:, i]
                output = new_output
                output = output[:, 0:num_classes] / torch.exp(softmax_temperature)
                output = torch.nn.functional.softmax(output, dim=-1)
                output_list.append(output)
                _, predicted = torch.max(output.cpu().data, 1)
                # print("New preds: ", predicted[:20, ...])
                # print("Targets: ", targets[:20, ...])
                prediction_list.append(predicted)
                target_list.append(targets)
            outputs = torch.cat(output_list, dim=0).cpu().numpy()
            predictions = torch.cat(prediction_list, dim=0).cpu().numpy()
            targets = torch.cat(target_list, dim=0).cpu().numpy()
            # print("Predictions shape: ", predictions.shape)
            # print("predictions numpy type: ", predictions.dtype)
            # print("Predictions: ", predictions[:20, ...])
            # print("Predictions: ", predictions[-20:, ...])
            # # print("Targets shape: ", targets.shape)
            # # print("Targets numpy type: ", targets.dtype)
            # print("Targets: ", targets[:20, ...])
            # print("Targets: ", targets[-20:, ...])
            # assert len(outputs) == len(predictions) == total, "Samples missing from eval: found {}, expected {}".format(len(outputs), total)
            # correct += (predictions == targets).sum().item()

        results = dict()
        warnings.filterwarnings("ignore")
        results['Accuracy'] = np.round(accuracy_score(targets, predictions), 3).item()
        results['Log_Loss'] = np.round(log_loss(targets, outputs, labels=np.arange(num_classes)), 3).item()
        results['F1_Weighted'] = np.round(f1_score(targets, predictions, average='weighted'), 3).item()
        results['F1_Macro'] = np.round(f1_score(targets, predictions, average='macro'), 3).item()
        try:
            if num_classes == 2:
                results['ROC_AUC'] = np.round(roc_auc_score(targets, outputs[:, 1], labels=np.arange(num_classes)), 3).item()
            else:
                results['ROC_AUC'] = np.round(roc_auc_score(targets, outputs, labels=np.arange(num_classes), multi_class='ovr'), 3).item()
        except Exception as e:
            print("Error calculating ROC AUC: ", e)
            results['ROC_AUC'] = 0.0
        results['ECE'] = np.round(um.ece(targets, outputs, num_bins=30), 3).item()
        results['TACE'] = np.round(um.tace(targets, outputs, num_bins=30), 3).item()
        # print("Targets shape: ", targets.shape)
        # print("Targets: ", targets[:20, ...])
        # print("Predictions shape: ", predictions.shape)
        # print("Predictions: ", predictions[:20, ...])
        # raise Exception("Stop here")
        warnings.filterwarnings("default")

        return results, outputs, targets
    
    def train_epoch(e_model, e_optimizer, boost_this_epoch=False):
        if max_time > 0 and time.time() - start_time > max_time:
            print("Max time reached. Exiting")
            exit(0)
        e_model.train()  # Turn on the train mode
        # Confirm that the correct params are frozen and unfrozen
        if do_prompt_tuning:
            e_model.freeze_parameters_except_named(params_to_optimize)
            for n, p in e_model.named_parameters():
                grad_reqd = False
                for s in params_to_optimize:
                    if s in n:
                        grad_reqd = True
                assert p.requires_grad == grad_reqd, "Parameter {} does not have the correct grad requirement!".format(n)

        total_loss = 0.
        total_positional_losses = 0.
        total_positional_losses_recorded = 0
        nan_steps = 0
        ignore_steps = 0
        time_to_get_batch = 0
        forward_time = 0
        step_time = 0
        before_get_batch = time.time()
        batches_seen = 0
        # assert len(dl) > aggregate_k_gradients, 'Number of batches per epoch must be greater than `aggregate_k_gradients`'
        # assert len(dl) % aggregate_k_gradients == 0, 'Please set the number of batches per epoch s.t. `aggregate_k_gradients` divides it.'
        # if do_prompt_tuning:
        #     wbu = e_model.prefix_embedding.weight.detach().clone()
        #     print("Prompt weights before: ", wbu[:10, ...])
            # print("Prompt requires grad: ", e_model.prefix_embedding.weight.requires_grad)
        shuffle_every_epoch = extra_prior_kwargs_dict.get('shuffle_every_epoch', False)
        permute_feature_pos = extra_prior_kwargs_dict.get('permute_feature_position_in_ensemble', False)
        for batch, (data, targets, single_eval_pos) in enumerate(dl):
            if isinstance(data, list):
                data = tuple(data)
            if isinstance(single_eval_pos, torch.Tensor) and single_eval_pos.numel() == 0:
                single_eval_pos = None
            if using_dist and not (batch % aggregate_k_gradients == aggregate_k_gradients - 1):
                print("using dist, No sync")
                cm = e_model.no_sync()
            else:
                cm = nullcontext()

            if permute_feature_pos:
                data = tuple([data[0][:, torch.randperm(data[0].shape[1])], data[1]])
            elif shuffle_every_epoch:
                seed_all(extra_prior_kwargs_dict.get('rand_seed', 0) + len(master_epoch_count))
                perm_idx = torch.randperm(data[0].shape[0])
                data = tuple([data[0][perm_idx, ...], data[1][perm_idx, ...]])
            with cm:
                time_to_get_batch = time.time() - before_get_batch
                before_forward = time.time()
                if boosting:
                    single_eval_pos = len(targets) // 2
                elif bptt_extra_samples is None:
                    single_eval_pos = single_eval_pos_gen() if callable(single_eval_pos_gen) else single_eval_pos_gen
                else:
                    single_eval_pos = targets.shape[0] - bptt_extra_samples
                # print("Size of data: ", len(data[0]))
                # print("Single eval pos: ", single_eval_pos)
                # print("BPTT extra samples: ", bptt_extra_samples)
                # print("Batch size: ", targets.shape[0])
                with autocast(enabled=scaler is not None):
                    # If style is set to None, it should not be transferred to device
                    output = e_model(tuple(e.to(device) if torch.is_tensor(e) else e for e in data) if isinstance(data, tuple) else data.to(device)
                                   , single_eval_pos=single_eval_pos)
                    assert output.requires_grad, "Output does not require gradients"
                    forward_time = time.time() - before_forward

                    if single_eval_pos is not None:
                        targets = targets[single_eval_pos:]
                    if isinstance(criterion, nn.GaussianNLLLoss):
                        assert output.shape[-1] == 2, \
                            'need to write a little bit of code to handle multiple regression targets at once'
                        mean_pred = output[..., 0]
                        var_pred = output[..., 1].abs()
                        losses = criterion(mean_pred.flatten(), targets.to(device).flatten(), var=var_pred.flatten())
                    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                        losses = criterion(output.flatten(), targets.to(device).flatten())
                    elif isinstance(criterion, nn.CrossEntropyLoss):
                        losses = criterion(output.reshape(-1, n_out), targets.to(device).long().flatten())
                    else:
                        losses = criterion(output, targets)
                    if boosting:
                        loss = losses.mean()
                        nan_share = torch.tensor([0])
                    else:
                        if len(output.shape) == 2:
                            output = output.unsqueeze(1)
                        # print("Losses shape: ", losses.shape)
                        # print("Outputs shape: ", output.shape)
                        losses = losses.view(*output.shape[0:2])

                        loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
                        loss = loss / aggregate_k_gradients

                if scaler: loss = scaler.scale(loss)
                if boosting and boost_this_epoch:
                    cur_grads = []
                    # Backward pass for each prediction/target pair
                    if prior_grad_dict is None:
                        prior_grad_iter = None
                    else:
                        prior_grad_iter = prior_grad_dict[batch].to(output.device)
                    output_grad = autograd.grad(loss, output)[0]
                    # print("Output grad shape: ", output_grad.shape)
                    gradient_dict[batch] = output_grad.detach().cpu().clone()
                    # cur_grads.append(output_grad.detach().cpu().clone())

                    if prior_grad_iter is not None:
                        grad_shape = output_grad.shape
                        flat_grad = output_grad.flatten()
                        grad_signs = torch.sign(flat_grad)
                        flat_prior_grad = prior_grad_iter.flatten()
                        cur_weight = 0.65
                        flat_grad_new = torch.sqrt(cur_weight * torch.pow(flat_grad, 2) + (1 - cur_weight) * torch.pow(flat_prior_grad, 2))
                        # ones = torch.ones_like(flat_grad)
                        # print("Flat grad shape: ", flat_grad.shape)
                        # print("Flat prior grad shape: ", flat_prior_grad.shape)
                        # flat_grad_new = torch.pow(flat_grad, ones + torch.log(torch.abs(flat_prior_grad)))
                        flat_grad_new_signs = torch.sign(flat_grad_new)
                        flat_grad_new[flat_grad_new_signs != grad_signs] *= -1
                        output_grad = flat_grad_new.reshape(grad_shape)

                    output.backward(output_grad)
                    # gradient_dict[batch] = torch.cat(cur_grads, dim=0)
                else:
                    loss.backward()             
                if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                    if scaler: scaler.unscale_(e_optimizer)
                    torch.nn.utils.clip_grad_norm_(e_model.parameters(), 1.)
                    try:
                        if scaler:
                            scaler.step(e_optimizer)
                            scaler.update()
                        else:
                            e_optimizer.step()
                    except:
                        print("Invalid optimization step encountered")
                    e_optimizer.zero_grad()

                step_time = time.time() - before_forward

                if not torch.isnan(loss):
                    total_loss += losses.mean().cpu().detach().item()
                    total_positional_losses += losses.mean(1).cpu().detach() if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)*\
                        losses[:bptt-single_eval_pos].mean().cpu().detach()

                    total_positional_losses_recorded += torch.ones(bptt) if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)
                nan_steps += nan_share
                ignore_steps += (targets == -100).float().mean()
            before_get_batch = time.time()
            batches_seen += 1
        #Total positional losses is a torch tensor of size bptt (batch size)
        if batches_seen < extra_prior_kwargs_dict.get('min_batches_per_epoch', 1):
            raise ValueError("Not enough batches seen in epoch: saw {} batches, expected at least {}".format(batches_seen, extra_prior_kwargs_dict.get('min_batches_per_epoch', 1)))

        if boosting:
            total_positional_losses = torch.zeros(bptt)
            total_positional_losses_recorded = torch.ones(bptt)
        if isinstance(total_positional_losses, float):
            total_positional_losses = torch.zeros(bptt)
        if isinstance(total_positional_losses_recorded, float):
            total_positional_losses_recorded = torch.ones(bptt)
        # if do_prompt_tuning:
        #     wbn = e_model.prefix_embedding.weight.detach().clone()
        #     print("Prompt weights after: ", wbn[:10, ...])
            # print("Prompt requires grad: ", e_model.prefix_embedding.weight.requires_grad)

        return total_loss / max(steps_per_epoch, 1), (total_positional_losses / total_positional_losses_recorded).tolist(),\
               time_to_get_batch, forward_time, step_time, nan_steps.cpu().item()/(batch+1),\
               ignore_steps.cpu().item()/(batch+1)

    def concat_embedding(ec, model, method):
        #extract embedding parameters
        device = ec.model.prefix_embedding.weight.device
        if method == "duplicate":
            ec.concatenated_embedding = torch.cat([ec.original_embedding, ec.original_embedding], dim=0).to(device)
            ec.concatenated_y_embedding = torch.cat([ec.original_y_embedding, ec.original_y_embedding], dim=0).to(device)
            ec.prefix_size = ec.original_prefix_size * 2
        elif method.startswith("rand-init"):
            num_to_concat = min(int(method.split("-")[-1]), len(ec.prefix_weights)+1)                
            print("Concatenating {} embeddings".format(num_to_concat))
            if num_to_concat == 1:
                ec.concatenated_embedding = ec.original_embedding
                ec.concatenated_y_embedding = ec.original_y_embedding
                ec.prefix_size = ec.original_prefix_size
            else:
                ec.concatenated_embedding = torch.cat([ec.original_embedding.to(device)] + [ec.prefix_weights[i]['prefix_weights'].to(device) for i in range(num_to_concat-1)], dim=0).to(device)
                ec.concatenated_y_embedding = torch.cat([ec.original_y_embedding.to(device)] + [ec.prefix_weights[i]['prefix_y_labels'].to(device) for i in range(num_to_concat-1)], dim=0).to(device)
                if "size-ctl" in method:
                    #select random sample of size prefix_size
                    if "perm" in method:
                        # random permutation
                        sel = torch.randperm(ec.concatenated_embedding.shape[0])[:ec.original_prefix_size].to(device)
                    else:
                        #first-k-samples
                        total_emb_size = ec.original_prefix_size
                        emb_size = total_emb_size // num_to_concat
                        orig_emb_size = ec.original_embedding.shape[0]
                        start_pos = [j * orig_emb_size for j in range(num_to_concat)]
                        sel = torch.cat([torch.arange(i, i+emb_size) for i in start_pos], dim=0).to(device)

                    ec.concatenated_embedding = ec.concatenated_embedding[sel]
                    ec.concatenated_y_embedding = ec.concatenated_y_embedding[sel]
                    ec.prefix_size = sel.shape[0]
                else:
                    ec.prefix_size = ec.original_prefix_size * num_to_concat
        else:
            raise NotImplementedError("Method {} not implemented!".format(method))
        model.prefix_embedding.weight = nn.Parameter(ec.concatenated_embedding)
        model.prefix_y_embedding = ec.concatenated_y_embedding
        model.prefix_size = ec.prefix_size
        return model

    def restore_embedding(ec, model):
        model.prefix_embedding.weight = nn.Parameter(ec.original_embedding)
        model.prefix_y_embedding = ec.original_y_embedding
        model.prefix_size = ec.original_prefix_size
        model.freeze_parameters_except_named(params_to_optimize)
        return model
    
    def save_prefix_weights(model, path, i, do_concat, prefix_weights_l):
        # Save prefix weights
        prefix_weights = model.state_dict()['prefix_embedding.weight'].cpu().numpy()
        prefix_fn = f"prefix_weights_{i}.npy"
        prefix_save_path = os.path.join(path, prefix_fn)
        np.save(prefix_save_path, prefix_weights)
        prefix_y_labels = model.prefix_y_embedding.cpu().numpy()
        prefix_y_fn = f"prefix_y_labels_{i}.npy"
        prefix_y_save_path = os.path.join(path, prefix_y_fn)
        np.save(prefix_y_save_path, prefix_y_labels)
        if do_concat:
            prefix_weights_l.append({"prefix_weights": torch.from_numpy(prefix_weights).float(), "prefix_y_labels": torch.from_numpy(prefix_y_labels)})
            # print("Prefix weights list length: ", len(prefix_weights_l))
        return prefix_weights_l

    def update_ensemble_acc(ens_acc, ens_acc_nc, ens_acc_test, ens_acc_test_nc):
        ece = np.round(um.ece(labels_np, probs_np, num_bins=30), 3)
        tace = np.round(um.tace(labels_np, probs_np, num_bins=30), 3)
        test_ece = np.round(um.ece(labels_np_test, probs_np_test, num_bins=30), 3)
        test_tace = np.round(um.tace(labels_np_test, probs_np_test, num_bins=30), 3)
        if do_prompt_tuning:
            nc_ece = np.round(um.ece(labels_np_nc, probs_np_nc, num_bins=30), 3)
            nc_tace = np.round(um.tace(labels_np_nc, probs_np_nc, num_bins=30), 3)
            nc_test_ece = np.round(um.ece(labels_np_nc_test, probs_np_nc_test, num_bins=30), 3)
            nc_test_tace = np.round(um.tace(labels_np_nc_test, probs_np_nc_test, num_bins=30), 3)
        else:
            nc_ece = 0
            nc_tace = 0
            nc_test_ece = 0
            nc_test_tace = 0
        new_res = {
            "Ens_Val_Accuracy": ens_acc,
            "Ens_Val_Accuracy_NC": ens_acc_nc,
            "Ens_Val_ECE": ece,
            "Ens_Val_TACE": tace,
            "Ens_Val_ECE_NC": nc_ece,
            "Ens_Val_TACE_NC": nc_tace,
            "Ens_Test_Accuracy": ens_acc_test,
            "Ens_Test_Accuracy_NC": ens_acc_test_nc,
            "Ens_Test_ECE": test_ece,
            "Ens_Test_TACE": test_tace,
            "Ens_Test_ECE_NC": nc_test_ece,
            "Ens_Test_TACE_NC": nc_test_tace
        }
        return new_res

    def train_test_loop(t_model, t_optim, t_sched, dl, val_dl, test_dl):        
        # Select a fixed training data prior of size bptt
        return_outputs = None
        return_targets = None
        res_dict = None
        best_val_score = best_val_score_nc = 0
        if do_prompt_tuning:
            best_val_embed = t_model.prefix_embedding.weight.detach().cpu()
        best_res_dict = None
        best_outputs = None
        best_targets = None
        is_best = False
        patience = 0

        for epoch in (range(1, epochs + 1) if epochs is not None else itertools.count(1)):
            is_best = False
            print('epoch', epoch, 'of', epochs)
            boost_this_epoch = True if epoch == 1 else False
            epoch_start_time = time.time()
            master_epoch_count.append(1)
            total_loss, total_positional_losses, time_to_get_batch, forward_time, step_time, nan_share, ignore_share =\
                train_epoch(t_model, t_optim, boost_this_epoch)
            val_score = val_score_nc = val_score_concat = val_score_nc_concat = test_score = test_score_nc = test_ece = test_tace = val_ece = val_tace = val_ece_nc = val_tace_nc = test_ece_nc = test_tace_nc = None
            res_dict = dict()
            if real_prior \
                and (epoch - 1) % validation_period == 0:
                val_results, val_outputs, val_targets = real_data_eval(r_model=t_model, cl=real_data_qty, train_data=data_for_fitting, val_dl=val_dl)
                res_dict = dict(res_dict, **{"Val_" + k : v for k, v in val_results.items()})
                val_score = res_dict["Val_Accuracy"]
                # val_score2, val_outputs2, val_targets2 = real_data_eval(r_model=t_model, cl=bptt, train_data=data_for_fitting, val_dl=val_dl)
                # if val_score != val_score2:
                #     print("Val score mismatch: ", val_score, val_score2)
                #     raise ValueError("Val score mismatch")

                test_results, test_outputs, test_targets = real_data_eval(r_model=t_model, cl=real_data_qty, train_data=data_for_fitting, val_dl=test_dl)
                res_dict = dict(res_dict, **{"Test_" + k : v for k, v in test_results.items()})
                return_outputs = [val_outputs, test_outputs]
                return_targets = [val_targets, test_targets]
                if do_prompt_tuning:
                    #TODO: will this work with context length 0? Should this be a hyperparameter?
                    if do_concat != "":
                        ec = EmbeddingConcatenator(t_model, do_concat, prefix_weights_l)
                        t_model = concat_embedding(ec, t_model, do_concat)
                        val_score_concat, _, _ = real_data_eval(r_model=ec.get_model(), cl=real_data_qty, train_data=data_for_fitting, val_dl=val_dl)
                        res_dict = dict(res_dict, **{"Val_concat_" + k : v for k, v in val_score_concat.items()})
                        val_score_nc_concat, _, _ = real_data_eval(r_model=ec.get_model(), cl=0, train_data=data_for_fitting, val_dl=val_dl)
                        res_dict = dict(res_dict, **{"Val_concat_nc_" + k : v for k, v in val_score_nc_concat.items()})
                        t_model = restore_embedding(ec, t_model)
                        # Update optimizer parameters to include new embedding
                        t_optim = torch.optim.AdamW(t_model.parameters(), lr=lr, weight_decay=weight_decay)
                        t_sched = scheduler(t_optim, warmup_epochs, epochs if epochs is not None else 100)
                    else:
                        val_score_nc_concat = ""
                        val_score_concat = ""
                    val_score_nc, val_outputs, val_targets = real_data_eval(r_model=t_model, cl=0, train_data=data_for_fitting, val_dl=val_dl)
                    return_outputs.append(val_outputs)
                    return_targets.append(val_targets)
                    res_dict = dict(res_dict, **{"Val_nc_" + k : v for k, v in val_score_nc.items()})
                    test_score_nc, test_outputs, test_targets = real_data_eval(r_model=t_model, cl=0, train_data=data_for_fitting, val_dl=test_dl)
                    return_outputs.append(test_outputs)
                    return_targets.append(test_targets)
                    res_dict = dict(res_dict, **{"Test_nc_" + k : v for k, v in test_score_nc.items()})
                if not extra_prior_kwargs_dict.get('uniform_bptt', False) \
                    and val_score and val_score > best_val_score:
                    #print("Train data:", data_for_fitting)
                    patience = 0
                    best_val_score = val_score
                    is_best = True
                    if do_prompt_tuning:
                        best_val_embed = t_model.prefix_embedding.weight.detach().cpu()
                elif extra_prior_kwargs_dict.get('uniform_bptt', False) and bptt <= 128 \
                    and do_prompt_tuning and \
                    res_dict.get("Val_nc_Accuracy", 0) > best_val_score_nc:
                    patience = 0
                    best_val_score_nc = res_dict.get("Val_nc_Accuracy", 0)
                    is_best = True
                    best_val_embed = t_model.prefix_embedding.weight.detach().cpu()
                else:
                    patience += 1
            elif hasattr(dl, 'validate') and epoch % validation_period == 0:
                with torch.no_grad():
                    val_score = dl.validate(model)

            if patience > extra_prior_kwargs_dict.get('early_stopping_patience', 2):
                print("Early stopping after {} epochs".format(epoch))
                break

            if verbose:
                get_time = (time.time() - epoch_start_time)
                print('-' * 89)
                print(
                    f'| end of epoch {epoch:3d} | time: {get_time:5.2f}s | mean loss {total_loss:5.2f} | '
                    #f"| pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])} | lr {scheduler.get_last_lr()[0]}"
                    f' | data time {time_to_get_batch:5.2f} | step time {step_time:5.2f}'
                    f' | forward time {forward_time:5.2f}' 
                    f' | nan share {nan_share:5.2f} | ignore share (for classification tasks) {ignore_share:5.4f}'
                    + (f' | val score {val_score}' if val_score is not None else '')
                    + (f' | val score nc {res_dict.get("Val_nc_Accuracy", 0)}' if val_score_nc is not None else '')
                )
                print('-' * 89)
                if epoch_callback is not None and rank == 0:
                    # print("Calling epoch callback")
                    # print("res dict is now")
                    # print(res_dict)
                    epoch_callback(model, epoch / epochs, res_dict)
                if val_score is not None:
                    # save the log to a json file
                    res_dict = dict(res_dict, **{'time' : get_time, 
                                'epoch': epoch, 
                    })
                    if extra_prior_kwargs_dict.get('wandb_log', False):
                        import wandb
                        wandb.log(res_dict, step=len(master_epoch_count), commit=True)
                    if is_best:
                        best_res_dict = res_dict
                        best_outputs = return_outputs
                        best_targets = return_targets
                    mstr = extra_prior_kwargs_dict.get('model_string')
                    boost_iter = f"ensemble_iter_{cur_boost_iter}" if is_ensemble else ""
                    log_path = os.path.join(extra_prior_kwargs_dict.get('save_path'), f'{mstr}_{boost_iter}_log_{epoch}.json')
                    with open(log_path, 'w') as f:
                        json.dump(res_dict, f, indent=4)

            # stepping with wallclock time based scheduler
            t_sched.step()

        if do_prompt_tuning and isinstance(best_val_embed, torch.Tensor):
            t_model.prefix_embedding.weight = nn.Parameter(best_val_embed.to(device))
            #set requires grad to true
            t_model.prefix_embedding.weight.requires_grad = True
            t_optim = torch.optim.AdamW(t_model.parameters(), lr=lr, weight_decay=weight_decay)
            t_sched = scheduler(t_optim, warmup_epochs, epochs if epochs is not None else 100)
            v_scr, val_outputs, val_targets = real_data_eval(r_model=t_model, cl=real_data_qty, train_data=data_for_fitting, val_dl=val_dl)
            if v_scr['Accuracy'] != best_res_dict['Val_Accuracy']:
                print("Best embedding score {} does not match best score {}!".format(v_scr, best_res_dict['Val_Accuracy']))

        return best_outputs, best_targets, best_res_dict

    # main training loop
    bagging = extra_prior_kwargs_dict.get("bagging", False)
    if bagging:
        dl_backup = dl
        split_size = 0.5
        split_indices = []
        for i in range(boosting_n_iters):
            np.random.seed(extra_prior_kwargs_dict.get('rand_seed') + i)
            split_indices.append(np.random.choice(np.arange(len(dl_backup.dataset)), size=int(split_size * len(dl_backup.dataset)), replace=False))
        # dl_backup = dl
        # split_indices = np.array_split(np.arange(len(dl_backup.dataset)), boosting_n_iters)
    is_ensemble = (boosting or bagging or rand_init_ensemble)
    prefix_weights_l = []
    cur_boost_iter = 0
    total_loss = float('inf')
    total_positional_losses = float('inf')
    output_dict = {}
    i = 0
    ensembling_acc = dict()
    res_dict_ensemble = dict()
    try:
        print("Starting training loop \n \n")
        if bagging:
            subset_dataset = Subset(dl_backup.dataset, split_indices[i])
            dl, bptt = get_train_dataloader(subset_dataset, 
                                            bptt=bptt, 
                                            shuffle=True, 
                                            num_workers=1, 
                                            drop_last=True, 
                                            agg_k_grads=aggregate_k_gradients)
        prior_grad_dict = None
        gradient_dict = {}
        output_dict[i], test_targets, results_dict = train_test_loop(model, optimizer, sched_obj, dl, val_dl, test_dl)
        res_dict_ensemble[i] = results_dict
        prior_grad_dict = gradient_dict
        # probs np and labels np are used by update_ensemble_acc for ECE and TACE
        #OUTPUT_DICT[0] contains val_outputs, test_outputs, val_outputs_nc, test_outputs_nc
        probs_np = output_dict[0][0]
        labels_np = test_targets[0]
        probs_np_test = output_dict[0][1]
        labels_np_test = test_targets[1]
        if do_prompt_tuning:
            probs_np_nc = output_dict[0][2]
            labels_np_nc = test_targets[2]
            probs_np_nc_test = output_dict[0][3]
            labels_np_nc_test = test_targets[3]
        if is_ensemble:
            master_epoch_count.append(1)
            ensembling_acc[i] = update_ensemble_acc(res_dict_ensemble[i]['Val_Accuracy'], 
                                                    res_dict_ensemble[i]['Val_nc_Accuracy'], 
                                                    res_dict_ensemble[i]['Test_Accuracy'], 
                                                    res_dict_ensemble[i]['Test_nc_Accuracy'])
            if not do_concat:
                with open(os.path.join(extra_prior_kwargs_dict.get('save_path'), 'ensembling_acc.json'), 'w') as f:
                    json.dump(ensembling_acc, f, indent=4)
                if extra_prior_kwargs_dict.get('wandb_log', False):
                    import wandb
                    wandb.log(ensembling_acc[i], step=len(master_epoch_count), commit=True)
        if do_prompt_tuning:
            prefix_weights_l = save_prefix_weights(model, extra_prior_kwargs_dict.get('save_path'), i, do_concat, prefix_weights_l)
    except KeyboardInterrupt:
        pass

    # boosting logic
    if is_ensemble:
        for i in range(1, boosting_n_iters):
            master_epoch_count.append(1)
            seed_all(extra_prior_kwargs_dict.get('rand_seed') + i)
            if extra_prior_kwargs_dict.get('reseed_data', True):
                #load data
                extra_prior_kwargs_dict['preprocess_type'] = np.random.choice(['none', 'power_all', 'robust_all', 'quantile_all'])
                X, y, X_val, y_val, X_test, y_test, invert_perm_map, steps_per_epoch, num_classes, label_weights, train_ds, val_ds, test_ds = make_datasets(extra_prior_kwargs_dict, do_permute=not_zs, bptt=bptt, steps_per_epoch=steps_per_epoch)
                old_bptt = bptt
                dl, val_dl, test_dl, bptt, data_for_fitting  = make_dataloaders(bptt=bptt)
                if old_bptt != bptt:
                    print("bptt changed from {} to {}".format(old_bptt, bptt))
                    if extra_prior_kwargs_dict.get('uniform_bptt', False): 
                        single_eval_pos_gen = lambda: np.random.randint(0, bptt)
                    else:
                        single_eval_pos_gen = bptt
                if bagging:
                    dl_backup = dl
            if bagging:
                subset_dataset = Subset(dl_backup.dataset, split_indices[i])
                dl = DataLoader(
                    subset_dataset, batch_size=bptt, shuffle=False, num_workers=1, drop_last=True,
                )
            cur_boost_iter = i
            print("Ensembling iteration: ", i+1, " of ", boosting_n_iters, "\n \n")
            model.init_prefix_weights()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            sched_obj = scheduler(optimizer, warmup_epochs, epochs if epochs is not None else 100)
            output_dict[i], test_targets, results_dict = train_test_loop(model, optimizer, sched_obj, dl, val_dl, test_dl)
            res_dict_ensemble[i] = results_dict
            if do_prompt_tuning:
                prefix_weights_l = save_prefix_weights(model, extra_prior_kwargs_dict.get('save_path'), i, do_concat, prefix_weights_l)
            prior_grad_dict = gradient_dict

            #No need to save ensembled results if we are concatenating; regular results are accurate
            if do_concat != "":
                 continue

            current_outs = dict()
            current_preds = dict()
            boosting_accs = dict()
            topk_ens_val = extra_prior_kwargs_dict.get('keep_topk_ensemble', 0)
            topk_key = extra_prior_kwargs_dict.get('topk_key', 'Val_Accuracy')
            if topk_ens_val > 0:
                print("keeping top {} of {} models, per provided key {}".format(topk_ens_val, i+1, topk_key))
                #sort by val score
                sorted_res = sorted(res_dict_ensemble.items(), key=lambda x: x[1][topk_key], reverse=True)
                models_to_include = [x[0] for x in sorted_res][:topk_ens_val]
            else:
                models_to_include = list(range(i + 1))
            # Evaluate average model on all available benchmarks
            for m in range(len(output_dict[0])):
                total = len(test_targets[m])
                if extra_prior_kwargs_dict.get('average_ensemble'):
                        current_outs[m] = torch.zeros_like(torch.from_numpy(output_dict[0][m]))
                        for j in range(i + 1):
                            if j not in models_to_include:
                                continue
                            current_outs[m] += output_dict[j][m]
                        current_outs[m] /= (i + 1)
                # Evaluate additive model
                else:
                    current_outs[m] = output_dict[0][m]
                    for j in range(1, i + 1):
                        if j not in models_to_include:
                            continue
                        boost_res = torch.mul(boosting_lr, torch.from_numpy(output_dict[j][m]))
                        current_outs[m] += boost_res
                _, current_preds[m] = torch.max(current_outs[m].cpu().data, 1)
                correct = (current_preds[m] == torch.from_numpy(test_targets[m])).sum().item()
                boosting_accs[m] = np.round(correct / total, 3)
            #TODO: this should not be hard-coded
            #OUTPUT_DICT[0] contains val_outputs, test_outputs, val_outputs_nc, test_outputs_nc
            probs_np = output_dict[0][0]
            labels_np = test_targets[0]
            probs_np_test = output_dict[0][1]
            labels_np_test = test_targets[1]
            if do_prompt_tuning:
                probs_np_nc = output_dict[0][2]
                labels_np_nc = test_targets[2]
                probs_np_nc_test = output_dict[0][3]
                labels_np_nc_test = test_targets[3]
            ensembling_acc[i] = update_ensemble_acc(boosting_accs[0], boosting_accs[2], boosting_accs[1], boosting_accs[3])
            if do_prompt_tuning:
                prefix_weights_l = save_prefix_weights(model, extra_prior_kwargs_dict.get('save_path'), i, do_concat, prefix_weights_l)
            # Save ensembled accuracy
            with open(os.path.join(extra_prior_kwargs_dict.get('save_path'), 'ensembling_acc.json'), 'w') as f:
                json.dump(ensembling_acc, f, indent=4)
            if extra_prior_kwargs_dict.get('wandb_log', False):
                import wandb
                wandb.log(ensembling_acc[i], step=len(master_epoch_count), commit=True)

    # break down training and return
    if rank == 0: # trivially true for non-parallel training
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            dl = None
        # todo: model_builder.py expects two outputs: model, results_dict
        #return total_loss, total_positional_losses, model.to('cpu'), dl
        return model, results_dict

    return model, results_dict

def _parse_args(config_parser, parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


if __name__ == '__main__':
    config_parser = argparse.ArgumentParser(description='Only used as a first parser for the config file path.')
    config_parser.add_argument('--config')
    parser = argparse.ArgumentParser()
    parser.add_argument('prior')
    parser.add_argument('--loss_function', default='gaussnll')
    # Optional Arg's for `--loss_function barnll`
    parser.add_argument('--min_y', type=float, help='barnll can only model y in strict ranges, this is the minimum y can take.')
    parser.add_argument('--max_y', type=float, help='barnll can only model y in strict ranges, this is the maximum y can take.')
    parser.add_argument('--num_features', default=None, type=int, help='Specify depending on the prior (can be None).')
    #parser.add_argument('--num_features', default=None, type=int, help='Specify depending on the prior.')
    parser.add_argument("--extra_prior_kwargs_dict", default={}, dest="extra_prior_kwargs_dict", action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL", help='Specify depending on the prior.')
    parser.add_argument('--encoder', default='linear', type=str, help='Specify depending on the prior.')
    parser.add_argument('--y_encoder', default='linear', type=str, help='Specify depending on the prior. You should specify this if you do not fuse x and y.')
    parser.add_argument('--pos_encoder', default='none', type=str, help='Specify depending on the prior.')
    parser.add_argument('--bptt', default=10, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', default=50, type=int)
    parser.add_argument('--validation_period', default=10, type=int)
    parser.add_argument('--permutation_invariant_max_eval_pos', default=None, type=int, help='Set this to an int to ')
    parser.add_argument('--permutation_invariant_sampling', default='weighted', help="Only relevant if --permutation_invariant_max_eval_pos is set.")
    parser.add_argument('--train_mixed_precision', action='store_true')

    # these can likely be mostly left at defaults
    parser.add_argument('--emsize', default=512, type=int) # sometimes even larger is better e.g. 1024
    parser.add_argument('--nlayers', default=6, type=int)
    parser.add_argument('--nhid', default=None, type=int) # 2*emsize is the default
    parser.add_argument('--nhead', default=4, type=int) # nhead = emsize / 64 in the original paper
    parser.add_argument('--dropout', default=.0, type=float)
    parser.add_argument('--steps_per_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--lr', '--learning_rate', default=.001, type=float) # try also .0003, .0001, go lower with lower batch size

    args, _ = _parse_args(config_parser, parser)

    if args.nhid is None:
        args.nhid = 2*args.emsize

    prior = args.__dict__.pop('prior')

    if prior == 'gp':
        prior = priors.fast_gp.DataLoader
    elif prior == 'ridge':
        prior = priors.ridge.DataLoader
    elif prior == 'stroke':
        prior = priors.stroke.DataLoader
    elif prior == 'mix_gp':
        prior = priors.fast_gp_mix.DataLoader
    else:
        raise NotImplementedError(f'Prior == {prior}.')

    loss_function = args.__dict__.pop('loss_function')

    criterion = nn.GaussianNLLLoss(reduction='none', full=True)
    classificiation_criterion = nn.CrossEntropyLoss(reduction='none')
    max_y = args.__dict__.pop('max_y')
    min_y = args.__dict__.pop('min_y')
    # criterion = nn.MSELoss(reduction='none')

    if loss_function == 'ce':
        criterion = nn.CrossEntropyLoss(reduction='none')
    elif loss_function == 'gaussnll':
        criterion = nn.GaussianNLLLoss(reduction='none', full=True)
    elif loss_function == 'mse':
        criterion = nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError(f'loss_function == {loss_function}.')



    encoder = args.__dict__.pop('encoder')
    y_encoder = args.__dict__.pop('y_encoder')

    def get_encoder_generator(encoder):
        if encoder == 'linear':
            encoder_generator = encoders.Linear
        elif encoder == 'mlp':
            encoder_generator = encoders.MLP
        elif encoder == 'positional':
            encoder_generator = encoders.Positional
        else:
            raise NotImplementedError(f'A {encoder} encoder is not valid.')
        return encoder_generator

    encoder_generator = get_encoder_generator(encoder)
    y_encoder_generator = get_encoder_generator(y_encoder)

    pos_encoder = args.__dict__.pop('pos_encoder')

    if pos_encoder == 'none':
        pos_encoder_generator = None
    elif pos_encoder == 'sinus':
        pos_encoder_generator = positional_encodings.PositionalEncoding
    elif pos_encoder == 'learned':
        pos_encoder_generator = positional_encodings.LearnedPositionalEncoding
    elif pos_encoder == 'paired_scrambled_learned':
        pos_encoder_generator = positional_encodings.PairedScrambledPositionalEncodings
    else:
        raise NotImplementedError(f'pos_encoer == {pos_encoder} is not valid.')

    permutation_invariant_max_eval_pos = args.__dict__.pop('permutation_invariant_max_eval_pos')
    permutation_invariant_sampling = args.__dict__.pop('permutation_invariant_sampling')
    if permutation_invariant_max_eval_pos is not None:
        if permutation_invariant_sampling == 'weighted':
            get_sampler = get_weighted_single_eval_pos_sampler
        elif permutation_invariant_sampling == 'uniform':
            get_sampler = get_uniform_single_eval_pos_sampler
        else:
            raise ValueError()
        args.__dict__['single_eval_pos_gen'] = get_sampler(permutation_invariant_max_eval_pos)


    print("ARGS for `train`:", args.__dict__)

    train(prior, criterion, encoder_generator,
          y_encoder_generator=y_encoder_generator, pos_encoder_generator=pos_encoder_generator,
          **args.__dict__)
