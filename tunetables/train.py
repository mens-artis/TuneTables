import argparse
import copy
import itertools
import json
import logging
import os
import time
import warnings
from contextlib import nullcontext

import numpy as np
import torch
import yaml
# import uncertainty_metrics.numpy as um
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from torch import autograd, Tensor
from torch import nn
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
try:
    import opacus
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.utils.batch_memory_manager import BatchMemoryManager
except ImportError as error:
    PrivacyEngine = ModuleValidator = BatchMemoryManager = None
    print("Could not import opacus (pip install opacus) for differential privacy")

import tunetables.utils as utils
from tunetables.transformer import TransformerModel
from tunetables.utils import (get_cosine_schedule_with_warmup, get_openai_lr, StoreDictKeyPair,
                              get_weighted_single_eval_pos_sampler, get_uniform_single_eval_pos_sampler,
                              load_and_combine_attention_weights)
import tunetables.priors as priors
from tunetables.priors.real import (summarize_after, process_data, loop_translate, TabDS, preprocess_input,
                                    get_train_dataloader, get_shuffle_index, get_subset_dl)
from tunetables.losses import kl_divergence
import tunetables.encoders as encoders
import tunetables.positional_encodings as positional_encodings
from tunetables.utils import init_dist, seed_all, EmbeddingConcatenator


def safe_round(x, decimals=3):
    rounded = np.round(np.atleast_1d(x), decimals=decimals)
    return rounded[0] if rounded.size == 1 else rounded


class GPULossTracker:
    """Tracks average loss across batches while keeping everything on GPU until final computation."""

    def __init__(self, device=None):
        self.running_loss = torch.tensor(0.0, device=device)
        self.count = 0

    def update(self, loss: torch.Tensor) -> None:
        """Update running loss with the current batch loss."""
        # Just detach without moving to CPU
        self.running_loss += loss.detach()
        self.count += 1

    def average(self) -> float:
        """Get the average loss, only moving to CPU at the end."""
        avg = self.running_loss / max(self.count, 1)
        return avg.cpu().item()

    def reset(self) -> None:
        """Reset the tracker for the next epoch."""
        self.running_loss.zero_()
        self.count = 0


def real_data_eval_out(r_model, cl=1000, train_data=None, val_dl=None,
                       softmax_temperature=torch.log(torch.tensor([0.8])), return_probs=False):
    verbose = False
    start_time = time.time()
    td = copy.deepcopy(train_data)
    num_classes_local = len(torch.unique(td[1]))
    td[0] = td[0][:cl, ...]
    td[1] = td[1][:cl, ...]
    single_eval_pos = len(td[0])
    device = next(r_model.parameters()).device
    softmax_temperature = softmax_temperature.to(device)
    with torch.inference_mode():
        # correct = 0
        # total = len(val_dl.dataset)
        prediction_list = []
        target_list = []
        output_list = []
        for batch, (data, targets, _) in enumerate(val_dl):
            # extra safeguard against learning from test set
            data_temp_idx = torch.randperm(data[1].nelement())
            data[1] = data[1].view(-1)[data_temp_idx].view(data[1].size())

            batch_data = tuple([torch.cat((td[0], data[0]), dim=0).to(torch.float32),
                                torch.cat((td[1], data[1]), dim=0).to(torch.float32)])
            output = r_model(
                tuple(e.to(device) if torch.is_tensor(e) else
                      e for e in batch_data) if isinstance(batch_data, tuple) else
                batch_data.to(device),
                single_eval_pos=single_eval_pos)
            output = output[:, 0:num_classes_local] / torch.exp(softmax_temperature)
            output = torch.nn.functional.softmax(output, dim=-1)
            output_list.append(output)
            _, predicted = torch.max(output.cpu().data, 1)
            prediction_list.append(predicted)
            target_list.append(targets)
        outputs = torch.cat(output_list, dim=0).cpu().numpy()
        predictions = torch.cat(prediction_list, dim=0).cpu().numpy()
        targets = torch.cat(target_list, dim=0).cpu().numpy()

    results = dict()
    warnings.filterwarnings("ignore")
    results['Eval_Time'] = safe_round(time.time() - start_time)
    results['Accuracy'] = safe_round(accuracy_score(targets, predictions))
    try:
        results['Log_Loss'] = safe_round(log_loss(targets, outputs, labels=np.arange(num_classes_local)))
    except Exception as e:
        if verbose:
            print("Error calculating log loss: ", e)
        results['Log_Loss'] = 0.0
    results['F1_Weighted'] = safe_round(f1_score(targets, predictions, average='weighted'))
    results['F1_Macro'] = safe_round(f1_score(targets, predictions, average='macro'))
    try:
        if num_classes_local == 2:
            results['ROC_AUC'] = safe_round(roc_auc_score(targets, outputs[:, 1], labels=np.arange(num_classes_local)))
        else:
            results['ROC_AUC'] = safe_round(
                roc_auc_score(targets, outputs, labels=np.arange(num_classes_local), multi_class='ovr'))
    except Exception as e:
        if verbose:
            print("Error calculating ROC AUC: ", e)
        results['ROC_AUC'] = 0.0

    warnings.filterwarnings("default")
    if return_probs:
        return results, outputs, targets
    else:
        return results, predictions, targets


def train(train_args, dataset, training_criterion, _encoder_generator, emsize=200, nhid=200, nlayers=6, nhead=2,
          dropout=0.0, epochs=10, steps_per_epoch=100, batch_size=200, bptt=10, lr=None, weight_decay=0.0,
          warmup_epochs=10, input_normalization=False,
          _y_encoder_generator=None, _pos_encoder_generator=None, decoder=None, extra_prior_kwargs_dict=None,
          scheduler=get_cosine_schedule_with_warmup,
          load_weights_from_this_state_dict=None, validation_period=10, single_eval_pos_gen=None,
          bptt_extra_samples=None, gpu_device='cuda:0',
          aggregate_k_gradients=1, verbose=False, style_encoder_generator=None, epoch_callback=None,
          initializer=None, initialize_with_model=None, train_mixed_precision=False, efficient_eval_masking=True,
          boosting=False, boosting_lr=1e-3, boosting_n_iters=10, rand_init_ensemble=False, do_concat="",
          is_wrapper=False, x_wrapper=None, y_wrapper=None,
          **model_extra_args
          ):
    # ulimit error fix
    if extra_prior_kwargs_dict is None:
        extra_prior_kwargs_dict = {}
    torch.multiprocessing.set_sharing_strategy('file_system')
    # fork warning fix
    torch.multiprocessing.set_start_method('spawn')
    # set gpu device
    device = gpu_device if torch.cuda.is_available() else 'cpu:0'
    using_dist, rank, device = init_dist(device)
    start_time = time.time()

    # FLAGS

    # set verbose to True
    if not verbose:
        verbose = True
        print("Currently, verbose must be set to True (pass --verbose); this will change in a future release")

    # verify that the save path exists
    if not os.path.exists(extra_prior_kwargs_dict.get('save_path')):
        try:
            os.makedirs(extra_prior_kwargs_dict.get('save_path'))
        except Exception as e:
            print("Error creating save path: ", e)
            print("Using current directory instead")
            extra_prior_kwargs_dict['save_path'] = os.getcwd()

    max_time = extra_prior_kwargs_dict.get('max_time', 0)
    do_kl_loss = extra_prior_kwargs_dict.get('kl_loss', False)
    do_private = extra_prior_kwargs_dict.get('private_model', False)
    private_data = False
    if extra_prior_kwargs_dict.get('private_data', False):
        private_data = True
        do_private = False
    n_workers = extra_prior_kwargs_dict.get('num_workers', 1)
    # TODO: main (3 lines)
    extra_prior_kwargs_dict['do_impute'] = True
    extra_prior_kwargs_dict['ohe'] = False
    linear = extra_prior_kwargs_dict.get('linear', False)
    # TODO: apacus (2 lines)
    not_zs = extra_prior_kwargs_dict.get('zs_eval_ensemble', 0) == 0
    do_zs = (not not_zs) and (not do_kl_loss)

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

    def make_datasets(_extra_prior_kwargs_dict, do_permute=True, _bptt=0, _steps_per_epoch=None, is_wrapper=False,
                      private_ds=False, do_private=private_data):

        train_args.summerize_after_prep = _extra_prior_kwargs_dict.get("summerize_after_prep", "False")
        train_args.preprocess_type = _extra_prior_kwargs_dict.get("preprocess_type", "none")
        train_args.rand_seed = _extra_prior_kwargs_dict.get('rand_seed', 0)
        train_args.private_data = private_ds
        train_args.epsilon = str(_extra_prior_kwargs_dict.get('epsilon', 0.0))
        train_args.private_val = _extra_prior_kwargs_dict.get('private_val', False)

        train_index = val_index = test_index = None
        if is_wrapper:
            train_index = dataset.split_indeces[0]
            val_index = dataset.split_indeces[1]
            test_index = dataset.split_indeces[1]
        else:
            for _i, split_dictionary in enumerate(dataset.split_indeces):
                # TODO: make stopping index a hyperparameter
                if _i != _extra_prior_kwargs_dict.get('split'):  # config['split']:
                    continue
                train_index = split_dictionary["train"]
                val_index = split_dictionary["val"]
                test_index = split_dictionary["test"]

        if True:
            # run pre-processing & split data (list of numpy arrays of length num_ensembles)
            processed_data = process_data(
                dataset,
                train_index,
                val_index,
                test_index,
                verbose=_extra_prior_kwargs_dict.get('verbose'),  # config['verbose'],
                scaler="None",
                one_hot_encode=_extra_prior_kwargs_dict.get('ohe', True),
                impute=_extra_prior_kwargs_dict.get('do_impute', True),
                args=train_args,
            )
            x_train, y_train = processed_data["data_train"]
            _x_val, _y_val = processed_data["data_val"]
            _x_test, _y_test = processed_data["data_test"]

            if is_wrapper:
                _extra_prior_kwargs_dict['shuffle_index'] = {
                    'train': np.arange(0, len(x_train)),
                    'val': np.arange(0, len(_x_val)),
                    'test': np.arange(0, len(_x_test)),
                }

            if _extra_prior_kwargs_dict.get("shuffle_index", None) is None:
                _extra_prior_kwargs_dict['shuffle_index'] = {
                    'train': get_shuffle_index(x_train),
                    'val': get_shuffle_index(_x_val),
                    'test': get_shuffle_index(_x_test),
                }

            x_train = x_train[_extra_prior_kwargs_dict['shuffle_index']['train']]
            y_train = y_train[_extra_prior_kwargs_dict['shuffle_index']['train']]
            _x_val = _x_val[_extra_prior_kwargs_dict['shuffle_index']['val']]
            _y_val = _y_val[_extra_prior_kwargs_dict['shuffle_index']['val']]
            _x_test = _x_test[_extra_prior_kwargs_dict['shuffle_index']['test']]
            _y_test = _y_test[_extra_prior_kwargs_dict['shuffle_index']['test']]

            n_features = x_train.shape[1]
            n_samples = x_train.shape[0]
            # config['num_classes'] = len(set(y_train))
            _num_classes = len(set(y_train))
            # config['num_steps'] = len(x_train) // config['_bptt']
            _steps_per_epoch = len(x_train) // _bptt

            if _bptt > n_samples:
                if verbose:
                    print(
                        f"WARNING: _bptt {_bptt} is larger than the number "
                        f"of samples in the training set, {n_samples}. "
                        f"Setting _bptt=128.")
                _bptt = 128

        seed_all(_extra_prior_kwargs_dict.get('rand_seed', 0))

        # Permutation of label order
        if do_permute and (not is_wrapper):
            label_perm = np.random.permutation(_num_classes)
        else:
            label_perm = np.arange(_num_classes)

        _invert_perm_map = {
            int(label_perm[_i]): np.int64(_i) for _i in range(_num_classes)
        }
        rev_invert_perm_map = {
            np.int64(_i): int(label_perm[_i]) for _i in range(_num_classes)
        }

        _x, _y = x_train, y_train
        # Permutation of feature order
        if do_permute and (not is_wrapper):
            feat_idx = np.random.permutation(_x.shape[1])
        else:
            feat_idx = np.arange(_x.shape[1])

        # Permutation of train data order
        idx = np.random.permutation(_x.shape[0])
        _x = _x[idx, ...]
        _y = _y[idx, ...]

        _y = loop_translate(_y, rev_invert_perm_map)

        _x = _x[:, feat_idx, ...]
        _x_val = _x_val[:, feat_idx, ...]
        _x_test = _x_test[:, feat_idx, ...]

        # label balancing
        num_classes = len(np.unique(np.unique(_y)))
        if do_prompt_tuning and _extra_prior_kwargs_dict.get('tuned_prompt_label_balance', 'equal') == 'proportional':
            int_y = _y.astype(int)
            _label_weights = np.bincount(int_y) / len(int_y)
            _label_weights = torch.from_numpy(_label_weights).float().to(device)
        else:
            _label_weights = None

        if _extra_prior_kwargs_dict.get("do_preprocess", False):
            preprocess_type = _extra_prior_kwargs_dict.get("preprocess_type", "none")
            summerize_after_prep = _extra_prior_kwargs_dict.get("summerize_after_prep", "False")

            _x = preprocess_input(torch.from_numpy(_x.copy().astype(np.float32)), preprocess_type, summerize_after_prep,
                                  train_args, is_train=True)
            _x_val = preprocess_input(torch.from_numpy(_x_val.copy().astype(np.float32)), preprocess_type,
                                      summerize_after_prep, train_args, is_train=False)
            _x_test = preprocess_input(torch.from_numpy(_x_test.copy().astype(np.float32)), preprocess_type,
                                       summerize_after_prep, train_args, is_train=False)
            if train_args.summerize_after_prep:
                _x, _x_val, _x_test = summarize_after(_x, _x_val, _x_test, _y, _y_val, _y_test, num_features, train_args)
        else:
            _x = torch.from_numpy(_x.copy().astype(np.float32))
            _x_val = torch.from_numpy(_x_val.copy().astype(np.float32))
            _x_test = torch.from_numpy(_x_test.copy().astype(np.float32))

        # feature padding
        do_pf = _extra_prior_kwargs_dict.get("pad_features", True)
        if do_pf:
            def pad_data(data):
                return torch.cat([data, torch.zeros(data.shape[0], num_features - data.shape[1])], dim=1)

            if _x.shape[1] < num_features:
                _x = pad_data(_x)
            if _x_val.shape[1] < num_features:
                _x_val = pad_data(_x_val)
            if _x_test.shape[1] < num_features:
                _x_test = pad_data(_x_test)

        _train_ds = TabDS(_x, _y)
        _val_ds = TabDS(_x_val, _y_val)
        _test_ds = TabDS(_x_test, _y_test)

        return (_x, _y, _x_val, _y_val, _x_test, _y_test, _invert_perm_map,
                _steps_per_epoch, num_classes, _label_weights, _train_ds, _val_ds, _test_ds)

    def make_dataloaders(_bptt=bptt, _not_zs=True):

        _dl, _bptt = get_train_dataloader(train_ds, bptt=_bptt, shuffle=False, num_workers=n_workers, drop_last=True,
                                          agg_k_grads=aggregate_k_gradients, not_zero_shot=_not_zs)

        _val_dl = DataLoader(
            val_ds, batch_size=min(_bptt, y_val.shape[0] // 2), shuffle=False, num_workers=n_workers,
        )

        _test_dl = DataLoader(
            test_ds, batch_size=min(_bptt, y_val.shape[0] // 2), shuffle=False, num_workers=n_workers,
        )
        # Fix the prior data TabPFN will use for fitting when including real data points
        x_data_for_fitting = []
        y_data_for_fitting = []
        # td is a list of tensors
        x_data_concat: Tensor = Tensor()
        y_data_concat: Tensor = Tensor()
        for idx, (td, _, _) in enumerate(_dl):
            x_data_for_fitting.append(td[0])
            y_data_for_fitting.append(td[1])
            x_data_concat = torch.cat(x_data_for_fitting, dim=0)
            y_data_concat = torch.cat(y_data_for_fitting, dim=0)
            if x_data_concat.shape[0] >= real_data_qty:
                break
        dl_data_for_fitting = [x_data_concat, y_data_concat]
        return _dl, _val_dl, _test_dl, _bptt, dl_data_for_fitting

    # REAL PRIOR
    if real_prior:
        # load data
        seed_all(extra_prior_kwargs_dict.get('rand_seed'))

        if do_kl_loss:
            if not extra_prior_kwargs_dict['uniform_bptt']:
                print("KL loss with TabPFN-zs only supports uniform _bptt")
                extra_prior_kwargs_dict['uniform_bptt'] = True

        data_for_fitting = None

        (x, y, x_val, y_val, x_test, y_test, invert_perm_map, steps_per_epoch,
         num_classes, label_weights, train_ds, val_ds, test_ds) = make_datasets(extra_prior_kwargs_dict,
                                                                                do_permute=not_zs, _bptt=bptt,
                                                                                _steps_per_epoch=steps_per_epoch,
                                                                                is_wrapper=is_wrapper,
                                                                                private_ds=private_data)
        old_bptt = bptt
        dl, val_dl, test_dl, bptt, data_for_fitting = make_dataloaders(_bptt=bptt, _not_zs=not_zs)
        val_dl = get_subset_dl(extra_prior_kwargs_dict, val_dl)
        if epochs == 0:
            return None, None, None, test_dl

        if verbose:
            print("Dataset information: ")
            print("Length, batch size of training dataloader: ", len(dl), dl.batch_size)
            print("Length of validation dataloader: ", len(val_dl), val_dl.batch_size)
            print("Length of test dataloader: ", len(test_dl), test_dl.batch_size)
            if data_for_fitting:
                print("Size of data for fitting: ", len(data_for_fitting[0]))

        if do_zs or do_kl_loss:
            from scripts.transformer_prediction_interface import TabPFNClassifier
            if extra_prior_kwargs_dict.get('zs_eval_ensemble', 0) > 0:
                ens_size = extra_prior_kwargs_dict.get('zs_eval_ensemble', 0)
            else:
                ens_size = 32
            eval_model = TabPFNClassifier(device='cuda',
                                          N_ensemble_configurations=ens_size,
                                          base_path=".",
                                          seed=extra_prior_kwargs_dict.get('rand_seed', 0),
                                          batch_size_inference=1,
                                          )
            if do_kl_loss:
                eval_model.fit(data_for_fitting[0], data_for_fitting[1], overwrite_warning=True)
        else:
            eval_model = None

        if old_bptt != bptt:
            max_pos = int((len(data_for_fitting[0]) // 10) * .8)
            if verbose:
                print("_bptt changed from {} to {}".format(old_bptt, bptt))
                print("max_pos: ", max_pos)
            if extra_prior_kwargs_dict.get('uniform_bptt', False):
                single_eval_pos_gen = lambda: np.random.randint(0, max_pos)
            else:
                single_eval_pos_gen = max_pos
        zs_res_dict = dict()
        if do_zs:
            def tpc_data_eval(_cl: int = 1000, _x=None, _y=None, x_val=None, y_val=None, ens_size=1):
                # update num_classes depending on the data
                num_classes_local = len(np.unique(_y))
                tpc_start_time = time.time()
                results = dict()
                if _cl > len(_x):
                    _cl = len(_x) - 1
                eval_model.fit(_x[:_cl, ...], _y[:_cl, ...], overwrite_warning=True)
                predictions = eval_model.predict(x_val).astype(np.int64)
                outputs = np.zeros((len(x_val), num_classes_local))
                output_eval = eval_model.predict_proba(x_val)
                for _j in range(output_eval.shape[1]):
                    outputs[:, invert_perm_map[_j]] = output_eval[:, _j]
                for _i in range(num_classes_local):
                    # try:
                    outputs[:, _i] = outputs[:, invert_perm_map[_i]]
                targets = y_val
                warnings.filterwarnings("ignore")
                end_time = time.time()
                results['Eval_Time'] = safe_round(end_time - tpc_start_time)
                results['Accuracy'] = safe_round(accuracy_score(targets, predictions))
                try:
                    results['Log_Loss'] = safe_round(log_loss(targets, outputs, labels=np.arange(num_classes_local)))
                except Exception as _e:
                    if verbose:
                        print("Error calculating log loss: ", _e)
                    results['Log_Loss'] = 0.0
                results['F1_Weighted'] = safe_round(f1_score(targets, predictions, average='weighted'))
                results['F1_Macro'] = safe_round(f1_score(targets, predictions, average='macro'))
                try:
                    if num_classes == 2:
                        results['ROC_AUC'] = safe_round(
                            roc_auc_score(targets, outputs[:, 1], labels=np.arange(num_classes_local)))
                    else:
                        results['ROC_AUC'] = safe_round(
                            roc_auc_score(targets, outputs, labels=np.arange(num_classes_local), multi_class='ovr'))
                except Exception as ex:
                    if verbose:
                        print("Error calculating ROC AUC: ", ex)
                    results['ROC_AUC'] = 0.0
                warnings.filterwarnings("default")
                return results

            # res_dict = dict()
            val_results = tpc_data_eval(_cl=real_data_qty, _x=data_for_fitting[0], _y=data_for_fitting[1], x_val=x_val,
                                        y_val=y_val, ens_size=extra_prior_kwargs_dict.get('zs_eval_ensemble', 0))
            zs_res_dict = dict(zs_res_dict, **{"Val_" + k: v for k, v in val_results.items()})
            test_results = tpc_data_eval(_cl=real_data_qty, _x=data_for_fitting[0], _y=data_for_fitting[1], x_val=x_test,
                                         y_val=y_test, ens_size=extra_prior_kwargs_dict.get('zs_eval_ensemble', 0))
            zs_res_dict = dict(zs_res_dict, **{"Test_" + k: v for k, v in test_results.items()})
            with open(os.path.join(extra_prior_kwargs_dict.get('save_path'), 'zs_eval_ensemble.json'), 'w') as f:
                json.dump(zs_res_dict, f)
            if extra_prior_kwargs_dict.get('wandb_log', False):
                import wandb
                wandb.log(zs_res_dict, step=1, commit=True)
    else:
        raise Exception("Excepted a real dataset")

    if do_zs:
        return "", zs_res_dict, None, None

    encoder = _encoder_generator(num_features, emsize)
    # style_def = _dl.get_test_batch()[0][0] # the style in batch of the form ((style, _x, y), target, single_eval_pos)
    style_def = None
    # print(f'Style definition of first 3 examples: {style_def[:3] if style_def is not None else None}')
    style_encoder = style_encoder_generator(style_def.shape[1], emsize) if (style_def is not None) else None
    if do_kl_loss:
        assert num_classes < 11, "KL loss with TabPFN-zs only supports 10 classes or fewer"
        n_out = 10
        training_criterion = kl_divergence
    elif isinstance(training_criterion, nn.GaussianNLLLoss):
        n_out = 2
    elif isinstance(training_criterion, nn.CrossEntropyLoss):
        n_out = training_criterion.weight.shape[0]
    else:
        n_out = 1
    model = TransformerModel(encoder, n_out, emsize, nhead, nhid, nlayers, dropout, style_encoder=style_encoder,
                             y_encoder=_y_encoder_generator(1, emsize), input_normalization=input_normalization,
                             pos_encoder=(_pos_encoder_generator or
                                          positional_encodings.NoPositionalEncoding)(emsize, bptt * 2),
                             decoder=decoder, init_method=initializer, efficient_eval_masking=efficient_eval_masking,
                             prefix_size=prefix_size,
                             n_classes=num_classes, prefix_label_probs=label_weights,
                             num_features=extra_prior_kwargs_dict.get("num_features", 100),
                             private=do_private, b_linear=extra_prior_kwargs_dict.get("linear", False),
                             **model_extra_args,
                             )
    model.criterion = training_criterion
    encoder_mismatch = False
    decoder_mismatch = False
    if load_weights_from_this_state_dict is not None:
        if do_kl_loss:
            load_weights_from_this_state_dict.pop('criterion.weight')
        if num_classes > 10:
            # initialize a new decoder head
            decoder_mismatch = True
            load_weights_from_this_state_dict['decoder.2.weight'] = model.state_dict()['decoder.2.weight']
            load_weights_from_this_state_dict['decoder.2.bias'] = model.state_dict()['decoder.2.bias']
            load_weights_from_this_state_dict['criterion.weight'] = model.state_dict()['criterion.weight']
        if load_weights_from_this_state_dict.get('prefix_embedding.weight', None) is None and model.state_dict().get(
                'prefix_embedding.weight', None) is not None:
            load_weights_from_this_state_dict['prefix_embedding.weight'] = model.state_dict()['prefix_embedding.weight']
        if do_private:
            load_weights_from_this_state_dict = load_and_combine_attention_weights(load_weights_from_this_state_dict,
                                                                                   nlayers)
        if load_weights_from_this_state_dict.get('encoder.weight', None) is not None:
            load_shape = load_weights_from_this_state_dict.get('encoder.weight', None).shape
            model_shape = model.state_dict().get('encoder.weight', None).shape
            if load_shape != model_shape:
                encoder_mismatch = True
                if verbose:
                    print("Encoder weight shape mismatch: ", load_shape, model_shape,
                          "Using randomly initialized encoder weights from model instead")
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
    if verbose:
        print("Params to optimize: ", params_to_optimize)

        print(f"Using a Transformer with {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.{2}f} M parameters")
    if do_prompt_tuning and do_private:
        model.freeze_parameters_except_named(params_to_optimize)
    if initialize_with_model:
        if hasattr(initialize_with_model,'state_dict'):
            try:
                for (k, v), (k2, v2) in zip(model.state_dict().items(), initialize_with_model.state_dict().items()):
                    print(k, ((v - v2) / v).abs().mean(), v.shape)
            except ValueError:
                pass

    model.to(device)
    if using_dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank,
                                                          broadcast_buffers=False)

    if not real_prior:
        dl.model = model

    # learning rate
    if lr is None:
        lr = get_openai_lr(model)
        if verbose:
            print(f"Using OpenAI max lr of {lr}.")

    if do_prompt_tuning:
        pto = (p for n, p in model.named_parameters() if any([x in n for x in params_to_optimize]))
    else:
        pto = model.parameters()

    optimizer = torch.optim.AdamW(pto, lr=lr, weight_decay=weight_decay)
    sched_obj = scheduler(optimizer, warmup_epochs,
                          # when training for fixed time lr schedule takes 100 steps
                          epochs if epochs is not None else 100)

    scaler = GradScaler() if (train_mixed_precision and not do_private) else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    if do_private:
        eps = extra_prior_kwargs_dict.get('epsilon', 1.0)
        delta = extra_prior_kwargs_dict.get('delta', 1e-5)
        max_grad_norm = extra_prior_kwargs_dict.get('max_grad_norm', 1.0)
        if verbose:
            print("DP training with epsilon: ", eps, "delta: ", delta, "max_grad_norm: ", max_grad_norm)
        errors = ModuleValidator.validate(model, strict=False)
        if len(errors) > 0 and verbose:
            print("Differentially private model architecture errors: ")
            print(errors)
        privacy_engine = PrivacyEngine()
        y_emb = model.prefix_y_embedding.clone()
        model, optimizer, dl = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dl,
            epochs=epochs,
            target_epsilon=eps,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
        )
        model.prefix_y_embedding = y_emb

    master_epoch_count = []

    def real_data_eval(r_model, cl=1000, train_data=None, real_data_val_dl=None,
                       softmax_temperature=torch.log(torch.tensor([0.8]))):
        eval_start_time = time.time()
        td = copy.deepcopy(train_data)
        num_classes_local = len(torch.unique(td[1]))
        td[0] = td[0][:cl, ...]
        td[1] = td[1][:cl, ...]
        single_eval_pos = len(td[0])
        softmax_temperature = softmax_temperature.to(device)
        # print("In real data eval, eval set size: ", len(_val_dl.dataset))

        with torch.inference_mode():
            # correct = 0
            # total = len(_val_dl.dataset)
            prediction_list = []
            target_list = []
            output_list = []
            for batch, (data, targets, _) in enumerate(real_data_val_dl):

                if batch == 0 and verbose:
                    print("turned off sample printing")
                    # print("Data sample (train features, train labels, val/test features, val/test labels): ",
                    #      td[0][:10], "\n", td[1][:10], "\n", data[0][:10], "\n", data[1][:10], "\n")

                if extra_prior_kwargs_dict.get('debug', False):
                    # Extra safeguard against test set contamination, permute label order before passing into model
                    data_temp_idx = torch.randperm(data[1].nelement())
                    data[1] = data[1].view(-1)[data_temp_idx].view(data[1].size())

                batch_data = tuple([torch.cat((td[0], data[0]), dim=0).to(torch.float32),
                                    torch.cat((td[1], data[1]), dim=0).to(torch.float32)])
                output = r_model(
                    tuple(_e.to(device) if torch.is_tensor(_e)
                          else _e for _e in batch_data) if isinstance(batch_data, tuple)
                    else batch_data.to(device), single_eval_pos=single_eval_pos)
                # invert permutation of labels
                new_output = loop_translate(output, invert_perm_map)
                output = new_output
                output = output[:, 0:num_classes_local] / torch.exp(softmax_temperature)
                output = torch.nn.functional.softmax(output, dim=-1)
                output_list.append(output)
                _, predicted = torch.max(output.cpu().data, 1)
                prediction_list.append(predicted)
                target_list.append(targets)
            outputs = torch.cat(output_list, dim=0).cpu().numpy()
            predictions = torch.cat(prediction_list, dim=0).cpu().numpy()
            targets = torch.cat(target_list, dim=0).cpu().numpy()
            # print("In real data eval, Targets: ", targets[:20])

        results = dict()
        warnings.filterwarnings("ignore")
        results['Eval_Time'] = safe_round(time.time() - eval_start_time)
        results['Accuracy'] = safe_round(accuracy_score(targets, predictions))
        try:
            results['Log_Loss'] = safe_round(log_loss(targets, outputs, labels=np.arange(num_classes_local)))
        except Exception as ex:
            if verbose:
                print("Error calculating log loss: ", ex)
            results['Log_Loss'] = 0.0
        results['F1_Weighted'] = safe_round(f1_score(targets, predictions, average='weighted'))
        results['F1_Macro'] = safe_round(f1_score(targets, predictions, average='macro'))
        try:
            if num_classes_local == 2:
                results['ROC_AUC'] = safe_round(
                    roc_auc_score(targets, outputs[:, 1], labels=np.arange(num_classes_local)))
            else:
                results['ROC_AUC'] = safe_round(
                    roc_auc_score(targets, outputs, labels=np.arange(num_classes_local), multi_class='ovr'))
        except Exception as ex:
            if verbose:
                print("Error calculating ROC AUC: ", ex)
            results['ROC_AUC'] = 0.0

        warnings.filterwarnings("default")

        return results, outputs, targets

    # def train_epoch(e_model, e_optimizer, _dl, boost_this_epoch=False, _eval_model=None, bptt_search=False):
    def train_epoch(e_model, e_optimizer, _dl, boost_this_epoch=False, eval_model=None, bptt_search=False):
        tracker = GPULossTracker(device=device)
        if 0 < max_time < time.time() - start_time:
            print("Max time reached. Exiting")
            exit(0)
        e_model.train()  # Turn on the train mode
        # Confirm that the correct params are frozen and unfrozen
        if do_prompt_tuning:
            if not do_private:
                e_model.freeze_parameters_except_named(params_to_optimize)
            for n, p in e_model.named_parameters():
                grad_reqd = False
                for s in params_to_optimize:
                    if s in n:
                        grad_reqd = True
                        break
                assert p.requires_grad == grad_reqd, "Parameter {} does not have the correct grad requirement!".format(
                    n)

        # total_loss = 0.
        # total_positional_losses = 0.
        # total_positional_losses_recorded = 0
        # nan_steps = 0
        # ignore_steps = 0
        epoch_start_time = time.time()
        time_to_get_batch = 0
        time_to_get_batches = 0
        forward_time = 0
        forward_times = 0
        backward_times = 0
        # loss_times = 0
        # grad_times = 0
        step_time = 0
        before_get_batch = time.time()
        batches_seen = 0
        shuffle_every_epoch = extra_prior_kwargs_dict.get('shuffle_every_epoch', False)
        permute_feature_pos = extra_prior_kwargs_dict.get('permute_feature_position_in_ensemble', False)
        for batch, (data, targets, single_eval_pos) in enumerate(_dl):
            if isinstance(data, list):
                data = tuple(data)
            if isinstance(single_eval_pos, torch.Tensor) and single_eval_pos.numel() == 0:
                single_eval_pos = None
            if using_dist and not (batch % aggregate_k_gradients == aggregate_k_gradients - 1):
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
                time_to_get_batches += time_to_get_batch
                before_forward = time.time()
                if boosting:
                    single_eval_pos = len(targets) // 2
                elif bptt_extra_samples is None:
                    single_eval_pos = single_eval_pos_gen() if callable(single_eval_pos_gen) else single_eval_pos_gen
                else:
                    single_eval_pos = max(targets.shape[0] - bptt_extra_samples, 0)
                with autocast('cuda', enabled=scaler is not None):
                    # If style is set to None, it should not be transferred to _device
                    output = e_model(
                        tuple(_e.to(torch.float32).to(device) if torch.is_tensor(_e)
                              else _e for _e in data) if isinstance(
                            data, tuple) else data.to(device)
                        , single_eval_pos=single_eval_pos)
                    if not bptt_search:
                        assert output.requires_grad, "Output does not require gradients"
                    forward_time = time.time() - before_forward
                    forward_times += forward_time
                    before_backward = time.time()
                    if single_eval_pos is not None:
                        targets = targets[single_eval_pos:]
                    if isinstance(training_criterion, nn.GaussianNLLLoss):
                        assert output.shape[-1] == 2, \
                            'need to write a little bit of code to handle multiple regression targets at once'
                        mean_pred = output[..., 0]
                        var_pred = output[..., 1].abs()
                        losses = training_criterion(mean_pred.flatten(),
                                                    targets.to(device).flatten(),
                                                    var=var_pred.flatten())
                    elif isinstance(training_criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                        losses = training_criterion(output.flatten(), targets.to(device).flatten())
                    elif isinstance(training_criterion, nn.CrossEntropyLoss):
                        losses = training_criterion(output.reshape(-1, n_out), targets.to(device).long().flatten())
                    elif do_kl_loss:
                        real_data_preds = eval_model.predict_proba(data[0])
                        if real_data_preds.shape[1] < output.shape[1]:
                            real_data_preds = np.concatenate([real_data_preds, np.zeros(
                                (real_data_preds.shape[0], output.shape[1] - real_data_preds.shape[1]))], axis=1)
                        if real_data_preds.shape[0] != output.shape[0]:
                            if verbose:
                                print(f"Real data preds and tuned prompt output have different shapes: ",
                                      real_data_preds.shape, output.shape)
                            smaller_shape = min(real_data_preds.shape[0], output.shape[0])
                            real_data_preds = real_data_preds[:smaller_shape, :]
                            output = output[:smaller_shape, :]
                        real_data_preds = torch.tensor(real_data_preds).to(device)
                        assert real_data_preds.shape == output.shape, (f"Real data preds and tuned prompt output have "
                                                                       f"different shapes: {real_data_preds.shape} "
                                                                       f"and {output.shape}")
                        losses = training_criterion(real_data_preds, output)
                    else:
                        losses = training_criterion(output, targets)
                    if boosting or do_kl_loss:
                        loss = losses.mean()
                        nan_share = torch.tensor([0])
                    else:
                        if len(output.shape) == 2:
                            output = output.unsqueeze(1)
                        losses = losses.view(*output.shape[0:2])

                        loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
                        loss = loss / aggregate_k_gradients

                if scaler:
                    loss = scaler.scale(loss)
                if boosting and boost_this_epoch:
                    cur_grads = []
                    # Backward pass for each prediction/target pair
                    if prior_grad_dict is None:
                        prior_grad_iter = None
                    else:
                        prior_grad_iter = prior_grad_dict[batch].to(output.device)
                    output_grad = autograd.grad(loss, output)[0]
                    gradient_dict[batch] = output_grad.detach().cpu().clone()
                    # cur_grads.append(output_grad.detach().cpu().clone())

                    if prior_grad_iter is not None:
                        grad_shape = output_grad.shape
                        flat_grad = output_grad.flatten()
                        grad_signs = torch.sign(flat_grad)
                        flat_prior_grad = prior_grad_iter.flatten()
                        cur_weight = 0.65
                        flat_grad_new = torch.sqrt(
                            cur_weight * torch.pow(flat_grad, 2) + (1 - cur_weight) * torch.pow(flat_prior_grad, 2))
                        flat_grad_new_signs = torch.sign(flat_grad_new)
                        flat_grad_new[flat_grad_new_signs != grad_signs] *= -1
                        output_grad = flat_grad_new.reshape(grad_shape)

                    output.backward(output_grad)
                    # gradient_dict[batch] = torch.cat(cur_grads, dim=0)
                elif bptt_search:
                    pass
                else:
                    loss.backward()
                if do_private:
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            if not hasattr(p, "grad_sample") or p.grad_sample is None:
                                p.grad_sample = p.grad.clone().unsqueeze(0)
                    loss.backward()
                backward_times += time.time() - before_backward
                tracker.update(loss)
                if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                    if scaler:
                        scaler.unscale_(e_optimizer)
                    if do_private:
                        pass
                    else:
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
            before_get_batch = time.time()
            batches_seen += 1
        # Total positional losses is a torch tensor of size _bptt (batch size)
        if batches_seen < extra_prior_kwargs_dict.get('min_batches_per_epoch', 1):
            raise ValueError(
                "Not enough batches seen in epoch: saw {} batches, "
                "expected at least {}".format(batches_seen,
                                              extra_prior_kwargs_dict.get(
                                                    'min_batches_per_epoch',
                                                    1)))

        total_loss = tracker.average()

        if verbose:
            print("train_epoch time: ", round(time.time() - epoch_start_time, 2))
            print("time to get batches: ", round(time_to_get_batches, 2))
            print("time in forward: ", round(forward_times, 2))
            print("time in backward: ", round(backward_times, 2))

        return total_loss, None, \
            time_to_get_batch, forward_time, step_time, None, \
            None

    def concat_embedding(ec, model, method):
        # extract embedding parameters
        _device = ec.model.prefix_embedding.weight.device
        if method == "duplicate":
            ec.concatenated_embedding = torch.cat([ec.original_embedding, ec.original_embedding], dim=0).to(_device)
            ec.concatenated_y_embedding = torch.cat([ec.original_y_embedding, ec.original_y_embedding], dim=0).to(
                _device)
            ec.prefix_size = ec.original_prefix_size * 2
        elif method.startswith("rand-init"):
            num_to_concat = min(int(method.split("-")[-1]), len(ec.prefix_weights) + 1)
            if verbose:
                print("Concatenating {} embeddings".format(num_to_concat))
            if num_to_concat == 1:
                ec.concatenated_embedding = ec.original_embedding
                ec.concatenated_y_embedding = ec.original_y_embedding
                ec.prefix_size = ec.original_prefix_size
            else:
                ec.concatenated_embedding = torch.cat(
                    [ec.original_embedding.to(_device)] + [ec.prefix_weights[_i]['prefix_weights'].to(_device) for _i in
                                                           range(num_to_concat - 1)], dim=0).to(_device)
                ec.concatenated_y_embedding = torch.cat(
                    [ec.original_y_embedding.to(_device)] + [ec.prefix_weights[_i]['prefix_y_labels'].to(_device) for
                                                             _i in range(num_to_concat - 1)], dim=0).to(_device)
                if "size-ctl" in method:
                    # select random sample of size prefix_size
                    if "perm" in method:
                        # random permutation
                        sel = torch.randperm(ec.concatenated_embedding.shape[0])[:ec.original_prefix_size].to(_device)
                    else:
                        # first-k-samples
                        total_emb_size = ec.original_prefix_size
                        emb_size = total_emb_size // num_to_concat
                        orig_emb_size = ec.original_embedding.shape[0]
                        start_pos = [_j * orig_emb_size for _j in range(num_to_concat)]
                        sel = torch.cat([torch.arange(_i, _i + emb_size) for _i in start_pos], dim=0).to(_device)

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

    def restore_embedding(ec, _model):
        _model.prefix_embedding.weight = nn.Parameter(ec.original_embedding)
        _model.prefix_y_embedding = ec.original_y_embedding
        _model.prefix_size = ec.original_prefix_size
        _model.freeze_parameters_except_named(params_to_optimize)
        return _model

    def save_prefix_weights(model, path, _i, do_concat, prefix_weights_l):
        # Save prefix weights
        prefix_weights = model.prefix_embedding.weight.cpu().detach().numpy()
        # prefix_weights = model.state_dict()['prefix_embedding.weight'].cpu().numpy()
        prefix_fn = f"prefix_weights_{_i}.npy"
        prefix_save_path = os.path.join(path, prefix_fn)
        # if not is_wrapper:
        np.save(prefix_save_path, prefix_weights)
        prefix_y_labels = model.prefix_y_embedding.cpu().numpy()
        prefix_y_fn = f"prefix_y_labels_{_i}.npy"
        prefix_y_save_path = os.path.join(path, prefix_y_fn)

        # if not is_wrapper:
        np.save(prefix_y_save_path, prefix_y_labels)
        if do_concat:
            prefix_weights_l.append({"prefix_weights": torch.from_numpy(prefix_weights).float(),
                                     "prefix_y_labels": torch.from_numpy(prefix_y_labels)})
            # print("Prefix weights list length: ", len(prefix_weights_l))
        return prefix_weights_l

    def update_ensemble_acc(ens_acc, ens_acc_nc, ens_acc_test, ens_acc_test_nc, _num_classes):
        num_classes_local_val = len(np.unique(labels_np))
        num_classes_local_test = len(np.unique(labels_np_test))
        predictions_np = np.argmax(probs_np, axis=1)
        predictions_np_test = np.argmax(probs_np_test, axis=1)
        try:
            if _num_classes == 2:
                roc_auc = safe_round(roc_auc_score(labels_np, probs_np[:, 1], labels=np.arange(num_classes_local_val)))
                test_roc_auc = safe_round(
                    roc_auc_score(labels_np_test, probs_np_test[:, 1], labels=np.arange(num_classes_local_test)))
            else:
                roc_auc = safe_round(
                    roc_auc_score(labels_np, probs_np, labels=np.arange(num_classes_local_val), multi_class='ovr'))
                test_roc_auc = safe_round(
                    roc_auc_score(labels_np_test, probs_np_test, labels=np.arange(num_classes_local_test),
                                  multi_class='ovr'))
        except Exception as ex:
            if verbose:
                print("Error calculating ROC AUC: ", ex)
            roc_auc = 0.0
            test_roc_auc = 0.0
        f1_weighted = safe_round(f1_score(labels_np, predictions_np, average='weighted'))
        f1_macro = safe_round(f1_score(labels_np, predictions_np, average='macro'))
        try:
            ll = safe_round(log_loss(labels_np, probs_np, labels=np.arange(num_classes_local_val)))
        except Exception as ex:
            if verbose:
                print("Error calculating ll/ECE/TACE: ", ex)
            ll = 0.0
        test_f1_weighted = safe_round(f1_score(labels_np_test, predictions_np_test, average='weighted'))
        test_f1_macro = safe_round(f1_score(labels_np_test, predictions_np_test, average='macro'))
        try:
            test_ll = safe_round(log_loss(labels_np_test, probs_np_test, labels=np.arange(num_classes_local_test)))
        except Exception as ex:
            if verbose:
                print("Error calculating ll/ECE/TACE: ", ex)
            test_ll = 0.0
        if do_prompt_tuning:
            predictions_np_nc = np.argmax(probs_np_nc, axis=1)
            predictions_np_nc_test = np.argmax(probs_np_nc_test, axis=1)
            nc_f1_weighted = safe_round(f1_score(labels_np_nc, predictions_np_nc, average='weighted'))
            nc_f1_macro = safe_round(f1_score(labels_np_nc, predictions_np_nc, average='macro'))
            try:
                if _num_classes == 2:
                    roc_auc_nc = safe_round(
                        roc_auc_score(labels_np_nc, probs_np_nc[:, 1], labels=np.arange(num_classes_local_val)))
                    test_roc_auc_nc = safe_round(roc_auc_score(labels_np_nc_test, probs_np_nc_test[:, 1],
                                                               labels=np.arange(num_classes_local_test)))
                else:
                    roc_auc_nc = safe_round(
                        roc_auc_score(labels_np_nc, probs_np_nc, labels=np.arange(num_classes_local_val),
                                      multi_class='ovr'))
                    test_roc_auc_nc = safe_round(
                        roc_auc_score(labels_np_nc_test, probs_np_nc_test, labels=np.arange(num_classes_local_test),
                                      multi_class='ovr'))
            except Exception as ex:
                if verbose:
                    print("Error calculating ROC AUC: ", ex)
                roc_auc_nc = 0.0
                test_roc_auc_nc = 0.0
            try:
                nc_ll = safe_round(log_loss(labels_np_nc, probs_np_nc, labels=np.arange(num_classes_local_val)))
            except Exception as ex:
                if verbose:
                    print("Error calculating ll/ECE/TACE: ", ex)
                nc_ll = 0.0
            nc_test_f1_weighted = safe_round(f1_score(labels_np_nc_test, predictions_np_nc_test, average='weighted'))
            nc_test_f1_macro = safe_round(f1_score(labels_np_nc_test, predictions_np_nc_test, average='macro'))
            try:
                nc_test_ll = safe_round(
                    log_loss(labels_np_nc_test, probs_np_nc_test, labels=np.arange(num_classes_local_test)))
            except Exception as ex:
                if verbose:
                    print("Error calculating ll/ECE/TACE: ", ex)
                nc_test_ll = 0.0
        else:
            nc_f1_weighted = 0
            nc_f1_macro = 0
            roc_auc_nc = 0
            test_roc_auc_nc = 0
            nc_test_f1_weighted = 0
            nc_test_f1_macro = 0
            nc_ll = 0
            nc_test_ll = 0
        if verbose:
            # print("In update ensemble acc, Targets: ", labels_np[:20])
            print("Ensemble accuracy: ", ens_acc, "Ensemble accuracy (NC): ", ens_acc_nc)
        new_res = {
            "Ens_Val_Accuracy": ens_acc,
            "Ens_Val_Accuracy_NC": ens_acc_nc,
            "Ens_Val_F1_Weighted": f1_weighted,
            "Ens_Val_F1_Macro": f1_macro,
            "Ens_Val_F1_Weighted_NC": nc_f1_weighted,
            "Ens_Val_F1_Macro_NC": nc_f1_macro,
            "Ens_Val_Log_Loss": ll,
            "Ens_Val_Log_Loss_NC": nc_ll,
            "Ens_Val_ROC_AUC": roc_auc,
            "Ens_Val_ROC_AUC_NC": roc_auc_nc,
            "Ens_Test_Accuracy": ens_acc_test,
            "Ens_Test_Accuracy_NC": ens_acc_test_nc,
            "Ens_Test_F1_Weighted": test_f1_weighted,
            "Ens_Test_F1_Macro": test_f1_macro,
            "Ens_Test_F1_Weighted_NC": nc_test_f1_weighted,
            "Ens_Test_F1_Macro_NC": nc_test_f1_macro,
            "Ens_Test_Log_Loss": test_ll,
            "Ens_Test_Log_Loss_NC": nc_test_ll,
            "Ens_Test_ROC_AUC": test_roc_auc,
            "Ens_Test_ROC_AUC_NC": test_roc_auc_nc,
        }
        return new_res

    def train_test_loop(t_model, t_optim, t_sched, _eval_model, _dl, _val_dl, _test_dl):
        # Select a fixed training data prior of size _bptt
        return_outputs = None
        return_targets = None
        _res_dict = None
        # best_val_score = best_val_score_nc = -1.0
        # best_val_score = best_val_score_nc = 0
        best_total_loss = 1e9
        best_val_embed: Tensor = Tensor()
        if do_prompt_tuning:
            best_val_embed = t_model.prefix_embedding.weight.detach().cpu()
        best_res_dict = None
        best_outputs = None
        best_targets = None
        # is_best = False
        patience = 0

        for epoch in (range(1, epochs + 1) if epochs is not None else itertools.count(1)):
            is_best = False
            if verbose:
                print('epoch', epoch, 'of', epochs)
            boost_this_epoch = True if epoch == 1 else False
            epoch_start_time = time.time()
            master_epoch_count.append(1)
            if do_private:
                with (BatchMemoryManager(
                        data_loader=_dl,
                        max_physical_batch_size=batch_size,
                        optimizer=optimizer
                ) as memory_safe_data_loader):
                    (total_loss, total_positional_losses, _time_to_get_batch,
                     _forward_time, step_time, nan_share, ignore_share) = train_epoch(
                        t_model, t_optim, memory_safe_data_loader, boost_this_epoch, eval_model=_eval_model,
                        bptt_search=False)
            else:
                (total_loss, total_positional_losses, _time_to_get_batch, _forward_time, step_time, nan_share,
                 ignore_share) = train_epoch(
                    t_model, t_optim, _dl, boost_this_epoch, eval_model=_eval_model, bptt_search=False)
            # val_score = val_score_nc = val_score_concat = val_score_nc_concat = test_score = test_score_nc =
            # test_ece = test_tace = val_ece = val_tace = val_ece_nc = val_tace_nc = test_ece_nc = test_tace_nc = None
            # todo: from main
            # Confirm that the correct params are frozen and unfrozen
            if do_prompt_tuning:
                t_model.freeze_parameters_except_named(params_to_optimize)
                for n, p in t_model.named_parameters():
                    grad_reqd = False
                    for s in params_to_optimize:
                        if s in n:
                            grad_reqd = True
                    assert p.requires_grad == grad_reqd, ("Parameter {} does"
                                                          " not have the correct grad requirement!").format(n)
            t_model.train()  # Turn on the train mode
            total_loss, _, _time_to_get_batch, _forward_time, step_time, _, _ = \
                train_epoch(t_model, t_optim, _dl, boost_this_epoch=boost_this_epoch, eval_model=_eval_model, bptt_search=False)
            val_score = val_score_nc = None
            # val_score_concat = val_score_nc_concat = test_score = test_score_nc = None
            ####
            _res_dict = dict()
            _res_dict['epoch_train_time'] = safe_round(time.time() - epoch_start_time)
            _res_dict['master_epoch_count'] = len(master_epoch_count)
            if do_private:
                epsilon = privacy_engine.get_epsilon(extra_prior_kwargs_dict.get('delta', 1e-5))
                if verbose:
                    print("DP Epsilon is now: ", epsilon)
                _res_dict['epsilon_budget'] = epsilon
            # LONG_VAL_EP = ((epoch - 1) % validation_period == 0)
            if real_prior:
                val_start_time = time.time()
                _val_results, val_outputs, val_targets = real_data_eval(r_model=t_model, cl=real_data_qty,
                                                                        train_data=data_for_fitting, real_data_val_dl=_val_dl)
                _res_dict = dict(_res_dict, **{"Val_" + k: v for k, v in _val_results.items()})
                val_score = _res_dict["Val_Accuracy"]
                return_outputs = [val_outputs]
                return_targets = [val_targets]
                if do_prompt_tuning:
                    # TODO: will this work with context length 0? Should this be a hyperparameter?
                    if do_concat != "":
                        ec = EmbeddingConcatenator(t_model, do_concat, prefix_weights_l)
                        t_model = concat_embedding(ec, t_model, do_concat)
                        val_score_concat, _, _ = real_data_eval(r_model=ec.get_model(), cl=real_data_qty,
                                                                train_data=data_for_fitting, real_data_val_dl=_val_dl)
                        _res_dict = dict(_res_dict, **{"Val_concat_" + k: v for k, v in val_score_concat.items()})
                        val_score_nc_concat, _, _ = real_data_eval(r_model=ec.get_model(), cl=0,
                                                                   train_data=data_for_fitting, real_data_val_dl=_val_dl)
                        _res_dict = dict(_res_dict, **{"Val_concat_nc_" + k: v for k, v in val_score_nc_concat.items()})
                        t_model = restore_embedding(ec, t_model)
                        # Update optimizer parameters to include new embedding
                        t_optim = torch.optim.AdamW(t_model.parameters(), lr=lr, weight_decay=weight_decay)
                        t_sched = scheduler(t_optim, warmup_epochs, epochs if epochs is not None else 100)
                    else:
                        val_score_nc_concat = ""
                        val_score_concat = ""
                    val_score_nc, val_outputs, val_targets = real_data_eval(r_model=t_model, cl=0,
                                                                            train_data=data_for_fitting, real_data_val_dl=_val_dl)
                    return_outputs.append(val_outputs)
                    return_targets.append(val_targets)
                    _res_dict = dict(_res_dict, **{"Val_nc_" + k: v for k, v in val_score_nc.items()})

                # Early stopping logic
                score_condition = (round(total_loss, 2) < round(best_total_loss, 2))

                if score_condition:
                    patience = 0
                    best_total_loss = total_loss
                    is_best = True
                    if do_prompt_tuning:
                        best_val_embed = t_model.prefix_embedding.weight.detach().cpu()
                else:
                    patience += 1
                if verbose:
                    print("val_epoch time: ", round(time.time() - val_start_time, 2))

            elif hasattr(_dl, 'validate') and epoch % validation_period == 0:
                with torch.no_grad():
                    val_score = _dl.validate(model)

            no_patience = (patience > extra_prior_kwargs_dict.get('early_stopping_patience', 2))
            if is_best or (no_patience and "Test_Accuracy" not in _res_dict):
                _test_results, test_outputs, _test_targets = real_data_eval(r_model=t_model, cl=real_data_qty,
                                                                            train_data=data_for_fitting, real_data_val_dl=_test_dl)
                _res_dict = dict(_res_dict, **{"Test_" + k: v for k, v in _test_results.items()})
                return_outputs = return_outputs[:1] + [test_outputs] + return_outputs[1:]
                return_targets = return_targets[:1] + [_test_targets] + return_targets[1:]
                if do_prompt_tuning:
                    test_score_nc, test_outputs, _test_targets = real_data_eval(r_model=t_model, cl=0,
                                                                                train_data=data_for_fitting,
                                                                                real_data_val_dl=_test_dl)
                    _res_dict = dict(_res_dict, **{"Test_nc_" + k: v for k, v in test_score_nc.items()})
                    return_outputs.append(test_outputs)
                    return_targets.append(_test_targets)
                if is_best:
                    best_outputs = return_outputs
                    best_targets = return_targets
                    best_res_dict = _res_dict
            if verbose:
                get_time = (time.time() - epoch_start_time)
                print('-' * 89)
                print(
                    f'| end of epoch {epoch:3d} | time: {get_time:5.2f}s | mean loss {total_loss:5.2f} | '
                    f' | data time {_time_to_get_batch:5.2f} | step time {step_time:5.2f}'
                    f' | forward time {_forward_time:5.2f}'
                    f' | val score {val_score}' if val_score is not None
                    else f' | val score nc {_res_dict.get("Val_nc_Accuracy", 0)}' if val_score_nc is not None
                    else f' | test score {_res_dict.get("Test_Accuracy", 0)}' if _res_dict.get(
                        "Test_Accuracy", 0) is not None else ''
                                                             f' | test score nc {_res_dict.get("Test_nc_Accuracy", 0)}'
                    if _res_dict.get(
                        "Test_nc_Accuracy", 0) is not None else ''
                )
                print('-' * 89)
                if epoch_callback is not None and rank == 0:
                    epoch_callback(model, epoch / epochs, _res_dict)
                if val_score is not None:
                    # save the log to a json file
                    _res_dict = dict(_res_dict, **{
                        'epoch': epoch,
                    })
                    if extra_prior_kwargs_dict.get('wandb_log', False):
                        import wandb
                        wandb.log(_res_dict, step=len(master_epoch_count), commit=True)
                    if is_best:
                        best_res_dict = _res_dict
                        best_outputs = return_outputs
                        best_targets = return_targets
                    mstr = extra_prior_kwargs_dict.get('model_string')
                    boost_iter = f"ensemble_iter_{cur_boost_iter}" if is_ensemble else ""
                    log_path = os.path.join(extra_prior_kwargs_dict.get('save_path'),
                                            f'{mstr}_{boost_iter}_log_{epoch}.json')
                    # if not is_wrapper:
                    with open(log_path, 'w') as fi:
                        json.dump(_res_dict, fi, indent=4)

                if no_patience:
                    break

            # stepping with wallclock time based scheduler
            t_sched.step()

        if do_prompt_tuning and not do_kl_loss and not do_private:  # and isinstance(best_val_embed, torch.Tensor):
            t_model.prefix_embedding.weight = nn.Parameter(best_val_embed.to(device))
            # set requires grad to true
            t_model.prefix_embedding.weight.requires_grad = True
            t_optim = torch.optim.AdamW(t_model.parameters(), lr=lr, weight_decay=weight_decay)
            t_sched = scheduler(t_optim, warmup_epochs, epochs if epochs is not None else 100)
            v_scr, val_outputs, val_targets = real_data_eval(r_model=t_model, cl=real_data_qty,
                                                             train_data=data_for_fitting, real_data_val_dl=_val_dl)
            if (v_scr['Accuracy'] != best_res_dict['Val_Accuracy']) and verbose:
                print("WARNING: Best embedding score {} does not match best score {}!".format(v_scr, best_res_dict[
                    'Val_Accuracy']))

        return best_outputs, best_targets, best_res_dict

    # Search for max _bptt
    if extra_prior_kwargs_dict.get('bptt_search', False):
        backup_epochs = epochs
        epochs = 1
        backup_unif_bptt = extra_prior_kwargs_dict.get('uniform_bptt', False)
        extra_prior_kwargs_dict['uniform_bptt'] = True
        bptt_intervals = ([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
        stop = False
        for bptt_idx, bptt in enumerate(bptt_intervals):
            if verbose:
                print("Trying _bptt: ", bptt)
            try:
                dl, bptt = get_train_dataloader(dl.dataset, bptt=bptt, shuffle=True, num_workers=n_workers,
                                                drop_last=True, agg_k_grads=aggregate_k_gradients)
                with torch.no_grad():
                    total_loss, _, time_to_get_batch, forward_time, step_time, nan_share, _ =\
                        train_epoch(model, optimizer, False,
                                    eval_model=eval_model, bptt_search=True)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if verbose:
                        print(f"OOM with batch size {bptt}")
                    stop = True
                    search_idx = max(bptt_idx, 2)
                    bptt = bptt_intervals[search_idx - 2]
                    print("Setting _bptt to ", bptt)
                    dl, bptt = get_train_dataloader(dl.dataset, bptt=bptt, shuffle=True, num_workers=n_workers,
                                                    drop_last=True, agg_k_grads=aggregate_k_gradients)
                else:
                    raise e
            if stop:
                break
        epochs = backup_epochs
        extra_prior_kwargs_dict['uniform_bptt'] = backup_unif_bptt

    # main training loop
    bagging = extra_prior_kwargs_dict.get("bagging", False)
    split_indices_for_bagging = []
    dl_backup_before_bagging = None
    if bagging:
        split_size = extra_prior_kwargs_dict.get("subset_rows_bagging", 10000)
        if split_size == 0:
            if verbose:
                print("WARNING: subsampling was 0, using full dataset for bagging")
            split_size = len(dl.dataset)
        dl_backup_before_bagging = dl
        for i in range(boosting_n_iters):
            np.random.seed(extra_prior_kwargs_dict.get('rand_seed') + i)
            # NOTE: split sizes as absolute numbers
            split_indices_for_bagging.append(np.random.choice(np.arange(len(dl_backup_before_bagging.dataset)),
                                                              size=split_size, replace=True))
            # NOTE: split sizes as percentages of the dataset
            # split_size = 0.5
            # split_indices_for_bagging.append(np.random.choice(np.arange(len(dl_backup_before_bagging.dataset)),
            # size=int(split_size * len(dl_backup_before_bagging.dataset)), replace=False))
        # dl_backup_before_bagging = _dl
        # split_indices_for_bagging = np.array_split(np.arange(len(dl_backup_before_bagging.dataset)), boosting_n_iters)
    is_ensemble = (boosting or bagging or rand_init_ensemble)
    prefix_weights_l = []
    cur_boost_iter = 0
    # total_loss = float('inf')
    # total_positional_losses = float('inf')
    output_dict = {}
    i = 0
    ensembling_acc = dict()
    res_dict_ensemble = dict()
    best_results = dict()

    # ***
    # train/ENSEMBLING 1st loop
    # ***
    best_ens_acc = 0
    top_k_ens_key = ""
    topk_key = ""
    try:
        topk_key = extra_prior_kwargs_dict.get('topk_key', 'Val_Accuracy')
        if "nc_" in topk_key:
            top_k_ens_key = "Ens_" + topk_key.replace("nc_", "") + "_NC"
        else:
            top_k_ens_key = "Ens_" + topk_key
        print("Starting training loop \n \n")
        if bagging:
            subset_dataset = Subset(dl_backup_before_bagging.dataset, split_indices_for_bagging[i])
            dl, bptt = get_train_dataloader(subset_dataset,
                                            bptt=bptt,
                                            shuffle=True,
                                            num_workers=n_workers,
                                            drop_last=True,
                                            agg_k_grads=aggregate_k_gradients)
        prior_grad_dict = None
        gradient_dict = {}
        output_dict[i], test_targets, results_dict = train_test_loop(model, optimizer, sched_obj, eval_model, dl,
                                                                     val_dl, test_dl)
        res_dict_ensemble[i] = best_results = results_dict
        prior_grad_dict = gradient_dict
        # OUTPUT_DICT[0] contains val_outputs, test_outputs, val_outputs_nc, test_outputs_nc

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
            best_ens_acc = res_dict_ensemble[i][topk_key]
            ensembling_acc[i] = update_ensemble_acc(res_dict_ensemble[i]['Val_Accuracy'],
                                                    res_dict_ensemble[i]['Val_nc_Accuracy'],
                                                    res_dict_ensemble[i]['Test_Accuracy'],
                                                    res_dict_ensemble[i]['Test_nc_Accuracy'],
                                                    len(np.unique(labels_np)))
            if not do_concat:
                with open(os.path.join(extra_prior_kwargs_dict.get('save_path'), 'ensembling_acc.json'), 'w') as f:
                    json.dump(ensembling_acc, f, indent=4)
                if extra_prior_kwargs_dict.get('wandb_log', False):
                    import wandb
                    wandb.log(ensembling_acc[i], step=len(master_epoch_count), commit=True)
        if do_prompt_tuning:
            prefix_weights_l = save_prefix_weights(model, extra_prior_kwargs_dict.get('save_path'), i, do_concat,
                                                   prefix_weights_l)
    except KeyboardInterrupt:
        pass

    # ***
    # train/ENSEMBLING 2-nth loop
    # ***
    ens_patience = 0
    gradient_dict = {}
    if is_ensemble:
        for i in range(1, boosting_n_iters):
            next_seed = extra_prior_kwargs_dict.get('rand_seed') + i
            seed_all(next_seed)

            # extra_prior_kwargs_dict['rand_seed'] = next_seed

            if extra_prior_kwargs_dict.get('reseed_data', True):
                # reset subset maker
                # if getattr(dataset, "ssm", None) is not None:
                #     delattr(dataset, "ssm")
                # load data
                extra_prior_kwargs_dict['do_impute'] = np.random.choice([True, False])
                extra_prior_kwargs_dict['ohe'] = np.random.choice([True, False])
                extra_prior_kwargs_dict['preprocess_type'] = np.random.choice(
                    ['none', 'power_all', 'robust_all', 'quantile_all'])
                (x, y, x_val, y_val, x_test, y_test, invert_perm_map, steps_per_epoch, num_classes, label_weights,
                 train_ds, val_ds, test_ds) = make_datasets(
                    extra_prior_kwargs_dict, do_permute=not_zs, _bptt=bptt, _steps_per_epoch=steps_per_epoch,
                    is_wrapper=is_wrapper, do_private=private_data)
                old_bptt = bptt
                dl, val_dl, test_dl, bptt, data_for_fitting = make_dataloaders(_bptt=bptt)
                if old_bptt != bptt:
                    if verbose:
                        print("_bptt changed from {} to {}".format(old_bptt, bptt))
                    if extra_prior_kwargs_dict.get('uniform_bptt', False):
                        single_eval_pos_gen = lambda: np.random.randint(0, bptt)
                    else:
                        single_eval_pos_gen = bptt
                if bagging:
                    dl_backup_before_bagging = dl
            if bagging:
                subset_dataset = Subset(dl_backup_before_bagging.dataset, split_indices_for_bagging[i])
                dl = DataLoader(
                    subset_dataset, batch_size=bptt, shuffle=False, num_workers=n_workers, drop_last=True,
                )
            cur_boost_iter = i
            print("Ensembling iteration: ", i + 1, " of ", boosting_n_iters, "\n \n")
            model.init_prefix_weights()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            sched_obj = scheduler(optimizer, warmup_epochs, epochs if epochs is not None else 100)
            output_dict[i], test_targets, results_dict = train_test_loop(model, optimizer, sched_obj, eval_model, dl,
                                                                         val_dl, test_dl)
            res_dict_ensemble[i] = results_dict
            if do_prompt_tuning:
                prefix_weights_l = save_prefix_weights(model, extra_prior_kwargs_dict.get('save_path'), i, do_concat,
                                                       prefix_weights_l)
            prior_grad_dict = gradient_dict

            # No need to save ensembled results if we are concatenating; regular results are accurate
            if do_concat != "":
                continue

            current_outs = dict()
            current_preds = dict()
            boosting_accs = dict()
            topk_ens_val = extra_prior_kwargs_dict.get('keep_topk_ensemble', 0)
            if topk_ens_val > 0:
                if verbose:
                    print("keeping top {} of {} models, per provided key {}".format(topk_ens_val, i + 1, topk_key))
                # sort by val score
                sorted_res = sorted(res_dict_ensemble.items(), key=lambda _x: _x[1][topk_key], reverse=True)
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
                boosting_accs[m] = safe_round(correct / total)
            # TODO: this should not be hard-coded
            # OUTPUT_DICT[0] contains val_outputs, test_outputs, val_outputs_nc, test_outputs_nc
            probs_np = output_dict[0][0]
            labels_np = test_targets[0]
            probs_np_test = output_dict[0][1]
            labels_np_test = test_targets[1]
            if do_prompt_tuning:
                probs_np_nc = output_dict[0][2]
                labels_np_nc = test_targets[2]
                probs_np_nc_test = output_dict[0][3]
                labels_np_nc_test = test_targets[3]
            best_results = ensembling_acc[i] = update_ensemble_acc(boosting_accs[0],
                                                                   boosting_accs[2],
                                                                   boosting_accs[1],
                                                                   boosting_accs[3],
                                                                   len(np.unique(labels_np)))
            cur_ens_acc = ensembling_acc[i][top_k_ens_key]
            if cur_ens_acc > best_ens_acc:
                ens_patience = 0
                best_ens_acc = cur_ens_acc
            else:
                ens_patience += 1
            if do_prompt_tuning:
                prefix_weights_l = save_prefix_weights(model, extra_prior_kwargs_dict.get('save_path'), i, do_concat,
                                                       prefix_weights_l)
            # Save ensembled accuracy
            with open(os.path.join(extra_prior_kwargs_dict.get('save_path'), 'ensembling_acc.json'), 'w') as f:
                json.dump(ensembling_acc, f, indent=4)
            if extra_prior_kwargs_dict.get('wandb_log', False):
                import wandb
                master_epoch_count.append(1)
                wandb.log(ensembling_acc[i], step=len(master_epoch_count), commit=True)

            # Early stopping
            if ens_patience > extra_prior_kwargs_dict.get('early_stopping_patience', 2):
                print("Early stopping after {} ensembles".format(i))
                break

    # break down training and return
    if rank == 0:  # trivially true for non-parallel training
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            dl = None
        # todo: model_builder.py expects two outputs: model, results_dict
        return model, best_results, data_for_fitting, None

    return model, best_results, data_for_fitting, None


def _parse_args(_config_parser, _parser):
    # Do we have a config file to parse?
    args_config, remaining = _config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            _parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    remaining_args = _parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(remaining_args.__dict__, default_flow_style=False)
    return remaining_args, args_text


if __name__ == '__main__':
    config_parser = argparse.ArgumentParser(description='Only used as a first parser for the config file path.')
    config_parser.add_argument('--config')
    parser = argparse.ArgumentParser()
    parser.add_argument('prior')
    parser.add_argument('--loss_function', default='gaussnll')
    # Optional Arg's for `--loss_function barnll`
    parser.add_argument('--min_y', type=float, help='barnll can only model y in strict ranges, this is the minimum y '
                                                    'can take.')
    parser.add_argument('--max_y', type=float, help='barnll can only model y in strict ranges, this is the maximum y'
                                                    'can take.')
    parser.add_argument('--num_features', default=None, type=int, help='Specify depending on the prior (can be None).')
    # parser.add_argument('--num_features', default=None, type=int, help='Specify depending on the prior.')
    parser.add_argument("--extra_prior_kwargs_dict", default={}, dest="extra_prior_kwargs_dict",
                        action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL", help='Specify depending on the prior.')
    parser.add_argument('--encoder', default='linear', type=str, help='Specify depending on the prior.')
    parser.add_argument('--y_encoder', default='linear', type=str, help='Specify depending on the prior. You should '
                                                                        'specify this if you do not fuse x and y.')
    parser.add_argument('--pos_encoder', default='none', type=str, help='Specify depending on the prior.')
    parser.add_argument('--bptt', default=10, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', default=50, type=int)
    parser.add_argument('--validation_period', default=10, type=int)
    parser.add_argument('--permutation_invariant_max_eval_pos', default=None, type=int, help='Set this to an int to ')
    parser.add_argument('--permutation_invariant_sampling', default='weighted',
                        help="Only relevant if --permutation_invariant_max_eval_pos is set.")
    parser.add_argument('--train_mixed_precision', action='store_true')

    # these can likely be mostly left at defaults
    parser.add_argument('--emsize', default=512, type=int)  # sometimes even larger is better e.g. 1024
    parser.add_argument('--nlayers', default=6, type=int)
    parser.add_argument('--nhid', default=None, type=int)  # 2*emsize is the default
    parser.add_argument('--nhead', default=4, type=int)  # nhead = emsize / 64 in the original paper
    parser.add_argument('--dropout', default=.0, type=float)
    parser.add_argument('--steps_per_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--lr', '--learning_rate', default=.001,
                        type=float)  # try also .0003, .0001, go lower with lower batch size

    args, _ = _parse_args(config_parser, parser)

    if args.nhid is None:
        args.nhid = 2 * args.emsize

    prior = args.__dict__.pop('prior')

    if prior == 'gp':
        prior = priors.fast_gp.DataLoader
    # elif prior == 'ridge':
    #     prior = priors.ridge.DataLoader
    # elif prior == 'stroke':
    #     prior = priors.stroke.DataLoader
    # elif prior == 'mix_gp':
    #     prior = priors.fast_gp_mix.DataLoader
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


    def get_encoder_generator(_encoder):
        _encoder_generator = None
        if _encoder == 'linear':
            _encoder_generator = encoders.Linear
        elif _encoder == 'mlp':
            _encoder_generator = encoders.MLP
        elif _encoder == 'positional':
            _encoder_generator = encoders.Positional
        else:
            raise NotImplementedError(f'A {_encoder} encoder is not valid.')
        return _encoder_generator


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
          _y_encoder_generator=y_encoder_generator, _pos_encoder_generator=pos_encoder_generator,
          **args.__dict__)
