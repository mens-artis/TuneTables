import argparse
import os
from pathlib import Path
from datetime import datetime
import time
import warnings

import pandas as pd
import torch
import wandb
import uncertainty_metrics.numpy as um
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

from tunetables.utils import get_wandb_api_key
from tunetables.scripts import tabular_baselines
from tunetables.scripts.tabular_baselines import *
from tunetables.scripts.tabular_evaluation import evaluate
from tunetables.scripts.tabular_metrics import calculate_score, make_ranks_and_wins_table, make_metric_matrix
from tunetables.scripts import tabular_metrics
from tunetables.scripts.baseline_prediction_interface import baseline_predict
from tunetables.priors.real import TabularDataset
from tunetables.priors.real import process_data

def eval_method(splits, device, method, cat_idx, metric_used, max_time=300):

    clf = clf_dict[method]

    # def xgb_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300, no_tune=None, gpu_id=None)
    # def catboost_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300, no_tune=None, gpu_id=None)
    # return metric, pred, best
    eval_sets_x = torch.cat([splits[1][0], splits[2][0]])
    eval_sets_y = torch.cat([splits[1][1], splits[2][1]])

    if method in ['random_forest', 'lightgbm', 'autogluon', 'autosklearn2', 'cocktail', 'knn', 'logistic', 'gp']:
        _, outputs, best_configs = clf(
            splits[0][0],
            splits[0][1],
            eval_sets_x,
            eval_sets_y,
            cat_idx,
            metric_used,
        )
    else:
        _, outputs, best_configs = clf(
            splits[0][0],
            splits[0][1],
            eval_sets_x,
            eval_sets_y,
            cat_idx,
            metric_used,
            gpu_id=device,
        )
    
    return outputs, best_configs

def run_eval(dataset_name, base_path, max_time, epsilon, local_args):

    def run_and_get_metrics():
        results = dict()
        start_time = time.time()
        outputs, best_configs = eval_method(splits, device, method, [], metric_used, max_time=args.max_time)
        end_time = time.time()
        run_time = end_time - start_time
        results[f'Run_Time'] = np.round(run_time, 3).item()
        outputs = outputs[:, 0:num_classes]
        predictions = np.argmax(outputs, axis=1)

        # Divide outputs and predictions into val and test splits
        val_outputs = outputs[:len(y_val)]
        val_predictions = predictions[:len(y_val)]
        test_outputs = outputs[len(y_val):]
        test_predictions = predictions[len(y_val):]

        results[f'Val_Accuracy'] = np.round(accuracy_score(y_val, val_predictions), 3).item()
        results[f'Val_Log_Loss'] = np.round(log_loss(y_val, val_outputs, labels=np.arange(num_classes)), 3).item()
        results[f'Val_F1_Weighted'] = np.round(f1_score(y_val, val_predictions, average='weighted'), 3).item()
        results[f'Val_F1_Macro'] = np.round(f1_score(y_val, val_predictions, average='macro'), 3).item()
        try:
            if num_classes == 2:
                results['Val_ROC_AUC'] = np.round(roc_auc_score(y_val, val_outputs[:, 1], labels=np.arange(num_classes)), 3).item()
            else:
                results['Val_ROC_AUC'] = np.round(roc_auc_score(y_val, val_outputs, labels=np.arange(num_classes), multi_class='ovr'), 3).item()
        except Exception as e:
            print("Error calculating ROC AUC: ", e)
            results['Val_ROC_AUC'] = 0.0
        results['Val_ECE'] = np.round(um.ece(y_val, val_outputs, num_bins=30), 3).item()
        results['Val_TACE'] = np.round(um.tace(y_val, val_outputs, num_bins=30), 3).item()
        results[f'Test_Accuracy'] = np.round(accuracy_score(y_test, test_predictions), 3).item()
        results[f'Test_Log_Loss'] = np.round(log_loss(y_test, test_outputs, labels=np.arange(num_classes)), 3).item()
        results[f'Test_F1_Weighted'] = np.round(f1_score(y_test, test_predictions, average='weighted'), 3).item()
        results[f'Test_F1_Macro'] = np.round(f1_score(y_test, test_predictions, average='macro'), 3).item()
        try:
            if num_classes == 2:
                results['Test_ROC_AUC'] = np.round(roc_auc_score(y_test, test_outputs[:, 1], labels=np.arange(num_classes)), 3).item()
            else:
                results['Test_ROC_AUC'] = np.round(roc_auc_score(y_test, test_outputs, labels=np.arange(num_classes), multi_class='ovr'), 3).item()
        except Exception as e:
            print("Error calculating ROC AUC: ", e)
            results['Test_ROC_AUC'] = 0.0
        results['Test_ECE'] = np.round(um.ece(y_test, test_outputs, num_bins=30), 3).item()
        results['Test_TACE'] = np.round(um.tace(y_test, test_outputs, num_bins=30), 3).item()
        if isinstance(best_configs, pd.DataFrame) or isinstance(best_configs, pd.Series):
            save_path = os.path.join(base_path, model_string + ".csv")
            best_configs.to_csv(save_path)
            wandb.save(save_path)
        elif isinstance(best_configs, dict) and best_configs.get('best') is not None:
            best_configs = dict(best_configs, **{f"Best_Config_{k}" : v for k, v in best_configs['best'].items()})
            #drop key 'best'
            best_configs.pop('best')
            results = dict(results, **best_configs)
        return results

    print("Running evaluation for dataset: ", dataset_name)

    metrics = [tabular_metrics.auc_metric, tabular_metrics.cross_entropy]

    # This is the metric used for fitting the models
    metric_used = tabular_metrics.auc_metric
    # todo: abacus branch has:     methods = ['random_forest', 'logistic', 'catboost', 'xgb', 'autosklearn2']
    #     # 'lightgbm', 'cocktail', 'autogluon', 'gp', 'knn'
    #     # methods = ['autogluon']
    #     # methods = ['knn']
    #
    # methods = ['random_forest', 'lightgbm', 'cocktail', 'logistic', 'gp', 'knn', 'catboost', 'xgb', 'autosklearn2', 'autogluon']
    methods = ['autogluon']
    # methods = ['knn']
    device = '0'

    config = dict()
    config['dataset'] = dataset_name
    config['base_path'] = base_path
    config['max_time'] = max_time
    config['metric_used'] = str(metric_used)
    config['device'] = device
    config['methods'] = ", ".join(str(x) for x in methods)
    config['subset_features'] = -1
    config['subset_rows'] = -1
    # todo abacus brnach has:
    #  merge config and vars(local_args)
    #  config.update(vars(local_args))
    args = argparse.Namespace(**config)

    # Get dataset
    dataset_path = os.path.join(base_path, dataset_name)

    # # get metadata
    # metadata_path = os.path.join(dataset_path, 'metadata.json')

    # with open(metadata_path) as f:
    #     metadata = json.load(f)
    if epsilon is None:
        args.private_data = False
        dataset = TabularDataset.read(Path(dataset_path).resolve())
    else:
        args.private_data = True
        args.epsilon = epsilon
        epsilon_vals = [epsilon]
        dataset = TabularDataset.read(p=Path(dataset_path).resolve(),
                                        epsilon_vals=epsilon_vals)

    # todo: main had:     dataset = TabularDataset.read(Path(dataset_path).resolve())
    for i, split_dictionary in enumerate(dataset.split_indeces):
        # todo: next 2 lines are from abacus branch
        if i not in args.splits:
            continue
        # TODO: make stopping index a hyperparameter
        train_index = split_dictionary["train"]
        val_index = split_dictionary["val"]
        test_index = split_dictionary["test"]

        # run pre-processing & split data (list of numpy arrays of length num_ensembles)
        processed_data = process_data(
            dataset,
            train_index,
            val_index,
            test_index,
            verbose=False,
            scaler="None",
            one_hot_encode=False,
            args=args,
        )

        X_train, y_train = processed_data["data_train"]
        X_val, y_val = processed_data["data_val"]
        X_test, y_test = processed_data["data_test"]

        #convert numpy arrays to torch tensors
        X_train = torch.from_numpy(X_train.astype(np.float32)).float()
        y_train = torch.from_numpy(y_train.astype(np.int32)).long()
        X_val = torch.from_numpy(X_val.astype(np.float32)).float()
        y_val = torch.from_numpy(y_val.astype(np.int32)).long()
        X_test = torch.from_numpy(X_test.astype(np.float32)).float()
        y_test = torch.from_numpy(y_test.astype(np.int32)).long()

        splits = [[X_train, y_train], [X_val, y_val], [X_test, y_test]]


        # NOTE: Categorical features have already been pre-processed, will this harm CatBoost perf?
        cat_features = dataset.cat_idx

        # Run evaluation
        for method in methods:
            # if method == 'lightgbm':
            #     metric_used = tabular_metrics.cross_entropy
            # else:
            #     metric_used = tabular_metrics.auc_metric
            config['method'] = method
            config['split'] = i
            config['n_configs'] = 100
            model_string = f"{dataset_name}" + '_' + f"{method}" + '_split_' + f"{i}"
            if args.privacy_sweep:
                model_string += f'_eps_{epsilon}'
            wandb.login(key=get_wandb_api_key())
            # todo: main had:
            # wandb.init(config=config, name=model_string, group='baselines',
            #                 project='tt-dp', entity='nyu-dice-lab')
            #
            wandb.init(config=config, name=model_string, group='baselines-0415',
                project='baselines-0415', entity='nyu-dice-lab')
            num_classes = len(np.unique(y_train))
            # todo: abacus branch wants to delete to line 265

            # if num_classes == 2 and method == 'lightgbm':
            #     #convert to 1-class problem
            #     print("y_train: ", y_train[:10])
            #     print("shape: ", y_train.shape)
            #     y_train = y_train == 1
            #     print("y_train: ", y_train[:10])
            #     print("shape: ", y_train.shape)
            #     raise NotImplementedError("LightGBM binary classification not implemented")
            results = dict()
            # try:
            start_time = time.time()
            outputs, best_configs = eval_method(splits, device, method, [], metric_used, max_time=config['max_time'])
            end_time = time.time()
            run_time = end_time - start_time
            results[f'Run_Time'] = np.round(run_time, 3).item()
            # except Exception as e:
            #     print("Error running method: ", e)
            #     wandb.finish()
            #     continue

            outputs = outputs[:, 0:num_classes]
            #numpy softmax
            # outputs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
            # # assert outputs sum to 1
            # assert np.allclose(np.sum(outputs, axis=1), np.ones(outputs.shape[0])), "Outputs do not sum to 1"

            predictions = np.argmax(outputs, axis=1)

            # Divide outputs and predictions into val and test splits
            val_outputs = outputs[:len(y_val)]
            val_predictions = predictions[:len(y_val)]
            test_outputs = outputs[len(y_val):]
            test_predictions = predictions[len(y_val):]

            results[f'Val_Accuracy'] = np.round(accuracy_score(y_val, val_predictions), 3).item()
            results[f'Val_Log_Loss'] = np.round(log_loss(y_val, val_outputs, labels=np.arange(num_classes)), 3).item()
            results[f'Val_F1_Weighted'] = np.round(f1_score(y_val, val_predictions, average='weighted'), 3).item()
            results[f'Val_F1_Macro'] = np.round(f1_score(y_val, val_predictions, average='macro'), 3).item()
            ####
            try:
                # todo: opcaus branch has: results = run_and_get_metrics()
                if num_classes == 2:
                    results['Val_ROC_AUC'] = np.round(roc_auc_score(y_val, val_outputs[:, 1], labels=np.arange(num_classes)), 3).item()
                else:
                    results['Val_ROC_AUC'] = np.round(roc_auc_score(y_val, val_outputs, labels=np.arange(num_classes), multi_class='ovr'), 3).item()
            except Exception as e:
                print("Error calculating ROC AUC: ", e)
                results['Val_ROC_AUC'] = 0.0
            # results['Val_ECE'] = np.round(um.ece(y_val, val_outputs, num_bins=30), 3).item()
            results['Val_TACE'] = np.round(um.tace(y_val, val_outputs, num_bins=30), 3).item()
            results[f'Test_Accuracy'] = np.round(accuracy_score(y_test, test_predictions), 3).item()
            results[f'Test_Log_Loss'] = np.round(log_loss(y_test, test_outputs, labels=np.arange(num_classes)), 3).item()
            results[f'Test_F1_Weighted'] = np.round(f1_score(y_test, test_predictions, average='weighted'), 3).item()
            results[f'Test_F1_Macro'] = np.round(f1_score(y_test, test_predictions, average='macro'), 3).item()
            try:
                if num_classes == 2:
                    results['Test_ROC_AUC'] = np.round(roc_auc_score(y_test, test_outputs[:, 1], labels=np.arange(num_classes)), 3).item()
                else:
                    results['Test_ROC_AUC'] = np.round(roc_auc_score(y_test, test_outputs, labels=np.arange(num_classes), multi_class='ovr'), 3).item()
            except Exception as e:
                print("Error calculating ROC AUC: ", e)
                results['Test_ROC_AUC'] = 0.0
            results['Test_ECE'] = np.round(um.ece(y_test, test_outputs, num_bins=30), 3).item()
            results['Test_TACE'] = np.round(um.tace(y_test, test_outputs, num_bins=30), 3).item()
            if isinstance(best_configs, pd.DataFrame) or isinstance(best_configs, pd.Series):
                save_path = os.path.join(base_path, model_string + ".csv")
                best_configs.to_csv(save_path)
                wandb.save(save_path)
            elif isinstance(best_configs, dict) and best_configs.get('best') is not None:
                best_configs = dict(best_configs, **{f"Best_Config_{k}" : v for k, v in best_configs['best'].items()})
                #drop key 'best'
                best_configs.pop('best')
                results = dict(results, **best_configs)
            wandb.log(results)
            wandb.finish()

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./tunetables/data')
    parser.add_argument('--datasets', type=str, default='./tunetables/metadata/subset.txt', help='Path to datasets text file')
    parser.add_argument('--max_time', type=int, default=300, help='Allowed run time (in seconds)')
    parser.add_argument('--privacy_sweep', action='store_true', help='Run privacy sweep')
    parser.add_argument('--rand_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--private_val', action='store_true', help='Use private validation set')
    parser.add_argument('--splits', nargs='+', type=int, default=[0], help='Splits to run')

    args = parser.parse_args()

    with open(args.datasets) as f:
        datasets = f.readlines()
    for dataset in datasets:
        dataset = dataset.strip()
        epsilon = None
        if args.privacy_sweep:
            for tgt_eps in ["0.01", "0.05", "0.1", "0.5", "1.0"]:
                run_eval(dataset, args.dataset_path, args.max_time, tgt_eps, args)
        else:
            run_eval(dataset, args.dataset_path, args.max_time, epsilon, args)