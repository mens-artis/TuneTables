import argparse
import os
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import torch
import wandb
# import pandas as pd

import uncertainty_metrics.numpy as um
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from diffprivlib.models import LogisticRegression, RandomForestClassifier

from tunetables.priors.real import TabularDataset, process_data
from tunetables.scripts.onedrive import od_sizes
from tunetables.train import safe_round
# from tunetables.scripts.onedrive import onedrive
from tunetables.utils import get_wandb_api_key


def eval_method(splits, method, eval_args):
    """
    Every time the model is trained with .fit(), a
    different model is produced due to the randomness of 
    differential privacy. The accuracy will therefore change, 
    even if it's re-trained with the same training data. 
    Try it for yourself to find out!
    """
    classifier = None
    if method == "dp-logreg":
        # https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/models/logistic_regression.py
        classifier = LogisticRegression(epsilon=eval_args.epsilon,
                                        max_iter=1000,
                                        tol=1e-3,
                                        random_state=eval_args.seed,
                                        fit_intercept=True)
    elif method == "dp-random-forest":
        # https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/models/forest.py
        classifier = RandomForestClassifier(n_estimators=1000,
                                            epsilon=eval_args.epsilon,
                                            random_state=eval_args.seed,
                                            )

    # Train model
    classifier.fit(splits[0][0], splits[0][1])
    val_outputs = classifier.predict_proba(splits[1][0])
    test_outputs = classifier.predict_proba(splits[2][0])
    val_outputs, test_outputs = val_outputs[:, 0:eval_args.num_classes], test_outputs[:, 0:eval_args.num_classes]
    val_predictions, test_predictions = np.argmax(val_outputs, axis=1), np.argmax(test_outputs, axis=1)
    return val_outputs, val_predictions, test_outputs, test_predictions


def run_eval(dataset_name, eval_args):
    print("Running eval for dataset: ", dataset_name)
    # dataset_path = os.path.join(eval_args.dataset_path, dataset_name)
    dataset_path = dataset_name
    eval_dataset = TabularDataset.read(Path(dataset_path).resolve(), "")
    #                                       epsilon_vals=["0.01"])  # , "0.05", "0.1", "0.5", "1.0"])
    for i, split_dictionary in enumerate(eval_dataset.split_indeces):
        # TODO: make stopping index a hyperparameter
        train_index = split_dictionary["train"]
        val_index = split_dictionary["val"]
        test_index = split_dictionary["test"]

        # logger.debug("running pre-processing & split data (list of numpy arrays of length num_ensembles)")
        # run pre-processing & split data (list of numpy arrays of length num_ensembles)
        processed_data = process_data(
            eval_dataset,
            train_index,
            val_index,
            test_index,
            verbose=False,
            scaler="None",
            one_hot_encode=False,
            args=eval_args,
        )
        x_train, y_train = processed_data["data_train"]
        eval_args.num_classes = len(np.unique(y_train))
        x_val, y_val = processed_data["data_val"]
        x_test, y_test = processed_data["data_test"]

        # convert numpy arrays to torch tensors
        x_train = torch.from_numpy(x_train.astype(np.float32)).float()
        y_train = torch.from_numpy(y_train.astype(np.int32)).long()
        x_val = torch.from_numpy(x_val.astype(np.float32)).float()
        y_val = torch.from_numpy(y_val.astype(np.int32)).long()
        x_test = torch.from_numpy(x_test.astype(np.float32)).float()
        y_test = torch.from_numpy(y_test.astype(np.int32)).long()

        splits = [[x_train, y_train], [x_val, y_val], [x_test, y_test]]

        # NOTE: Categorical features have already been pre-processed, will this harm CatBoost perf?
        cat_features = eval_dataset.cat_idx

        # Run evaluation
        for method in eval_args.methods:
            config = vars(eval_args)
            config['method'] = method
            config['split'] = i
            config['n_configs'] = 100
            model_string = f"{dataset_name}" + '_' + f"{method}" + '_split_' + f"{i}" + '_' + datetime.now().strftime(
                "%m_%d_%Y_%H_%M_%S")
            try:
                wandb.login(key=get_wandb_api_key())
                wandb.init(config=config, name=model_string, group='baselines',
                           project='tt-dp', entity='nyu-dice-lab')
            except FileNotFoundError as ex:
                "Should I get a wandb API key? | pass {}".format(ex)
            results = dict()
            # try:
            start_time = time.time()
            eval_args.epsilon = 0.01
            eval_args.delta = 1.0
            val_outputs, val_predictions, test_outputs, test_predictions = eval_method(splits, method, eval_args)
            end_time = time.time()
            run_time = end_time - start_time
            results[f'Run_Time'] = safe_round(run_time)

            # METRICS
            results[f'Val_Accuracy'] = safe_round(accuracy_score(y_val, val_predictions))
            results[f'Val_Log_Loss'] = safe_round(log_loss(y_val, val_outputs, labels=np.arange(eval_args.num_classes)))
            results[f'Val_F1_Weighted'] = safe_round(f1_score(y_val, val_predictions, average='weighted'))
            results[f'Val_F1_Macro'] = safe_round(f1_score(y_val, val_predictions, average='macro'))
            try:
                if eval_args.num_classes == 2:
                    results['Val_ROC_AUC'] = safe_round(roc_auc_score(y_val, val_outputs[:, 1],
                                                                      labels=np.arange(eval_args.num_classes)))
                else:
                    results['Val_ROC_AUC'] = safe_round(roc_auc_score(y_val, val_outputs,
                                                                      labels=np.arange(eval_args.num_classes),
                                                                      multi_class='ovr'))
            except Exception as e:
                print("Error calculating ROC AUC: ", e)
                results['Val_ROC_AUC'] = 0.0
            try:
                results['Val_ECE'] = "uncrtianty metrics bug"
                results['Val_ECE'] = safe_round(um.ece(y_val, val_outputs, num_bins=30))
                results['Val_TACE'] = "uncrtianty metrics bug"
                results['Val_TACE'] = safe_round(um.tace(y_val, val_outputs, num_bins=30))
            except Exception as e:
                print("Error calculating ECE: ", e)
                results['Val_ECE'] = 0.0
                results['Val_TACE'] = 0.0
            results[f'Test_Accuracy'] = safe_round(accuracy_score(y_test, test_predictions))
            results[f'Test_Log_Loss'] = safe_round(
                log_loss(y_test, test_outputs, labels=np.arange(eval_args.num_classes)))
            results[f'Test_F1_Weighted'] = safe_round(f1_score(y_test, test_predictions, average='weighted'))
            results[f'Test_F1_Macro'] = safe_round(f1_score(y_test, test_predictions, average='macro'))
            try:
                if eval_args.num_classes == 2:
                    results['Test_ROC_AUC'] = safe_round(
                        roc_auc_score(y_test, test_outputs[:, 1], labels=np.arange(eval_args.num_classes)))
                else:
                    results['Test_ROC_AUC'] = safe_round(
                        roc_auc_score(y_test, test_outputs, labels=np.arange(eval_args.num_classes), multi_class='ovr'))
            except Exception as e:
                print("Error calculating ROC AUC: ", e)
                results['Test_ROC_AUC'] = 0.0
            try:
                results['Test_ECE'] = "uncrtianty metrics bug"
                # np.round(um.ece(y_test, test_outputs, num_bins=30), 3).item()
                results['Test_TACE'] = "uncrtianty metrics bug"
                # np.round(um.tace(y_test, test_outputs, num_bins=30), 3).item()
            except Exception as e:
                print("Error calculating ECE: ", e)
                results['Test_ECE'] = 0.0
                results['Test_TACE'] = 0.0
            # if isinstance(best_configs, pd.DataFrame) or isinstance(best_configs, pd.Series):
            #     save_path = os.path.join(base_path, model_string + ".csv")
            #     best_configs.to_csv(save_path)
            #     wandb.save(save_path)
            # elif isinstance(best_configs, dict) and best_configs.get('best') is not None:
            #     best_configs = dict(best_configs, **{f"Best_Config_{k}" : v for k, v in best_configs['best'].items()})
            #     #drop key 'best'
            #     best_configs.pop('best')
            #     results = dict(results, **best_configs)
            # todo
            #  wandb.log(results)
            #  wandb.finish()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--datasets', type=str, default='../metadata/small_datasets.txt',
                        help='Path to datasets text file')
    parser.add_argument('--max_time', type=int, default=300, help='Allowed run time (in seconds)')
    # parser.add_argument('--epsilon', type=float, default=1.0, help='Epsilon for differential privacy')
    # parser.add_argument('--epsilon', type=float, default=["0.01", "0.05", "0.1", "0.5", "1.0"],
    # help='Epsilon for differential privacy')
    parser.add_argument('--epsilon', type=str, default="0.01", help='Epsilon for differential privacy')

    # private_val

    parser.add_argument('--private_data', type=bool, default=False, help="has th to do with DP or GEM")
    parser.add_argument('--methods', nargs='+', type=str, default=["dp-random-forest", "dp-logreg"],
                        help="List of methods to evaluate")
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--subset_features', type=int, default=0, help='Number of features to subset')
    parser.add_argument('--subset_rows', type=int, default=0, help='Number of samples to subset')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    args = parser.parse_args()

    # p = Path(args.dataset_path).resolve()
    # open(p)
    # with open(args.dataset_path) as f:
    sets_sorted_by_size = TabularDataset.size("f", args)
    # live (non-syncing) OD DS download
    # d ="https://onedrive.live.com/?authkey=%21AJIqEu%2D6EfnU4pk&id=B6996B25210D10CC%21277824&cid=B6996B25210D10CC"
    # onedrive(d, "-w")
    #        od_sizes()

    with open(args.datasets) as f:
        datasets = f.readlines()
    # for dataset in sets_sorted_by_size:  # datasets:
    continue_from_here = False
    for size, dataset in sets_sorted_by_size:  # datasets:
        # dataset = dataset.strip()
        print(size, dataset)
        if 'openml__anneal__2867' in str(dataset):
            continue_from_here = True
            continue
        if continue_from_here:
            try:
                run_eval(dataset, args)
            except Exception as ex:
                pass
