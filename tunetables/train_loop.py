import time
from datetime import datetime
import json
import os

import wandb
import ConfigSpace

from tunetables.scripts.model_builder import get_model, save_model
from tunetables.scripts.model_configs import *
from tunetables.priors.utils import uniform_int_sampler_f
from tunetables.notebook_utils import *
from tunetables.utils import make_serializable, wandb_init

def train_function(config_sample, i=0, add_name='', is_wrapper = False, x_wrapper = None, y_wrapper = None, cat_idx = []):

    if config_sample['boosting'] or config_sample['rand_init_ensemble'] or config_sample['bagging']:
        #don't save checkpoints for ensembling, just prefixes
        save_every_k = config_sample['epochs'] + 1
    else:
        save_every_k = config_sample['save_every_k_epochs']
    epochs = []

    def save_callback(model, epoch, values_to_log):
        #NOTE: I think the 'epoch' value is actually 1 / config['epochs']
        epochs.append(epoch)
        if not hasattr(model, 'last_saved_epoch'):
            model.last_saved_epoch = 0
        if len(epochs) % save_every_k == 0:
            print('Saving model..')
            config_sample['epoch_in_training'] = epoch
            save_model(model, config_sample['base_path'], f'prior_diff_real_checkpoint{add_name}_n_{i}_epoch_{model.last_saved_epoch}.cpkt',
                           config_sample)
            model.last_saved_epoch = model.last_saved_epoch + 1 # TODO: Rename to checkpoint

    def no_callback(model, epoch, values_to_log):
        pass

    if config_sample['boosting'] or config_sample['rand_init_ensemble'] or config_sample['bagging']:
        my_callback = no_callback
    else:
        my_callback = save_callback

    #TODO: get_model shouldn't be the method that trains the model
    model, results_dict, data_for_fitting, test_loader = get_model(config_sample
                      , config_sample["device"]
                      , should_train=True
                      , state_dict=config_sample["state_dict"]
                      , epoch_callback = my_callback, is_wrapper = is_wrapper, x_wrapper = x_wrapper, y_wrapper = y_wrapper, cat_idx = cat_idx)
    
    if is_wrapper:
        return model, data_for_fitting, test_loader
    else:
        return results_dict

def set_compatibility_params(config, args):
    """
    The parameters listed here either are known to have no effect when using real data priors, or we don't know whether they have an effect.
    """

    # Evaluation parameters from original TabPFN code?

    # config["large_datasets"] = True
    # config["max_samples"] = 10000 if config["large_datasets"] else 5000
    # config["suite"]='cc'

    #Value set to true in the script; seems to have no effect on zs accuracy
    config['recompute_attn'] = True

    #parameters related to synthetic priors
    if args.prior_type == 'prior_bag':
        config['prior_type'], config['differentiable'], config['flexible'] = 'prior_bag', True, True
    else:
        #TODO: check this
        config['prior_type'], config['differentiable'], config['flexible'] = args.prior_type, True, False
    config['output_multiclass_ordered_p'] = 0.
    try:
        del config['differentiable_hyperparameters']['output_multiclass_ordered_p']
    except:
        pass
    try:
        del config['differentiable_hyperparameters']['multiclass_type']
    except:
        pass
    try:
            del config['differentiable_hyperparameters']['sampling']
    except:
        pass
    config['multiclass_type'] = 'rank'
    
    config['sampling'] = 'normal' # vielleicht schlecht?
    config['pre_sample_causes'] = True
    config['multiclass_loss_type'] = 'nono' # 'compatible'
    config['categorical_feature_p'] = .2 # diff: .0
    config['nan_prob_no_reason'] = .0
    config['nan_prob_unknown_reason'] = .0 # diff: .0
    config['set_value_to_nan'] = .1 # diff: 1.
    config['new_mlp_per_example'] = True
    config['prior_mlp_scale_weights_sqrt'] = True
    config['batch_size_per_gp_sample'] = None
    config['differentiable_hps_as_style'] = False
    config['normalize_ignore_label_too'] = False
    config["mix_activations"] = False # False heisst eig True
    config['multiclass_type'] = config['multiclass_type'] if 'multiclass_type' in config else 'rank'
    config['balanced'] = False
    config['eval_positions'] = [int(config['bptt'] * 0.95)] if config['bptt_extra_samples'] is None else [int(config['bptt'])]
    # ?
    config['canonical_y_encoder'] = False

    # Can't find where in the code where this is used -- would be useful if it worked
    config['total_available_time_in_s'] = None #60*60*22 # 22 hours for some safety...

    # Seems to have no effect on ZS accuracy
    config['efficient_eval_masking'] = True

    return config

def reload_config(config_type='causal', task_type='multiclass', longer=0, args=None):
    if config_type == 'real':
        config = {
        "dropout": 0.0,
        "emsize": 512,
        "nlayers": 12,
        "num_features": 100,
        "nhead": 4,
        "nhid_factor": 2,
        "eval_positions": None,
        "seq_len_used": args.bptt,
        "sampling": 'normal',
        "mix_activations": False,
        "pre_sample_causes": True,
        "multiclass_type": 'rank'
    }
    else:
        config = get_prior_config(config_type=config_type)
        
    #hard-coded limits of original TabPFN model
    config['max_num_classes'] = args.max_num_classes
    config["max_features"] = 100

    #prompt tuning
    config['prompt_tuning'] = args.prompt_tuning
    config['tuned_prompt_size'] = args.tuned_prompt_size
    config['tuned_prompt_label_balance'] = args.tuned_prompt_label_balance

    #eval fit samples and min batches per epoch
    config['num_eval_fitting_samples'] = args.num_eval_fitting_samples
    config['min_batches_per_epoch'] = args.min_batches_per_epoch

    # zs eval parameters
    config['zs_eval_ensemble'] = args.zs_eval_ensemble
    config['random_feature_rotation'] = True if config['zs_eval_ensemble'] > 0 else False
    config['rotate_normalized_labels'] = True if config['zs_eval_ensemble'] > 0 else False

    # core parameters
    config['lr'] = args.lr
    config['early_stopping_patience'] = args.early_stopping
    config['rand_seed'] = args.seed
    config['emsize'] = 512
    config['nhead'] = config['emsize'] // 128
    config['max_eval_pos'] = config['bptt'] = args.bptt
    config['batch_size'] = args.batch_size
    config['bptt_search'] = args.bptt_search
    config['aggregate_k_gradients'] = args.aggregate_k_gradients
    config['epochs'] = args.epochs
    config['warmup_epochs'] = args.epochs // 10
    if args.real_data_qty > 0:
        config['real_data_qty'] = args.real_data_qty

    # data preprocessing
    config['do_preprocess'] = args.do_preprocess
    config['preprocess_type'] = args.preprocess_type
    config['normalize_with_sqrt'] = False
    config['split'] = args.split
    config['pad_features'] = args.pad_features
    config['reseed_data'] = args.reseed_data
    config['normalize_to_ranking'] = False # This should be kept to false, it has learning from the future issues
    config['workers'] = args.workers

    #differential privacy
    config['private_model'] = args.private_model
    config['private_data'] = args.private_data
    config['epsilon'], config['delta'], config['gradnorm'] = float(args.edg[0]), float(args.edg[1]), float(args.edg[2])
    
    #meta-parameters
    config['validation_period'] = args.validation_period
    config['val_subset_size'] = args.val_subset_size
    config['verbose'] = args.verbose
    config['save_every_k_epochs'] = args.save_every_k_epochs
    config['max_time'] = args.max_time
    config['shuffle_every_epoch'] = args.shuffle_every_epoch
    config['topk_key'] = args.topk_key
    config['optuna_objective'] = args.optuna_objective

    # concatenation
    config['concat_method'] = args.concat_method

    #amp, cuda, paths
    config["device"] = 'cuda'
    config['data_path'] = args.data_path
    config["base_path"] = args.save_path
    config['train_mixed_precision'] = True

    if args.resume is not None:
        model_state, optimizer_state_load, config_sample_load = torch.load(args.resume, map_location='cpu')
        module_prefix = 'module.'
        config["state_dict"] = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    else:
        config["state_dict"] = None

    #Boosting parameters
    config['boosting'] = args.boosting
    config['boosting_lr'] = args.ensemble_lr
    config['boosting_n_iters'] = args.ensemble_size
    if config['boosting']:
        config['min_eval_pos'] = config['max_eval_pos'] = config['bptt'] = 1024
        config['aggregate_k_gradients'] = 1
    
    #Ensembling parameters
    config['rand_init_ensemble'] = args.rand_init_ensemble
    config['average_ensemble'] = args.average_ensemble
    config['permute_feature_position_in_ensemble'] = args.permute_feature_position_in_ensemble
    config['keep_topk_ensemble'] = args.keep_topk_ensemble

    #Bagging parameters
    config['bagging'] = args.bagging

    #BPTT and batch size
    config['uniform_bptt'] = args.uniform_bptt
    if config['uniform_bptt']:
        assert config['bptt'] % config['batch_size'] == 0, "bptt should be divisible by batch size when using uniform bptt"
        config['bptt_extra_samples'] = config['bptt']

        #NOTE: old logic
        # config['bptt_extra_samples'] = 128
        # if config['bptt'] < 128:
        #     print("Warning: bptt should be >= 128 when using uniform bptt, as currently 128 samples per batch are reserved for evaluation. Setting bptt to 128.")
        #     config['bptt'] = 128


    else:
        config['bptt_extra_samples'] = None

    #Feature subset selection
    config['subset_features'] = args.subset_features
    if args.bagging:
        config['subset_rows'] = 0
        config['subset_rows_bagging'] = args.subsampling
    else:
        config['subset_rows'] =  args.subsampling
        config['subset_rows_bagging'] = 0
    config['subset_features_method'] = args.subset_features_method
    config['subset_rows_method'] = 'random'


    #Preprocessing
    config['summerize_after_prep'] = args.summerize_after_prep

    #loss fn
    config['kl_loss'] = args.kl_loss

    # wandb
    # todo: for now, most are hard-coded
    config['wandb_log'] = args.wandb_log
    # config_sample['wandb_name'] = args.wandb_name
    config['wandb_group'] = args.wandb_group
    config['wandb_project'] = args.wandb_project
    config['wandb_entity'] = args.wandb_entity
    config['wandb_log_test_interval'] = args.validation_period

    config = set_compatibility_params(config, args)
    
    model_string = '_multiclass' + '_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    config['model_string'] = model_string

    return config, model_string

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--resume', type=str, default="../models/prior_diff_real_checkpoint_n_0_epoch_42.cpkt", help='Path to model checkpoint to resume from.')
    parser.add_argument('--save_path', type=str, default="./logs", help='Path to save new checkpoints.')
    parser.add_argument('--prior_type', type=str, default="real", help='Type of prior to use (real, prior_bag).')
    parser.add_argument('--data_path', type=str, default=".", help='Path to data.')
    parser.add_argument('--prompt_tuning', action='store_true', help='Whether to tune the prompt.')
    parser.add_argument('--tuned_prompt_size', type=int, default=0, help='Size of the tuned prompt.')
    parser.add_argument('--tuned_prompt_label_balance', type=str, default='equal', help='Label balance for the tuned prompt (equal, proportional).')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--bptt', type=int, default=1152, help='Batch per train time.')
    parser.add_argument('--bptt_search', action='store_true', help='Search for the near-maximum bptt that will fit in memory in range (32, 65536).')
    parser.add_argument('--uniform_bptt', action='store_true', help='Whether to use uniform bptt. Note that uniform bptt adds 128 extra samples per batch (for evaluation), so bptt should be > 128 when using uniform_bptt.')
    parser.add_argument('--seed', type=int, default=135798642, help='Random seed.')
    parser.add_argument('--early_stopping', type=int, default=2, help='Patience (for early stopping).')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for.')
    parser.add_argument('--num_eval_fitting_samples', type=int, default=1000, help='How many samples from the training set to draw when fitting the eval set.')
    parser.add_argument('--split', type=int, default=0, help='Which split to use (0-9?).')
    parser.add_argument('--boosting', action='store_true', help='Whether to use boosting.')
    parser.add_argument('--bagging', action='store_true', help='Whether to produce a bagged ensemble.')
    parser.add_argument('--subsampling', type=int, default=0, help='Qty of data to subsample during training (0 = no subsampling).')
    parser.add_argument('--rand_init_ensemble', action='store_true', help='Ensemble over random initialization.')
    parser.add_argument('--ensemble_lr', type=float, default=0.5, help='Additive learning factor for boosting / ensembling.')
    parser.add_argument('--ensemble_size', type=int, default=5, help='Number of ensemble members.')
    parser.add_argument('--reseed_data', action='store_true', help='Whether to randomly rotate features, labels and fitting data in the ensemble.')
    parser.add_argument('--aggregate_k_gradients', type=int, default=1, help='How many gradients to aggregate.')
    parser.add_argument('--average_ensemble', action='store_true', help='Whether to average the ensemble.')
    parser.add_argument('--permute_feature_position_in_ensemble', action='store_true', help='Whether to ensemble over feature position permutations.')
    parser.add_argument('--concat_method', type=str, default="", help='concatenation method (duplicate, empty = none)')
    parser.add_argument('--save_every_k_epochs', type=int, default=10, help='How often to save new checkpoints.')
    parser.add_argument('--validation_period', type=int, default=4, help='How often to validate on the entire val set.')
    parser.add_argument('--val_subset_size', type=int, default=2000, help='How many samples to use for fast validation.')
    parser.add_argument('--wandb_name', type=str, default='tt-airlines', help='Name for wandb logging.')
    parser.add_argument('--wandb_log', action='store_true', help='Whether to log to wandb.')
    parser.add_argument('--wandb_group', type=str, default='temp', help='Group for wandb logging.')
    parser.add_argument('--wandb_project', type=str, default='tt-dp', help='Project for wandb logging.')
    parser.add_argument('--wandb_entity', type=str, default='nyu-dice-lab', help='Entity for wandb logging.')
    parser.add_argument('--subset_features_method', type=str, default='mutual_information', help='Method for feature subset selection ("mutual_information, random, first, pca").')
    parser.add_argument('--subset_features', type=int, default=100, help='Number of features to use for feature subset selection.')
    parser.add_argument('--pad_features', action='store_true', help='Whether to pad features to the maximum number of features.')
    parser.add_argument('--do_preprocess', action='store_true', help='Whether to add tabpfn-style preprocessing to the data.')
    parser.add_argument('--zs-eval-ensemble', type=int, default=0, help='Whether to do ensembled zero-shot evaluation.')
    parser.add_argument('--min_batches_per_epoch', type=int, default=1, help='Minimum number of batches per epoch.')
    parser.add_argument('--keep_topk_ensemble', type=int, default=0, help='Whether to keep only the top-k ensemble members.')
    parser.add_argument('--topk_key', type=str, default='Val_Accuracy', help='Key to use for top-k ensemble selection.')
    parser.add_argument('--max_time', type=int, default=0, help='Maximum time to run for (in seconds).')
    parser.add_argument('--preprocess_type', type=str, default='none', help='Type of preprocessing to use (none, power_all, quantile_all, robust_all).')
    parser.add_argument('--optuna_objective', type=str, default='Val_Accuracy', help='Objective for optuna.')
    parser.add_argument('--verbose', action='store_true', help='Whether to print more information during training.')
    parser.add_argument('--shuffle_every_epoch', action='store_true', help='Whether to shuffle the order of the data every epoch (can help when bptt is large).')
    parser.add_argument('--max_num_classes', type=int, default=10, help='Maximum number of classes to use.')
    parser.add_argument('--real_data_qty', type=int, default=0, help='Number of real data samples to use for fitting.')
    parser.add_argument('--summerize_after_prep', action='store_true', help='train_feature_extractor.')
    parser.add_argument('--kl_loss', action='store_true', help='Whether to use KL loss.')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for data loading.')
    parser.add_argument('--private_model', action='store_true', help='Train model with differential privacy.')
    parser.add_argument('--private_data', action='store_true', help='Train with differential privacy added to the dataset.')
    parser.add_argument('--edg', nargs='+', type=str, default=["50", "1e-4", "1.2"], help="Epsilon, delta, gradnorm for differential privacy.")
    args = parser.parse_args()
    return args

def train_loop():
    args = parse_args()

    config, model_string = reload_config(config_type="real", longer=1, args=args)

    #TODO: check whether config_sample should be iterated within train_function
    # config_sample = evaluate_hypers(config, args)

    print("Saving config ...")
    simple_config = make_serializable(config.copy())
    os.mkdir(f'{config["base_path"]}/{model_string}')
    config['base_path'] = f'{config["base_path"]}/{model_string}'
    with open(f'{config["base_path"]}/config_diff_real_{model_string}_n_{0}.json', 'w') as f:
        json.dump(simple_config, f, indent=4)

    print("Training model ...")

    if config['wandb_log']:
        wandb_init(config, model_string)

    #clean out optuna params
    for k, v in config.items():
        if isinstance(v, ConfigSpace.hyperparameters.CategoricalHyperparameter):
            config[k] = v.default_value

    results_dict = train_function(config, 0, model_string, is_wrapper = False)

    if config['wandb_log']:
        wandb.finish()
    print("run complete")
    print("^RESULTS\n" + json.dumps(results_dict))
    return results_dict

if __name__ == '__main__':
    import signal
    import sys

    def signal_handler(sig, frame):
        signal.signal(sig, signal.SIG_IGN)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    _ = train_loop()