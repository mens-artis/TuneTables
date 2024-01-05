import time
from datetime import datetime
import argparse
import json
import os
import wandb
import optuna

from scripts.model_builder import get_model, save_model
from scripts.model_configs import *
from priors.utils import uniform_int_sampler_f
from notebook_utils import *
from utils import get_wandb_api_key

def train_function(config_sample, i=0, add_name=''):

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

        # save to wandb
        if config_sample['wandb_log'] and len(epochs) % config_sample['wandb_log_test_interval'] == 0:
            wandb.log(values_to_log, step=len(epochs), commit=True)

    def no_callback(model, epoch):
        pass

    if config_sample['boosting'] or config_sample['rand_init_ensemble'] or config_sample['bagging']:
        my_callback = no_callback
    else:
        my_callback = save_callback

    # todo: get_model shouldn't be the method that trains the model
    model, results_dict = get_model(config_sample, 
                              config_sample["device"], 
                              should_train=True, 
                              verbose=1, 
                              state_dict=config_sample["state_dict"], 
                              epoch_callback=my_callback)

    return results_dict

def reload_config(config_type='causal', task_type='multiclass', longer=0):
    config = get_prior_config(config_type=config_type)
    
    model_string = ''
    
    #TODO: check this, it was set to true in the script
    config['recompute_attn'] = True

    config['max_num_classes'] = 10
    config['num_classes'] = uniform_int_sampler_f(2, config['max_num_classes'])
    config['balanced'] = False
    model_string = model_string + '_multiclass'
    
    model_string = model_string + '_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    return config, model_string

def objective(trial):
    config, model_string = reload_config(longer=1)

    config['bptt'] = trial.suggest_int('bptt', 128, 8192, log=True)
    config['lr'] = trial.suggest_float('lr', .0001, .3)
    config['aggregate_k_gradients'] = trial.suggest_int('aggregate_k_gradients', 1, 2)
    config['tuned_prompt_label_balance'] = trial.suggest_categorical('tuned_prompt_label_balance', ['equal', 'proportional'])
    config['pad_features'] = trial.suggest_categorical('pad_features', [False, True])
    config['ens_random_feature_rotation'] = trial.suggest_categorical('ens_random_feature_rotation', [False, True])
    config['efficient_eval_masking'] = trial.suggest_categorical('efficient_eval_masking', [False, True])
    #config_sample['subset_features_method'] = trial.suggest_categorical('subset_features_method', ['random', 'first', 'mutual_information'])

    config['subset_features_method'] = 'mutual_information'
    config['subset_features'] = 100
    config['subset_rows'] = -1
    config['subset_rows_method'] = 'random'

    config['zs_eval_ensemble'] = False
    config['model_string'] = model_string
    config['prompt_tuning'] = True
    config['tuned_prompt_size'] = 1000
    config['num_eval_fitting_samples'] = 1000
    config['split'] = 0
    config['data_path'] = "/home/colin/TabPFN-pt/tabpfn/data/openml__airlines__189354"
    config['concat_method'] = ""
    config['validation_period'] = 10

    prior_type = "real"
    if prior_type == 'prior_bag':
        config['prior_type'], config['differentiable'], config['flexible'] = 'prior_bag', True, True
    else:
        #TODO: check this
        config['prior_type'], config['differentiable'], config['flexible'] = 'real', True, False

    resume = "/home/colin/TabPFN-pt/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
    if resume is not None:
        model_state, optimizer_state_load, config_sample_load = torch.load(resume, map_location='cpu')
        module_prefix = 'module.'
        config["state_dict"] = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    else:
        config["state_dict"] = None

    config['bptt_extra_samples'] = None

    # config["large_datasets"] = True
    # config["max_samples"] = 10000 if config["large_datasets"] else 5000
    # config["bptt"] = 10000 if config["large_datasets"] else 3000
    # config["suite"]='cc'
    # config["max_features"] = 100

    config["device"] = 'cuda'
    config["base_path"] = "./logs"

    # diff
    config['output_multiclass_ordered_p'] = 0.
    del config['differentiable_hyperparameters']['output_multiclass_ordered_p']

    config['multiclass_type'] = 'rank'
    del config['differentiable_hyperparameters']['multiclass_type']

    config['sampling'] = 'normal' # vielleicht schlecht?
    del config['differentiable_hyperparameters']['sampling']

    config['pre_sample_causes'] = True
    # end diff

    config['multiclass_loss_type'] = 'nono' # 'compatible'
    config['normalize_to_ranking'] = False # False

    config['categorical_feature_p'] = .2 # diff: .0

    # turn this back on in a random search!?
    config['nan_prob_no_reason'] = .0
    config['nan_prob_unknown_reason'] = .0 # diff: .0
    config['set_value_to_nan'] = .1 # diff: 1.

    config['normalize_with_sqrt'] = False

    config['new_mlp_per_example'] = True
    config['prior_mlp_scale_weights_sqrt'] = True
    config['batch_size_per_gp_sample'] = None

    config['normalize_ignore_label_too'] = False

    config['differentiable_hps_as_style'] = False
    config['max_eval_pos'] = 1000

    config['random_feature_rotation'] = False
    config['rotate_normalized_labels'] = False

    config["mix_activations"] = False # False heisst eig True

    config['emsize'] = 512
    config['nhead'] = config['emsize'] // 128
    config['canonical_y_encoder'] = False
    config['rand_seed'] = 135798642

    config['num_steps'] = 1024//config['aggregate_k_gradients']
    config['epochs'] = 31
    config['total_available_time_in_s'] = None #60*60*22 # 22 hours for some safety...

    config['train_mixed_precision'] = True
    config['efficient_eval_masking'] = True

    config['batch_size'] = 4
    config['save_every_k_epochs'] = 30

    #Boosting parameters
    config['boosting'] = False
    config['boosting_lr'] = 0.5
    config['boosting_n_iters'] = 5

    if config['boosting']:
        print('warning: bptt is being reset')
        config['min_eval_pos'] = config['max_eval_pos'] = config['bptt'] = 1024

    #Random initialization of ensemble
    config['rand_init_ensemble'] = False
    config['average_ensemble'] = False
    config['permute_feature_position_in_ensemble'] = False

    #Bagging parameters
    config['bagging'] = False

    # wandb
    # todo: for now, most are hard-coded
    config['wandb_log'] = True
    config['wandb_name'] = 'optuna2_airlines_{}'.format(trial.number)
    config['wandb_group'] = 'abacus'
    config['wandb_project'] = 'tabpfn'
    config['wandb_entity'] = 'crwhite14'
    config['wandb_log_test_interval'] = 1


    #TODO: check whether config_sample should be iterated within train_function
    config_sample = evaluate_hypers(config)

    print("Saving config ...")
    config_sample_copy = config_sample.copy()

    def make_serializable(config_sample):
        if isinstance(config_sample, torch.Tensor):
            config_sample = "tensor"
        if isinstance(config_sample, dict):
            config_sample = {k: make_serializable(config_sample[k]) for k in config_sample}
        if isinstance(config_sample, list):
            config_sample = [make_serializable(v) for v in config_sample]
        if callable(config_sample):
            config_sample = str(config_sample)
        return config_sample

    # todo: do some optuna stuff

    config_sample_copy = make_serializable(config_sample_copy)
    
    os.mkdir(f'{config_sample["base_path"]}/{model_string}')
    config_sample['base_path'] = f'{config_sample["base_path"]}/{model_string}'

    with open(f'{config_sample["base_path"]}/config_diff_real_{model_string}_n_{0}.json', 'w') as f:
        json.dump(config_sample_copy, f, indent=4)

    if config_sample['wandb_log']:
        wandb.login(key=get_wandb_api_key())
        wandb.init(config=config_sample, name=config_sample['wandb_name'], group=config_sample['wandb_group'],
                project=config_sample['wandb_project'], entity=config_sample['wandb_entity'])

    print("Training model ...")

    # run training routine
    try:
        results_dict = train_function(config_sample, 0, model_string)
    except:
        results_dict = {'val_score': 0}

    if config_sample['wandb_log']:
        wandb.finish()
    print("Done")
    return results_dict['val_score']


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1000)