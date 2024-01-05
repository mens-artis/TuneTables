import time
from datetime import datetime
import argparse
import json
import os
import wandb

from scripts.model_builder import get_model, save_model
from scripts.model_configs import *
from priors.utils import uniform_int_sampler_f
from notebook_utils import *
from utils import get_wandb_api_key

os.environ["WANDB_API_KEY"] = "1f23ab4cce0ac3852702eca5e3a176732a0752a4"


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

    def no_callback(model, epoch, values_to_log):
        pass

    if config_sample['boosting'] or config_sample['rand_init_ensemble'] or config_sample['bagging']:
        my_callback = no_callback
    else:
        my_callback = save_callback

    # todo: get_model shouldn't be the method that trains the model
    model = get_model(config_sample
                      , config_sample["device"]
                      , should_train=True
                      , verbose=1
                      , state_dict=config_sample["state_dict"]
                      , epoch_callback = my_callback)
    
    return

def reload_config(config_type='causal', task_type='multiclass', longer=0):
    config = get_prior_config(config_type=config_type)
    
    model_string = ''
    
    #TODO: check this, it was set to true in the script; seems to have no effect on zs accuracy
    config['recompute_attn'] = True
    # config['recompute_attn'] = False

    config['max_num_classes'] = 10
    config['num_classes'] = uniform_int_sampler_f(2, config['max_num_classes'])
    config['balanced'] = False
    model_string = model_string + '_multiclass'
    
    model_string = model_string + '_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    return config, model_string

def train_loop():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--resume', type=str, default="./models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt", help='Path to model checkpoint to resume from.')
    parser.add_argument('--save_path', type=str, default="./logs", help='Path to save new checkpoints.')
    parser.add_argument('--prior_type', type=str, default="real", help='Type of prior to use (real, prior_bag).')
    parser.add_argument('--data_path', type=str, default=".", help='Path to data.')
    parser.add_argument('--prompt_tuning', action='store_true', help='Whether to tune the prompt.')
    parser.add_argument('--tuned_prompt_size', type=int, default=0, help='Size of the tuned prompt.')
    parser.add_argument('--tuned_prompt_label_balance', type=str, default='equal', help='Label balance for the tuned prompt (equal, proportional).')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
    parser.add_argument('--bptt', type=int, default=1152, help='Batch per train time.')
    parser.add_argument('--seed', type=int, default=135798642, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train for.')
    parser.add_argument('--num_eval_fitting_samples', type=int, default=1000, help='How many samples from the training set to draw when fitting the eval set.')
    parser.add_argument('--split', type=int, default=0, help='Which split to use (0-9?).')
    parser.add_argument('--boosting', action='store_true', help='Whether to use boosting.')
    parser.add_argument('--bagging', action='store_true', help='Whether to produce a bagged ensemble.')
    parser.add_argument('--rand_init_ensemble', action='store_true', help='Ensemble over random initialization.')
    parser.add_argument('--ensemble_lr', type=float, default=0.5, help='Additive learning factor for boosting / ensembling.')
    parser.add_argument('--ensemble_size', type=int, default=5, help='Number of ensemble members.')
    parser.add_argument('--ensemble_random_feature_rotation', action='store_true', help='Whether to randomly rotate features in the ensemble.')
    parser.add_argument('--aggregate_k_gradients', type=int, default=1, help='How many gradients to aggregate.')
    parser.add_argument('--average_ensemble', action='store_true', help='Whether to average the ensemble.')
    parser.add_argument('--permute_feature_position_in_ensemble', action='store_true', help='Whether to ensemble over feature position permutations.')
    parser.add_argument('--concat_method', type=str, default="", help='concatenation method (duplicate, empty = none)')
    parser.add_argument('--save_every_k_epochs', type=int, default=10, help='How often to save new checkpoints.')
    parser.add_argument('--validation_period', type=int, default=4, help='How often to validate.')
    parser.add_argument('--wandb_name', type=str, default='tabpfn_pt_airlines', help='Name for wandb logging.')
    parser.add_argument('--wandb_log', action='store_true', help='Whether to log to wandb.')
    parser.add_argument('--feature_subset_method', type=str, default='mutual_information', help='Method for feature subset selection ("mutual_information, random").')
    parser.add_argument('--pad_features', action='store_true', help='Whether to pad features to the maximum number of features.')
    parser.add_argument('--zs-eval-ensemble', type=int, default=0, help='Whether to do ensembled zero-shot evaluation.')
    parser.add_argument('--wandb_project_name', type=str, default="tabpfn", help='wandb project name')



    args = parser.parse_args()

    config, model_string = reload_config(longer=1)
    config['model_string'] = model_string
    config['prompt_tuning'] = args.prompt_tuning
    config['tuned_prompt_size'] = args.tuned_prompt_size
    config['tuned_prompt_label_balance'] = args.tuned_prompt_label_balance
    config['num_eval_fitting_samples'] = args.num_eval_fitting_samples
    config['split'] = args.split
    config['zs_eval_ensemble'] = args.zs_eval_ensemble
    config['pad_features'] = args.pad_features
    config['validation_period'] = args.validation_period

    # concatenation
    config['concat_method'] = args.concat_method

    if args.prior_type == 'prior_bag':
        config['prior_type'], config['differentiable'], config['flexible'] = 'prior_bag', True, True
    else:
        #TODO: check this
        config['prior_type'], config['differentiable'], config['flexible'] = args.prior_type, True, False

    config['data_path'] = args.data_path
    config['lr'] = args.lr

    if args.resume is not None:
        model_state, optimizer_state_load, config_sample_load = torch.load(args.resume, map_location='cpu')
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
    config["base_path"] = args.save_path

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
    config['ens_random_feature_rotation'] = args.ensemble_random_feature_rotation
    config['rotate_normalized_labels'] = False

    config["mix_activations"] = False # False heisst eig True

    config['emsize'] = 512
    config['nhead'] = config['emsize'] // 128
    config['bptt'] = args.bptt
    config['canonical_y_encoder'] = False
    config['rand_seed'] = args.seed

        
    config['aggregate_k_gradients'] = args.aggregate_k_gradients
    # config['batch_size'] = 8*config['aggregate_k_gradients']
    config['num_steps'] = 1024//config['aggregate_k_gradients']
    config['epochs'] = args.epochs
    config['total_available_time_in_s'] = None #60*60*22 # 22 hours for some safety...

    config['train_mixed_precision'] = True

    # Seems to have no effect on ZS accuracy
    # config['efficient_eval_masking'] = False
    config['efficient_eval_masking'] = True

    #TODO: check whether config_sample should be iterated within train_function
    config_sample = evaluate_hypers(config)

    config_sample['batch_size'] = args.batch_size
    config_sample['save_every_k_epochs'] = args.save_every_k_epochs

    #Boosting parameters
    config_sample['boosting'] = args.boosting
    config_sample['boosting_lr'] = args.ensemble_lr
    config_sample['boosting_n_iters'] = args.ensemble_size
    if config_sample['boosting']:
        config_sample['min_eval_pos'] = config_sample['max_eval_pos'] = config_sample['bptt'] = 1024

    #Random initialization of ensemble
    config_sample['rand_init_ensemble'] = args.rand_init_ensemble
    config_sample['average_ensemble'] = args.average_ensemble
    config_sample['permute_feature_position_in_ensemble'] = args.permute_feature_position_in_ensemble

    #Bagging parameters
    config_sample['bagging'] = args.bagging

    #Feature subset selection
    config_sample['subset_features'] = 100
    config_sample['subset_rows'] = -1
    config_sample['subset_features_method'] = args.feature_subset_method
    config_sample['subset_rows_method'] = 'random'

    # wandb
    # todo: for now, most are hard-coded
    config_sample['wandb_log'] = args.wandb_log
    config_sample['wandb_name'] = args.wandb_name
    config_sample['wandb_group'] = 'nyu-dice-lab'
    config_sample['wandb_project'] = args.wandb_project_name #'tabpfn-niv-try2'
    config_sample['wandb_entity'] = 'nyu-dice-lab'
    config_sample['wandb_log_test_interval'] = 1


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
    
    config_sample_copy = make_serializable(config_sample_copy)
        
    os.mkdir(f'{config_sample["base_path"]}/{model_string}')
    config_sample['base_path'] = f'{config_sample["base_path"]}/{model_string}'

    with open(f'{config_sample["base_path"]}/config_diff_real_{model_string}_n_{0}.json', 'w') as f:
        json.dump(config_sample_copy, f, indent=4)

    if config_sample['wandb_log']:
        wandb.login(key=get_wandb_api_key())
        #wandb.init(config=config_sample, name=config_sample['wandb_name'], group=config_sample['wandb_group'],
        #        project=config_sample['wandb_project'], entity=config_sample['wandb_entity'])
        wandb.init(config=config_sample, name=config_sample['wandb_name'], group=config_sample['wandb_group'],
                project=config_sample['wandb_project'], entity=config_sample['wandb_entity'])

    print("Training model ...")
    
    train_function(config_sample, 0, model_string)

    if config_sample['wandb_log']:
        wandb.finish()

    print("Done")

if __name__ == '__main__':
    train_loop()