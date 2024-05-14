from datetime import datetime
import os
import json

import wandb
import optuna
import ConfigSpace

from tunetables.train_loop import parse_args, reload_config, train_function
from tunetables.utils import wandb_init

def objective(trial):
    args = parse_args()
    config, model_string = reload_config(longer=1, args=args)
    for k, v in config.items():
        if isinstance(v, ConfigSpace.hyperparameters.CategoricalHyperparameter):
            config[k] = trial.suggest_categorical(k, v.choices)

    #manually set optuna params
    for k, v in config.items():
        if isinstance(v, float) or isinstance(v, int) or isinstance(v, str):
            trial.set_user_attr(k, v)
    # config['bptt'] = trial.suggest_int('bptt', 128, 8192, log=True)
    config['lr'] = trial.suggest_float('lr', .1, .5)
    # config['aggregate_k_gradients'] = trial.suggest_int('aggregate_k_gradients', 1, 4)
    # config['feature_subset_method'] = trial.suggest_categorical('feature_subset_method', ['pca', 'mutual_information'])
    config['preprocess_type'] = trial.suggest_categorical('preprocess_type', ['none', 'power_all', 'quantile_all'])
    config['epochs'] = trial.suggest_int('epochs', 1, 101, step=10)
    config['early_stopping'] = config['epochs']
    config['tuned_prompt_label_balance'] = trial.suggest_categorical('label_balance', ['proportional', 'equal'])
    config['wandb_group'] = config['wandb_group'] + "_epochs_preprocess_lr"
    print("Training model ...")

    if config['wandb_log']:
        wandb_init(config, model_string)
    try:
        results_dict = train_function(config, 0, model_string)
    except Exception as e:
        print("Exception during optuna trial: ", e)
        if config['wandb_log']:
                wandb.finish()
        return 0.0

    if config['wandb_log']:
        wandb.finish()

    return results_dict[args.optuna_objective]


if __name__ == '__main__':
    sampler = optuna.samplers.TPESampler(seed=13579)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=30)
    # Print best trial
    trial = study.best_trial
    current_time = '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")    
    # Save study results

    res_dict = study.best_params
    res_dict['best_value'] = trial.value
    os.makedirs('logs/optuna', exist_ok=True)
    with open(f'logs/optuna/optuna_results_{current_time}.json', 'w') as f:
        json.dump(res_dict, f)