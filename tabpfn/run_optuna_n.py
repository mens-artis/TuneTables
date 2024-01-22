from datetime import datetime
import json
import wandb
import optuna
import ConfigSpace

from train_loop import parse_args, reload_config, train_function
# from scripts.model_builder import get_model, save_model
# from scripts.model_configs import *
# from priors.utils import uniform_int_sampler_f
# from notebook_utils import *
from utils import wandb_init

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
    config['lr'] = trial.suggest_float('lr', .0001, .3)
    #config['aggregate_k_gradients'] = trial.suggest_int('aggregate_k_gradients', 1, 1)
    config['feature_subset_method'] = trial.suggest_categorical('feature_subset_method', ['pca', 'mutual_information', 'random'])
    config['preprocess_type'] = trial.suggest_categorical('preprocess_type', ['none', 'power_all', 'quantile_all', 'robust_all'])
    config['early_stopping'] = trial.suggest_int('early_stopping', 2, 6)
    config['tuned_prompt_label_balance'] = trial.suggest_categorical('label_balance', ['proportional', 'equal'])

    while config['bptt'] % config['aggregate_k_gradients'] != 0:
        config['aggregate_k_gradients'] -= 1
 
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

    print(results_dict)
    return results_dict['Val_Accuracy'], results_dict['Val_Demographic parity']


if __name__ == '__main__':
    sampler = optuna.samplers.TPESampler(seed=13579)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(directions=['maximize', 'minimize'], sampler=sampler)
    study.optimize(objective, n_trials=10)
    # Print best trial
    trials = study.best_trials
    current_time = '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")    
    # Save study results

    # trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[0])
    # print(f"Trial with highest accuracy: ")
    # print(f"\tnumber: {trial_with_highest_accuracy.number}")
    # print(f"\tparams: {trial_with_highest_accuracy.params}")
    # print(f"\tvalues: {trial_with_highest_accuracy.values}")
    #
    # trial_with_highest_fairness = max(study.best_trials, key=lambda t: t.values[1])
    # print(f"Trial with highest accuracy: ")
    # print(f"\tnumber: {trial_with_highest_fairness.number}")
    # print(f"\tparams: {trial_with_highest_fairness.params}")
    # print(f"\tvalues: {trial_with_highest_fairness.values}")

    res_dict = study.best_params
    res_dict['best_value'] = trial.value

    with open(f'logs/optuna_results_{current_time}.json', 'w') as f:
        json.dump(res_dict, f)