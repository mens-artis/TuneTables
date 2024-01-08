from collections import defaultdict
import copy

base_dict = {
    'prior_type' : 'real',
    'pad_features' : '',
    'resume' : '/home/benfeuer/TabPFN-pt/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt',
    'epochs' : 31,
    'validation_period' : 3,
    'save_every_k_epochs' : 15,
    'aggregate_k_gradients' : 1,
    'lr' : 1e-3,
    'feature_subset_method' : 'mutual_information',
    'wandb_log' : '',
}

pt10_dict = {
        'prompt_tuning' : '',
        'tuned_prompt_size' : 10,
        'lr' : 0.1,
}

ens_bagged_avg_dict = {
        'epochs': 61,
        'save_every_k_epochs' : 62,
        'bagging' : '',
        'ensemble_lr' : 1.0,
        'average_ensemble' : '',
        'ensemble_size' : 5,
}

pt10_dict = dict(base_dict, **pt10_dict)
pt10_dict['save_every_k_epochs'] = base_dict['epochs'] + 1
pt100_dict = copy.deepcopy(pt10_dict)
pt100_dict['tuned_prompt_size'] = 100
pt100_prop_dict = copy.deepcopy(pt100_dict)
pt100_prop_dict['tuned_prompt_label_balance'] = 'proportional'
pt100_pca_dict = copy.deepcopy(pt100_dict)
pt100_pca_dict['feature_subset_method'] = 'pca'
pt100_rand_dict = copy.deepcopy(pt100_dict)
pt100_rand_dict['feature_subset_method'] = 'random'
pt1000_dict = copy.deepcopy(pt10_dict)
pt1000_dict['tuned_prompt_size'] = 1000

ens_bagged_avg_dict = dict(pt1000_dict, **ens_bagged_avg_dict)
ens_bagged_avg_top2_dict = ens_bagged_avg_dict
ens_bagged_avg_top2_dict['keep_topk_ensemble'] = 2

all_tasks = {
    'ft' : base_dict,
    'pt10' : pt10_dict,
    'pt100': pt100_dict,
    'pt100-prop' : pt100_prop_dict,
    'pt100-pca' : pt100_pca_dict,
    'pt100-rand' : pt100_rand_dict,
    'pt1000': pt1000_dict,
    'pt1000-5ens-bagged-avg' : ens_bagged_avg_dict,
    'pt1000-5ens-bagged-avg-top2' : ens_bagged_avg_top2_dict,
}

def get_all_tasks():
    return all_tasks