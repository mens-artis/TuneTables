from collections import defaultdict
import copy

base_dict = {
    'prior_type' : 'real',
    'pad_features' : '',
    'resume' : '/cmlscratch/vcherepa/TabPFN-pt/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt',
    'epochs' : 31,
    'validation_period' : 3,
    'save_every_k_epochs' : 15,
    'aggregate_k_gradients' : 1,
    'lr' : 1e-3,
    'feature_subset_method' : 'pca',
    'wandb_log' : '',
    'do_preprocess' : '',
    'verbose' : '',
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

#Prompt tuning with 10 prompts
pt10_dict = dict(base_dict, **pt10_dict)
pt10_dict['save_every_k_epochs'] = base_dict['epochs'] + 1
pt10_short_dict = copy.deepcopy(pt10_dict)
pt10_short_dict['epochs'] = 7
pt10_short_dict['save_every_k_epochs'] = 12
pt10_short_dict['early_stopping'] = 7
pt10_short_dict['validation_period'] = 3
pt10_short_dict['lr'] = 0.2
pt10_shorter_dict = copy.deepcopy(pt10_dict)
pt10_shorter_dict['epochs'] = 4
pt10_shorter_dict['save_every_k_epochs'] = 5
pt10_shorter_dict['early_stopping'] = 5
pt10_shorter_dict['validation_period'] = 3
pt10_shorter_dict['lr'] = 0.3
pt10_short_unif_dict = copy.deepcopy(pt10_short_dict)
pt10_short_unif_dict['uniform_bptt'] = ''
pt10_shorter_unif_dict = copy.deepcopy(pt10_shorter_dict)
pt10_shorter_unif_dict['uniform_bptt'] = ''
pt10_unif_dict = copy.deepcopy(pt10_dict)
pt10_unif_dict['uniform_bptt'] = ''
pt10_prop_dict = copy.deepcopy(pt10_dict)
pt10_prop_dict['tuned_prompt_label_balance'] = 'proportional'
pt10_pca_dict = copy.deepcopy(pt10_dict)
pt10_pca_dict['feature_subset_method'] = 'pca'
pt10_mut_dict = copy.deepcopy(pt10_dict)
pt10_mut_dict['feature_subset_method'] = 'mutual_information'
pt10_rand_dict = copy.deepcopy(pt10_dict)
pt10_rand_dict['feature_subset_method'] = 'random'
pt10_powerall_dict = copy.deepcopy(pt10_dict)
pt10_powerall_dict['preprocess_type'] = 'power_all'

#Prompt tuning with 100 prompts
pt100_dict = copy.deepcopy(pt10_dict)
pt100_dict['tuned_prompt_size'] = 100
pt100_short_dict = copy.deepcopy(pt100_dict)
pt100_short_dict['epochs'] = 7
pt100_short_dict['save_every_k_epochs'] = 12
pt100_short_dict['early_stopping'] = 7
pt100_short_dict['validation_period'] = 3
pt100_short_dict['lr'] = 0.2
pt100_shorter_dict = copy.deepcopy(pt100_dict)
pt100_shorter_dict['epochs'] = 4
pt100_shorter_dict['save_every_k_epochs'] = 5
pt100_shorter_dict['early_stopping'] = 5
pt100_shorter_dict['validation_period'] = 3
pt100_shorter_dict['lr'] = 0.3
pt100_short_unif_dict = copy.deepcopy(pt100_short_dict)
pt100_short_unif_dict['uniform_bptt'] = ''
pt100_shorter_unif_dict = copy.deepcopy(pt100_shorter_dict)
pt100_shorter_unif_dict['uniform_bptt'] = ''
pt100_unif_dict = copy.deepcopy(pt100_dict)
pt100_unif_dict['uniform_bptt'] = ''
pt100_prop_dict = copy.deepcopy(pt100_dict)
pt100_prop_dict['tuned_prompt_label_balance'] = 'proportional'
pt100_pca_dict = copy.deepcopy(pt100_dict)
pt100_pca_dict['feature_subset_method'] = 'pca'
pt100_mut_dict = copy.deepcopy(pt100_dict)
pt100_mut_dict['feature_subset_method'] = 'mutual_information'
pt100_rand_dict = copy.deepcopy(pt100_dict)
pt100_rand_dict['feature_subset_method'] = 'random'
pt100_powerall_dict = copy.deepcopy(pt100_dict)
pt100_powerall_dict['preprocess_type'] = 'power_all'

#Prompt tuning with 1000 prompts
pt1000_dict = copy.deepcopy(pt10_dict)
pt1000_dict['tuned_prompt_size'] = 1000
# #
# pt1000_dict["lr"]=0.19404061382273632
# pt1000_dict["feature_subset_method"]="mutual_information"
# pt1000_dict["preprocess_type"] = "none"
# pt1000_dict["early_stopping"] = 3
# pt1000_dict["tuned_prompt_label_balance"] = "proportional"

pt1000_short_dict = copy.deepcopy(pt1000_dict)
pt1000_short_dict['epochs'] = 7
pt1000_short_dict['save_every_k_epochs'] = 12
pt1000_short_dict['early_stopping'] = 7
pt1000_short_dict['validation_period'] = 3
pt1000_short_dict['lr'] = 0.2
pt1000_shorter_dict = copy.deepcopy(pt1000_dict)
pt1000_shorter_dict['epochs'] = 4
pt1000_shorter_dict['save_every_k_epochs'] = 5
pt1000_shorter_dict['early_stopping'] = 5
pt1000_shorter_dict['validation_period'] = 3
pt1000_shorter_dict['lr'] = 0.3
pt1000_short_unif_dict = copy.deepcopy(pt1000_short_dict)
pt1000_short_unif_dict['uniform_bptt'] = ''
pt1000_shorter_unif_dict = copy.deepcopy(pt1000_shorter_dict)
pt1000_shorter_unif_dict['uniform_bptt'] = ''
pt1000_unif_dict = copy.deepcopy(pt1000_dict)
pt1000_unif_dict['uniform_bptt'] = ''
pt1000_unif_highep_lowlr_dict = copy.deepcopy(pt1000_unif_dict)
pt1000_unif_highep_lowlr_dict['epochs'] = 101
pt1000_unif_highep_lowlr_dict['lr'] = 0.05
pt1000_unif_highep_lowlr_dict['save_every_k_epochs'] = 31
pt1000_unif_highep_lowlr_dict['validation_period'] = 5
pt1000_unif_highep_lowlr_dict['early_stopping'] = 10
pt1000_prop_dict = copy.deepcopy(pt1000_dict)
pt1000_prop_dict['tuned_prompt_label_balance'] = 'proportional'
pt1000_pca_dict = copy.deepcopy(pt1000_dict)
pt1000_pca_dict['feature_subset_method'] = 'pca'
pt1000_mut_dict = copy.deepcopy(pt1000_dict)
pt1000_mut_dict['feature_subset_method'] = 'mutual_information'
pt1000_rand_dict = copy.deepcopy(pt1000_dict)
pt1000_rand_dict['feature_subset_method'] = 'random'
pt1000_powerall_dict = copy.deepcopy(pt1000_dict)
pt1000_powerall_dict['preprocess_type'] = 'power_all'

#Ensemble presets
ens_bagged_avg_dict = dict(pt1000_dict, **ens_bagged_avg_dict)
ens_randinit_avg_dict = copy.deepcopy(ens_bagged_avg_dict)
ens_randinit_avg_dict.pop('bagging')
ens_randinit_avg_dict['rand_init_ensemble'] = ''
ens_randinit_avg_dict = dict(pt1000_dict, **ens_randinit_avg_dict)
ens_bagged_avg_top2_dict = copy.deepcopy(ens_bagged_avg_dict)
ens_bagged_avg_top2_dict['keep_topk_ensemble'] = 2
ens_randinit_avg_top2_dict = copy.deepcopy(ens_randinit_avg_dict)
ens_randinit_avg_top2_dict['keep_topk_ensemble'] = 2
ens_bagged_avg_top2_powerall_dict = copy.deepcopy(ens_bagged_avg_top2_dict)
ens_bagged_avg_top2_powerall_dict['preprocess_type'] = 'power_all'
ens_bagged_avg_top2_reseed_dict = copy.deepcopy(ens_bagged_avg_top2_dict)
ens_bagged_avg_top2_reseed_dict['reseed_data'] = ''
ens_bagged_avg_top2_reseed_dict['ensemble_size'] = 10
ens_randinit_avg_top2_reseed_dict = copy.deepcopy(ens_randinit_avg_top2_dict)
ens_randinit_avg_top2_reseed_dict['reseed_data'] = ''
ens_randinit_avg_top2_reseed_dict['ensemble_size'] = 10
ens_randinit_avg_top2_unif_reseed_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_dict)
ens_randinit_avg_top2_unif_reseed_dict['uniform_bptt'] = ''

all_tasks = {
    'ft' : base_dict,
    'pt10': pt10_dict,
    'pt10-prop' : pt10_prop_dict,
    'pt10-pca' : pt10_pca_dict,
    'pt10-mut' : pt10_mut_dict,
    'pt10-rand' : pt10_rand_dict,
    'pt10-uniform' : pt10_unif_dict,
    'pt10-powerall' : pt10_powerall_dict,
    'pt10-short' : pt10_short_dict,
    'pt10-shorter' : pt10_shorter_dict,
    'pt10-uniform-short' : pt10_short_unif_dict,
    'pt10-uniform-shorter' : pt10_shorter_unif_dict,
    'pt100': pt100_dict,
    'pt100-prop' : pt100_prop_dict,
    'pt100-pca' : pt100_pca_dict,
    'pt100-mut' : pt100_mut_dict,
    'pt100-rand' : pt100_rand_dict,
    'pt100-uniform' : pt100_unif_dict,
    'pt100-powerall' : pt100_powerall_dict,
    'pt100-short' : pt100_short_dict,
    'pt100-shorter' : pt100_shorter_dict,
    'pt100-uniform-short' : pt100_short_unif_dict,
    'pt100-uniform-shorter' : pt100_shorter_unif_dict,
    'pt1000': pt1000_dict,
    'pt1000-prop' : pt1000_prop_dict,
    'pt1000-pca' : pt1000_pca_dict,
    'pt1000-mut' : pt1000_mut_dict,
    'pt1000-rand' : pt1000_rand_dict,
    'pt1000-uniform' : pt1000_unif_dict,
    'pt1000-uniform-highep-lowlr' : pt1000_unif_highep_lowlr_dict,
    'pt1000-powerall' : pt1000_powerall_dict,
    'pt1000-short' : pt1000_short_dict,
    'pt1000-shorter' : pt1000_shorter_dict,
    'pt1000-uniform-short' : pt1000_short_unif_dict,
    'pt1000-uniform-shorter' : pt1000_shorter_unif_dict,
    'pt1000-5ens-bagged-avg' : ens_bagged_avg_dict,
    'pt1000-5ens-randinit-avg' : ens_randinit_avg_dict,
    'pt1000-5ens-bagged-avg-top2' : ens_bagged_avg_top2_dict,
    'pt1000-5ens-bagged-avg-top2-powerall' : ens_bagged_avg_top2_powerall_dict,
    'pt1000-5ens-randinit-avg-top2' : ens_randinit_avg_top2_dict,
    'pt1000-10ens-bagged-avg-top2-reseed' : ens_bagged_avg_top2_reseed_dict,
    'pt1000-10ens-randinit-avg-top2-reseed' : ens_randinit_avg_top2_reseed_dict,
    'pt1000-10ens-randinit-avg-top2-unif-reseed' : ens_randinit_avg_top2_unif_reseed_dict,
}

def get_all_tasks():
    return all_tasks