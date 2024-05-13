import copy

base_dict = {
    'prior_type' : 'real',
    'pad_features' : '',
    'epochs' : 101,
    'validation_period' : 2,
    'save_every_k_epochs' : 15,
    'aggregate_k_gradients' : 1,
    'early_stopping' : 5,
    'lr' : 1e-3,
    'subset_features_method' : 'pca',
    'wandb_log' : '',
    'do_preprocess' : '',
    'max_time' : 36000,
    'workers' : 4,
}

pt10_dict = {
        'prompt_tuning' : '',
        'save_every_k_epochs' : 102,
        'tuned_prompt_size' : 10,
        'lr' : 0.1,
}

#combines with pt1000 dict
ens_bagged_avg_dict = {
        'bagging' : '',
        'ensemble_lr' : 1.0,
        'average_ensemble' : '',
        'ensemble_size' : 10,
}

#Prompt tuning with 2 prompts

pt2_dict = dict(base_dict, **pt10_dict)
pt2_dict['tuned_prompt_size'] = 2
pt2_unif_dict = copy.deepcopy(pt2_dict)
pt2_unif_dict['uniform_bptt'] = ''
pt2_unif_dict['topk_key'] = 'Val_nc_Accuracy'
pt2_unif_kl_dict = copy.deepcopy(pt2_unif_dict)
pt2_unif_kl_dict['kl_loss'] = ''

# Prompt tuning with 5 prompts

pt5_dict = dict(base_dict, **pt10_dict)
pt5_dict['tuned_prompt_size'] = 5
pt5_unif_dict = copy.deepcopy(pt5_dict)
pt5_unif_dict['uniform_bptt'] = ''
pt5_unif_dict['topk_key'] = 'Val_nc_Accuracy'
pt5_unif_kl_dict = copy.deepcopy(pt5_unif_dict)
pt5_unif_kl_dict['kl_loss'] = ''
pt5_unif_kl_nopp_dict = copy.deepcopy(pt5_unif_kl_dict)
pt5_unif_kl_nopp_dict.pop('do_preprocess')
pt5_unif_kl_nopp_lowlr_dict = copy.deepcopy(pt5_unif_kl_nopp_dict)
pt5_unif_kl_nopp_lowlr_dict['lr'] = 0.01
pt5_unif_kl_prop_dict = copy.deepcopy(pt5_unif_kl_dict)
pt5_unif_kl_prop_dict['tuned_prompt_label_balance'] = 'proportional'

#Prompt tuning with 10 prompts
pt10_dict = dict(base_dict, **pt10_dict)
pt10_dict['save_every_k_epochs'] = base_dict['epochs'] + 1
pt10_short_dict = copy.deepcopy(pt10_dict)
pt10_short_dict['epochs'] = 7
pt10_short_dict['save_every_k_epochs'] = 12
pt10_short_dict['early_stopping'] = 7
pt10_short_dict['validation_period'] = 3
pt10_short_dict['lr'] = 0.2
pt10_short_lowlr_dict = copy.deepcopy(pt10_short_dict)
pt10_short_lowlr_dict['lr'] = 0.03
pt10_short_lowlr_prop_dict = copy.deepcopy(pt10_short_lowlr_dict)
pt10_short_lowlr_prop_dict['tuned_prompt_label_balance'] = 'proportional'
pt10_short_lowlr_prop_pca_dict = copy.deepcopy(pt10_short_lowlr_prop_dict)
pt10_short_lowlr_prop_pca_dict['subset_features_method'] = 'pca'
pt10_short_lowlr_prop_sumafter_pca_dict = copy.deepcopy(pt10_short_lowlr_prop_pca_dict)
pt10_short_lowlr_prop_sumafter_pca_dict['summerize_after_prep'] = ''
pt10_short_lowlr_prop_mutinf_dict = copy.deepcopy(pt10_short_lowlr_prop_dict)
pt10_short_lowlr_prop_mutinf_dict['subset_features_method'] = 'mutual_information'
pt10_short_lowlr_prop_sumafter_mutinf_dict = copy.deepcopy(pt10_short_lowlr_prop_mutinf_dict)
pt10_short_lowlr_prop_sumafter_mutinf_dict['summerize_after_prep'] = ''
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
pt10_unif_dict['topk_key'] = 'Val_nc_Accuracy'
pt10_unif_kl_dict = copy.deepcopy(pt10_unif_dict)
pt10_unif_kl_dict['kl_loss'] = ''
pt10_unif_kl_dict['early_stopping'] = 2
pt10_unif_kl_nopp_dict = copy.deepcopy(pt10_unif_kl_dict)
pt10_unif_kl_nopp_dict.pop('do_preprocess')
pt10_unif_kl_nopp_pca_dict = copy.deepcopy(pt10_unif_kl_nopp_dict)
pt10_unif_kl_nopp_pca_dict['subset_features_method'] = 'pca'
pt10_unif_kl_nopp_sumafter_pca_dict = copy.deepcopy(pt10_unif_kl_nopp_pca_dict)
pt10_unif_kl_nopp_sumafter_pca_dict['summerize_after_prep'] = ''
pt10_unif_kl_nopp_mutinf_dict = copy.deepcopy(pt10_unif_kl_nopp_dict)
pt10_unif_kl_nopp_mutinf_dict['subset_features_method'] = 'mutual_information'
pt10_unif_kl_nopp_sumafter_mutinf_dict = copy.deepcopy(pt10_unif_kl_nopp_mutinf_dict)
pt10_unif_kl_nopp_sumafter_mutinf_dict['summerize_after_prep'] = ''
pt10_unif_kl_prop_dict = copy.deepcopy(pt10_unif_kl_dict)
pt10_unif_kl_prop_dict['tuned_prompt_label_balance'] = 'proportional'
pt10_unif_kl_nopp_prop_dict = copy.deepcopy(pt10_unif_kl_nopp_dict)
pt10_unif_kl_nopp_prop_dict['tuned_prompt_label_balance'] = 'proportional'
pt10_unif_kl_nopp_prop_pca_dict = copy.deepcopy(pt10_unif_kl_nopp_prop_dict)
pt10_unif_kl_nopp_prop_pca_dict['subset_features_method'] = 'pca'
pt10_unif_kl_nopp_prop_sumafter_pca_dict = copy.deepcopy(pt10_unif_kl_nopp_prop_pca_dict)
pt10_unif_kl_nopp_prop_sumafter_pca_dict['summerize_after_prep'] = ''
pt10_unif_kl_nopp_prop_mutinf_dict = copy.deepcopy(pt10_unif_kl_nopp_prop_dict)
pt10_unif_kl_nopp_prop_mutinf_dict['subset_features_method'] = 'mutual_information'
pt10_unif_kl_nopp_prop_sumafter_mutinf_dict = copy.deepcopy(pt10_unif_kl_nopp_prop_mutinf_dict)
pt10_unif_kl_nopp_prop_sumafter_mutinf_dict['summerize_after_prep'] = ''
pt10_prop_dict = copy.deepcopy(pt10_dict)
pt10_prop_dict['tuned_prompt_label_balance'] = 'proportional'
pt10_pca_dict = copy.deepcopy(pt10_dict)
pt10_pca_dict['subset_features_method'] = 'pca'
pt10_mut_dict = copy.deepcopy(pt10_dict)
pt10_mut_dict['subset_features_method'] = 'mutual_information'
pt10_rand_dict = copy.deepcopy(pt10_dict)
pt10_rand_dict['subset_features_method'] = 'random'
pt10_powerall_dict = copy.deepcopy(pt10_dict)
pt10_powerall_dict['preprocess_type'] = 'power_all'
pt10_sumafter_pca_dict = copy.deepcopy(pt10_pca_dict)
pt10_sumafter_pca_dict['summerize_after_prep'] = ''
pt10_sumafter_pca_unif_dict = copy.deepcopy(pt10_sumafter_pca_dict)
pt10_sumafter_pca_unif_dict['uniform_bptt'] = ''
pt10_sumafter_mutinf_dict = copy.deepcopy(pt10_mut_dict)
pt10_sumafter_mutinf_dict['summerize_after_prep'] = ''
pt10_sumafter_mutinf_unif_dict = copy.deepcopy(pt10_sumafter_mutinf_dict)
pt10_sumafter_mutinf_unif_dict['uniform_bptt'] = ''

#debug
debug_dict = copy.deepcopy(pt10_dict)
debug_dict['epochs'] = 10

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
pt100_unif_dict['topk_key'] = 'Val_nc_Accuracy'
pt100_unif_search_dict = copy.deepcopy(pt100_unif_dict)
pt100_unif_search_dict['bptt_search'] = ''
pt100_unif_kl_dict = copy.deepcopy(pt100_unif_dict)
pt100_unif_kl_dict['kl_loss'] = ''
pt100_unif_kl_dict['early_stopping'] = 2
pt100_unif_kl_nopp_dict = copy.deepcopy(pt100_unif_kl_dict)
pt100_unif_kl_nopp_dict.pop('do_preprocess')
pt100_prop_dict = copy.deepcopy(pt100_dict)
pt100_prop_dict['tuned_prompt_label_balance'] = 'proportional'
pt100_pca_dict = copy.deepcopy(pt100_dict)
pt100_pca_dict['subset_features_method'] = 'pca'
pt100_mut_dict = copy.deepcopy(pt100_dict)
pt100_mut_dict['subset_features_method'] = 'mutual_information'
pt100_rand_dict = copy.deepcopy(pt100_dict)
pt100_rand_dict['subset_features_method'] = 'random'
pt100_powerall_dict = copy.deepcopy(pt100_dict)
pt100_powerall_dict['preprocess_type'] = 'power_all'
pt100_sumafter_pca_dict = copy.deepcopy(pt100_pca_dict)
pt100_sumafter_pca_dict['summerize_after_prep'] = ''
pt100_sumafter_pca_unif_dict = copy.deepcopy(pt100_sumafter_pca_dict)
pt100_sumafter_pca_unif_dict['uniform_bptt'] = ''
pt100_sumafter_mutinf_dict = copy.deepcopy(pt100_mut_dict)
pt100_sumafter_mutinf_dict['summerize_after_prep'] = ''
pt100_sumafter_mutinf_unif_dict = copy.deepcopy(pt100_sumafter_mutinf_dict)
pt100_sumafter_mutinf_unif_dict['uniform_bptt'] = ''

#Prompt tuning with 500 prompts
pt500_dict = copy.deepcopy(pt10_dict)
pt500_dict['tuned_prompt_size'] = 500

#Prompt tuning with 1000 prompts
pt1000_dict = copy.deepcopy(pt10_dict)
pt1000_dict['tuned_prompt_size'] = 1000
pt1000_long_dict = copy.deepcopy(pt1000_dict)
pt1000_long_dict['epochs'] = 101
pt1000_long_dict['save_every_k_epochs'] = 102
pt1000_long_dict['validation_period'] = 5
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
pt1000_unif_dict['topk_key'] = 'Val_nc_Accuracy'
pt1000_unif_search_dict = copy.deepcopy(pt1000_unif_dict)
pt1000_unif_search_dict['bptt_search'] = ''
pt1000_unif_kl_dict = copy.deepcopy(pt1000_unif_dict)
pt1000_unif_kl_dict['kl_loss'] = ''
pt1000_unif_kl_dict['early_stopping'] = 2
pt1000_unif_highep_lowlr_dict = copy.deepcopy(pt1000_unif_dict)
pt1000_unif_highep_lowlr_dict['epochs'] = 101
pt1000_unif_highep_lowlr_dict['lr'] = 0.05
pt1000_unif_highep_lowlr_dict['save_every_k_epochs'] = 102
pt1000_unif_highep_lowlr_dict['validation_period'] = 5
pt1000_unif_highep_lowlr_dict['early_stopping'] = 10
pt1000_prop_dict = copy.deepcopy(pt1000_dict)
pt1000_prop_dict['tuned_prompt_label_balance'] = 'proportional'
pt1000_pca_dict = copy.deepcopy(pt1000_dict)
pt1000_pca_dict['subset_features_method'] = 'pca'
pt1000_mut_dict = copy.deepcopy(pt1000_dict)
pt1000_mut_dict['subset_features_method'] = 'mutual_information'
pt1000_rand_dict = copy.deepcopy(pt1000_dict)
pt1000_rand_dict['subset_features_method'] = 'random'
pt1000_powerall_dict = copy.deepcopy(pt1000_dict)
pt1000_powerall_dict['preprocess_type'] = 'power_all'
pt1000_sumafter_pca_dict = copy.deepcopy(pt1000_pca_dict)
pt1000_sumafter_pca_dict['summerize_after_prep'] = ''
pt1000_sumafter_pca_unif_dict = copy.deepcopy(pt1000_sumafter_pca_dict)
pt1000_sumafter_pca_unif_dict['uniform_bptt'] = ''
pt1000_sumafter_mutinf_dict = copy.deepcopy(pt1000_mut_dict)
pt1000_sumafter_mutinf_dict['summerize_after_prep'] = ''
pt1000_sumafter_mutinf_unif_dict = copy.deepcopy(pt1000_sumafter_mutinf_dict)
pt1000_sumafter_mutinf_unif_dict['uniform_bptt'] = ''

# Prompt Tuning with 2000 prompts
pt2000_dict = copy.deepcopy(pt10_dict)
pt2000_dict['tuned_prompt_size'] = 2000

"""
Ensemble Presets
"""

#Bagging
ens_bagged_avg_dict = dict(pt1000_dict, **ens_bagged_avg_dict)
ens_bagged_avg_top2_dict = copy.deepcopy(ens_bagged_avg_dict)
ens_bagged_avg_top2_dict['keep_topk_ensemble'] = 2
ens_bagged_avg_top2_powerall_dict = copy.deepcopy(ens_bagged_avg_top2_dict)
ens_bagged_avg_top2_powerall_dict['preprocess_type'] = 'power_all'

#Bagging Top2 Reseed
ens_bagged_avg_top2_reseed_dict = copy.deepcopy(ens_bagged_avg_top2_dict)
ens_bagged_avg_top2_reseed_dict['reseed_data'] = ''
ens_bagged_avg_top2_reseed_ss10000_dict = copy.deepcopy(ens_bagged_avg_top2_reseed_dict)
ens_bagged_avg_top2_reseed_ss10000_dict['subsampling'] = 10000

#Bagging Top2 Uniform Reseed
ens_bagged_avg_top2_unif_reseed_dict = copy.deepcopy(ens_bagged_avg_top2_reseed_dict)
ens_bagged_avg_top2_unif_reseed_dict['uniform_bptt'] = ''
ens_bagged_avg_top2_unif_reseed_dict['topk_key'] = 'Val_nc_Accuracy'
ens_bagged_avg_top2_unif_reseed_ss10000_dict = copy.deepcopy(ens_bagged_avg_top2_unif_reseed_dict)
ens_bagged_avg_top2_unif_reseed_ss10000_dict['subsampling'] = 10000
ens_bagged_avg_top2_unif_reseed_ss50000_dict = copy.deepcopy(ens_bagged_avg_top2_unif_reseed_dict)
ens_bagged_avg_top2_unif_reseed_ss50000_dict['subsampling'] = 50000

# Other variations
ens_bagged_avg_top4_unif_reseed_ss50000_dict = copy.deepcopy(ens_bagged_avg_top2_unif_reseed_ss50000_dict)
ens_bagged_avg_top4_unif_reseed_ss50000_dict['keep_topk_ensemble'] = 4

#Randinit
ens_randinit_avg_dict = copy.deepcopy(ens_bagged_avg_dict)
ens_randinit_avg_dict.pop('bagging')
ens_randinit_avg_dict['rand_init_ensemble'] = ''
ens_randinit_avg_top2_dict = copy.deepcopy(ens_randinit_avg_dict)
ens_randinit_avg_top2_dict['keep_topk_ensemble'] = 2

ens_randinit_avg_dict = dict(pt1000_dict, **ens_randinit_avg_dict)
ens_randinit_avg_top2_reseed_dict = copy.deepcopy(ens_randinit_avg_top2_dict)
ens_randinit_avg_top2_reseed_dict['reseed_data'] = ''
ens_randinit_avg_top2_reseed_pca_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_dict)
ens_randinit_avg_top2_reseed_pca_dict['subset_features_method'] = 'pca'
ens_randinit_avg_top1_reseed_pca_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_pca_dict)
ens_randinit_avg_top1_reseed_pca_dict['keep_topk_ensemble'] = 1
ens_randinit_avg_top2_pt100_reseed_pca_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_pca_dict)
ens_randinit_avg_top2_pt100_reseed_pca_dict['tuned_prompt_size'] = 100
ens_randinit_avg_top2_reseed_sumafter_pca_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_pca_dict)
ens_randinit_avg_top2_reseed_sumafter_pca_dict['summerize_after_prep'] = ''
ens_randinit_avg_top1_reseed_sumafter_pca_dict = copy.deepcopy(ens_randinit_avg_top1_reseed_pca_dict)
ens_randinit_avg_top1_reseed_sumafter_pca_dict['summerize_after_prep'] = ''
ens_randinit_avg_top2_pt100_reseed_sumafter_pca_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_sumafter_pca_dict)
ens_randinit_avg_top2_pt100_reseed_sumafter_pca_dict['tuned_prompt_size'] = 100
ens_randinit_avg_top2_reseed_mutinf_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_dict)
ens_randinit_avg_top2_reseed_mutinf_dict['subset_features_method'] = 'mutual_information'
ens_randinit_avg_top1_reseed_mutinf_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_mutinf_dict)
ens_randinit_avg_top1_reseed_mutinf_dict['keep_topk_ensemble'] = 1
ens_randinit_avg_top2_pt100_reseed_mutinf_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_mutinf_dict)
ens_randinit_avg_top2_pt100_reseed_mutinf_dict['tuned_prompt_size'] = 100
ens_randinit_avg_top2_reseed_sumafter_mutinf_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_mutinf_dict)
ens_randinit_avg_top2_reseed_sumafter_mutinf_dict['summerize_after_prep'] = ''
ens_randinit_avg_top1_reseed_sumafter_mutinf_dict = copy.deepcopy(ens_randinit_avg_top1_reseed_mutinf_dict)
ens_randinit_avg_top1_reseed_sumafter_mutinf_dict['summerize_after_prep'] = ''
ens_randinit_avg_top2_pt100_reseed_sumafter_mutinf_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_sumafter_mutinf_dict)
ens_randinit_avg_top2_pt100_reseed_sumafter_mutinf_dict['tuned_prompt_size'] = 100
ens_randinit_avg_top2_reseed_100cl_long_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_dict)
ens_randinit_avg_top2_reseed_100cl_long_dict['max_num_classes'] = 100
ens_randinit_avg_top2_reseed_100cl_long_dict['epochs'] = 101
ens_randinit_avg_top2_reseed_100cl_long_dict['lr'] = 0.05
ens_randinit_avg_top2_reseed_100cl_long_dict['early_stopping'] = 10
ens_randinit_avg_top2_reseed_100cl_long_dict['save_every_k_epochs'] = 102
ens_randinit_avg_top2_reseed_100cl_long_dict['validation_period'] = 4
ens_randinit_avg_top2_reseed_100cl_long_pca_dict = ens_randinit_avg_top2_reseed_100cl_long_dict
ens_randinit_avg_top2_reseed_100cl_long_sumafter_pca_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_100cl_long_dict)
ens_randinit_avg_top2_reseed_100cl_long_sumafter_pca_dict['summerize_after_prep'] = ''
ens_randinit_avg_top2_reseed_100cl_long_mutinf_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_100cl_long_dict)
ens_randinit_avg_top2_reseed_100cl_long_mutinf_dict['subset_features_method'] = 'mutual_information'
ens_randinit_avg_top2_reseed_100cl_long_sumafter_mutinf_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_100cl_long_mutinf_dict)
ens_randinit_avg_top2_reseed_100cl_long_sumafter_mutinf_dict['summerize_after_prep'] = ''
ens_randinit_avg_top2_reseed_lowlr_short_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_dict)
ens_randinit_avg_top2_reseed_lowlr_short_dict['lr'] = 0.03
ens_randinit_avg_top2_reseed_lowlr_short_dict['epochs'] = 7

#Randinit top2 uniform reseed
ens_randinit_avg_top2_unif_reseed_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_dict)
ens_randinit_avg_top2_unif_reseed_dict['uniform_bptt'] = ''
ens_randinit_avg_top2_unif_reseed_dict['topk_key'] = 'Val_nc_Accuracy'
ens_randinit_avg_top2_unif_reseed_pca_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_dict)
ens_randinit_avg_top2_unif_reseed_pca_dict['subset_features_method'] = 'pca'
ens_randinit_avg_top2_unif_pt100_reseed_pca_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_pca_dict)
ens_randinit_avg_top2_unif_pt100_reseed_pca_dict['tuned_prompt_size'] = 100
ens_randinit_avg_top2_unif_reseed_sumafter_pca_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_dict)
ens_randinit_avg_top2_unif_reseed_sumafter_pca_dict['summerize_after_prep'] = ''
ens_randinit_avg_top2_unif_pt100_reseed_sumafter_pca_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_sumafter_pca_dict)
ens_randinit_avg_top2_unif_pt100_reseed_sumafter_pca_dict['tuned_prompt_size'] = 100
ens_randinit_avg_top2_unif_reseed_mutinf_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_dict)
ens_randinit_avg_top2_unif_reseed_mutinf_dict['subset_features_method'] = 'mutual_information'
ens_randinit_avg_top2_unif_pt100_reseed_mutinf_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_mutinf_dict)
ens_randinit_avg_top2_unif_pt100_reseed_mutinf_dict['tuned_prompt_size'] = 100
ens_randinit_avg_top2_unif_reseed_sumafter_mutinf_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_mutinf_dict)
ens_randinit_avg_top2_unif_reseed_sumafter_mutinf_dict['summerize_after_prep'] = ''
ens_randinit_avg_top2_unif_pt100_reseed_sumafter_mutinf_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_sumafter_mutinf_dict)
ens_randinit_avg_top2_unif_pt100_reseed_sumafter_mutinf_dict['tuned_prompt_size'] = 100
ens_randinit_avg_top2_unif_reseed_ss10000_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_dict)
ens_randinit_avg_top2_unif_reseed_ss10000_dict['subsampling'] = 10000
ens_randinit_avg_top2_unif_reseed_100cl_long_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_dict)
ens_randinit_avg_top2_unif_reseed_100cl_long_dict['max_num_classes'] = 100
ens_randinit_avg_top2_unif_reseed_100cl_long_dict['epochs'] = 101
ens_randinit_avg_top2_unif_reseed_100cl_long_dict['lr'] = 0.05
ens_randinit_avg_top2_unif_reseed_100cl_long_dict['early_stopping'] = 10
ens_randinit_avg_top2_unif_reseed_100cl_long_dict['save_every_k_epochs'] = 102
ens_randinit_avg_top2_unif_reseed_100cl_long_dict['validation_period'] = 4
ens_randinit_avg_top2_unif_reseed_100cl_long_pca_dict = ens_randinit_avg_top2_unif_reseed_100cl_long_dict
ens_randinit_avg_top2_unif_reseed_100cl_long_sumafter_pca_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_100cl_long_dict)
ens_randinit_avg_top2_unif_reseed_100cl_long_sumafter_pca_dict['summerize_after_prep'] = ''
ens_randinit_avg_top2_unif_reseed_100cl_long_mutinf_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_100cl_long_dict)
ens_randinit_avg_top2_unif_reseed_100cl_long_mutinf_dict['subset_features_method'] = 'mutual_information'
ens_randinit_avg_top2_unif_reseed_100cl_long_sumafter_mutinf_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_100cl_long_mutinf_dict)
ens_randinit_avg_top2_unif_reseed_100cl_long_sumafter_mutinf_dict['summerize_after_prep'] = ''
ens_randinit_avg_top2_unif_reseed_lowlr_short_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_dict)
ens_randinit_avg_top2_unif_reseed_lowlr_short_dict['lr'] = 0.03
ens_randinit_avg_top2_unif_reseed_lowlr_short_dict['epochs'] = 7

# Other variations
ens_randinit_avg_top4_unif_reseed_ss10000_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_ss10000_dict)
ens_randinit_avg_top4_unif_reseed_ss10000_dict['keep_topk_ensemble'] = 4
ens_randinit_avg_top2_unif_reseed_stopearly_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_dict)
ens_randinit_avg_top2_unif_reseed_stopearly_dict['early_stopping'] = 2
ens_randinit_avg_top3_unif_reseed_highlr_short_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_dict)
ens_randinit_avg_top3_unif_reseed_highlr_short_dict['lr'] = 0.2
ens_randinit_avg_top3_unif_reseed_highlr_short_dict['epochs'] = 7
ens_randinit_avg_top3_unif_reseed_highlr_short_dict['keep_topk_ensemble'] = 3
ens_randinit_avg_top3_unif_reseed_sumafter_pca_highlr_short = copy.deepcopy(ens_randinit_avg_top3_unif_reseed_highlr_short_dict)
ens_randinit_avg_top3_unif_reseed_sumafter_pca_highlr_short['summerize_after_prep'] = ''
ens_randinit_avg_top1_reseed_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_dict)
ens_randinit_avg_top1_reseed_dict['keep_topk_ensemble'] = 1
ens3_randinit_avg_top1_reseed_dict = copy.deepcopy(ens_randinit_avg_top1_reseed_dict)
ens3_randinit_avg_top1_reseed_dict['ensemble_size'] = 3
ens3_pt100_randinit_avg_top1_reseed_dict = copy.deepcopy(ens3_randinit_avg_top1_reseed_dict)
ens3_pt100_randinit_avg_top1_reseed_dict['tuned_prompt_size'] = 100
ens3_pt100_randinit_avg_top2_reseed_dict = copy.deepcopy(ens3_pt100_randinit_avg_top1_reseed_dict)
ens3_pt100_randinit_avg_top2_reseed_dict['keep_topk_ensemble'] = 2
ens3_pt100_randinit_avg_top1_reseed_unif_dict = copy.deepcopy(ens3_pt100_randinit_avg_top1_reseed_dict)
ens3_pt100_randinit_avg_top1_reseed_unif_dict['uniform_bptt'] = ''
ens3_pt100_randinit_avg_top1_reseed_unif_dict['topk_key'] = 'Val_nc_Accuracy'
ens3_pt100_randinit_avg_top2_reseed_unif_dict = copy.deepcopy(ens3_pt100_randinit_avg_top1_reseed_unif_dict)
ens3_pt100_randinit_avg_top2_reseed_unif_dict['keep_topk_ensemble'] = 2

ens_randinit_avg_top1_unif_reseed_stopearly_dict = copy.deepcopy(ens_randinit_avg_top2_unif_reseed_stopearly_dict)
ens_randinit_avg_top1_unif_reseed_stopearly_dict['keep_topk_ensemble'] = 1
ens_randinit_avg_top1_unif_reseed_stopearly_pca_dict = copy.deepcopy(ens_randinit_avg_top1_unif_reseed_stopearly_dict)
ens_randinit_avg_top1_unif_reseed_stopearly_pca_dict['subset_features_method'] = 'pca'
ens_randinit_avg_top1_unif_reseed_stopearly_mutinf_dict = copy.deepcopy(ens_randinit_avg_top1_unif_reseed_stopearly_dict)
ens_randinit_avg_top1_unif_reseed_stopearly_mutinf_dict['subset_features_method'] = 'mutual_information'
ens_randinit_avg_top1_unif_reseed_highlr_short_dict = copy.deepcopy(ens_randinit_avg_top3_unif_reseed_highlr_short_dict)
ens_randinit_avg_top1_unif_reseed_highlr_short_dict['keep_topk_ensemble'] = 1
ens_randinit_avg_top1_unif_reseed_highlr_short_powerall_dict = copy.deepcopy(ens_randinit_avg_top1_unif_reseed_highlr_short_dict)
ens_randinit_avg_top1_unif_reseed_highlr_short_powerall_dict['preprocess_type'] = 'power_all'
ens_randinit_avg_top1_unif_reseed_highlr_short_powerall_mutinf_dict = copy.deepcopy(ens_randinit_avg_top1_unif_reseed_highlr_short_powerall_dict)
ens_randinit_avg_top1_unif_reseed_highlr_short_powerall_mutinf_dict['subset_features_method'] = 'mutual_information'
ens_randinit_avg_top1_reseed_large_highlr_longep_dict = copy.deepcopy(ens_randinit_avg_top2_reseed_dict)
ens_randinit_avg_top1_reseed_large_highlr_longep_dict['reseed_data'] = ''
ens_randinit_avg_top1_reseed_large_highlr_longep_dict['lr'] = 0.3
ens_randinit_avg_top1_reseed_large_highlr_longep_dict['epochs'] = 101
ens_randinit_avg_top1_reseed_large_highlr_longep_dict['save_every_k_epochs'] = 102
ens_randinit_avg_top1_reseed_large_highlr_longep_dict['tuned_prompt_size'] = 2000
ens_randinit_avg_top1_reseed_large_highlr_longep_dict['keep_topk_ensemble'] = 1
ens_randinit_avg_top1_reseed_large_lowlr_longep_dict_noes = copy.deepcopy(ens_randinit_avg_top1_reseed_large_highlr_longep_dict)
ens_randinit_avg_top1_reseed_large_lowlr_longep_dict_noes['lr'] = 0.05
ens_randinit_avg_top1_reseed_large_lowlr_longep_dict_noes['early_stopping'] = 10
ens_randinit_avg_top2_reseed_large_lowlr_longep_dict_noes = copy.deepcopy(ens_randinit_avg_top1_reseed_large_lowlr_longep_dict_noes)
ens_randinit_avg_top2_reseed_large_lowlr_longep_dict_noes['keep_topk_ensemble'] = 2
ens_randinit_avg_top1_reseed_small_highlr_longep_dict = copy.deepcopy(ens_randinit_avg_top1_reseed_large_highlr_longep_dict)
ens_randinit_avg_top1_reseed_small_highlr_longep_dict['tuned_prompt_size'] = 10

all_tasks = {
    'debug' : debug_dict,
    'ft' : base_dict,
    'pt2' : pt2_dict,
    'pt2-uniform' : pt2_unif_dict,
    'pt2-uniform-kl' : pt2_unif_kl_dict,
    'pt5' : pt5_dict,
    'pt5-uniform' : pt5_unif_dict,
    'pt5-uniform-kl' : pt5_unif_kl_dict,
    'pt5-uniform-kl-nopp' : pt5_unif_kl_nopp_dict,
    'pt5-uniform-kl-nopp-lowlr' : pt5_unif_kl_nopp_lowlr_dict,
    'pt10': pt10_dict,
    'pt10-prop' : pt10_prop_dict,
    'pt10-pca' : pt10_pca_dict,
    'pt10-mutual_information' : pt10_mut_dict,
    'pt10-rand' : pt10_rand_dict,
    'pt10-powerall' : pt10_powerall_dict,
    'pt10-sumafter-pca' : pt10_sumafter_pca_dict,
    'pt10-sumafter-mutual_information' : pt10_sumafter_mutinf_dict,
    'pt10-short' : pt10_short_dict,
    'pt10-short-lowlr' : pt10_short_lowlr_dict,
    'pt10-short-lowlr-prop' : pt10_short_lowlr_prop_dict,
    'pt10-short-lowlr-prop-pca' : pt10_short_lowlr_prop_pca_dict,
    'pt10-short-lowlr-prop-mutual_information' : pt10_short_lowlr_prop_mutinf_dict,
    'pt10-short-lowlr-prop-sumafter-pca' : pt10_short_lowlr_prop_sumafter_pca_dict,
    'pt10-short-lowlr-prop-sumafter-mutual_information' : pt10_short_lowlr_prop_sumafter_mutinf_dict,
    'pt10-shorter' : pt10_shorter_dict,
    'pt10-uniform' : pt10_unif_dict,
    'pt10-uniform-kl' : pt10_unif_kl_dict,
    'pt10-uniform-kl-nopp' : pt10_unif_kl_nopp_dict,
    'pt10-uniform-kl-nopp-pca' : pt10_unif_kl_nopp_pca_dict,
    'pt10-uniform-kl-nopp-mutual_information' : pt10_unif_kl_nopp_mutinf_dict,
    'pt10-uniform-kl-nopp-sumafter-pca' : pt10_unif_kl_nopp_sumafter_pca_dict,
    'pt10-uniform-kl-nopp-sumafter-mutual_information' : pt10_unif_kl_nopp_sumafter_mutinf_dict,
    'pt10-uniform-kl-prop' : pt10_unif_kl_prop_dict,
    'pt10-uniform-kl-nopp-prop' : pt10_unif_kl_nopp_prop_dict,
    'pt10-uniform-kl-nopp-prop-pca' : pt10_unif_kl_nopp_prop_pca_dict,
    'pt10-uniform-kl-nopp-prop-mutual_information' : pt10_unif_kl_nopp_prop_mutinf_dict,
    'pt10-uniform-kl-nopp-prop-sumafter-pca' : pt10_unif_kl_nopp_prop_sumafter_pca_dict,
    'pt10-uniform-kl-nopp-prop-sumafter-mutual_information' : pt10_unif_kl_nopp_prop_sumafter_mutinf_dict,
    'pt10-uniform-sumafter-pca' : pt10_sumafter_pca_unif_dict,
    'pt10-uniform-sumafter-mutual_information' : pt10_sumafter_mutinf_unif_dict,
    'pt10-uniform-short' : pt10_short_unif_dict,
    'pt10-uniform-shorter' : pt10_shorter_unif_dict,
    'pt10-5ens-randinit-avg-top1-reseed-highlr-longep' : ens_randinit_avg_top1_reseed_small_highlr_longep_dict,
    'pt100': pt100_dict,
    'pt100-prop' : pt100_prop_dict,
    'pt100-pca' : pt100_pca_dict,
    'pt100-mut' : pt100_mut_dict,
    'pt100-rand' : pt100_rand_dict,
    'pt100-uniform' : pt100_unif_dict,
    'pt100-uniform-search' : pt100_unif_search_dict,
    'pt100-uniform-kl' : pt100_unif_kl_dict,
    'pt100-uniform-kl-nopp' : pt100_unif_kl_nopp_dict,
    'pt100-uniform-sumafter-pca' : pt100_sumafter_pca_unif_dict,
    'pt100-uniform-sumafter-mutual_information' : pt100_sumafter_mutinf_unif_dict,
    'pt100-uniform-short' : pt100_short_unif_dict,
    'pt100-uniform-shorter' : pt100_shorter_unif_dict,
    'pt100-powerall' : pt100_powerall_dict,
    'pt100-sumafter-pca' : pt100_sumafter_pca_dict,
    'pt100-sumafter-mutual_information' : pt100_sumafter_mutinf_dict,
    'pt100-short' : pt100_short_dict,
    'pt100-shorter' : pt100_shorter_dict,
    'pt100-3ens-randinit-avg-top1-reseed' : ens3_randinit_avg_top1_reseed_dict,
    'pt100-3ens-randinit-avg-top2-reseed' : ens3_pt100_randinit_avg_top2_reseed_dict,
    'pt100-3ens-randinit-avg-top1-reseed-unif' : ens3_pt100_randinit_avg_top1_reseed_unif_dict,
    'pt100-3ens-randinit-avg-top2-reseed-unif' : ens3_pt100_randinit_avg_top2_reseed_unif_dict,
    'pt500': pt500_dict,
    'pt1000': pt1000_dict,
    'pt1000-prop' : pt1000_prop_dict,
    'pt1000-pca' : pt1000_pca_dict,
    'pt1000-mut' : pt1000_mut_dict,
    'pt1000-rand' : pt1000_rand_dict,
    'pt1000-uniform' : pt1000_unif_dict,
    'pt1000-uniform-search' : pt1000_unif_search_dict,
    'pt1000-uniform-kl' : pt1000_unif_kl_dict,
    'pt1000-uniform-sumafter-pca' : pt1000_sumafter_pca_unif_dict,
    'pt1000-uniform-sumafter-mutual_information' : pt1000_sumafter_mutinf_unif_dict,
    'pt1000-uniform-highep-lowlr' : pt1000_unif_highep_lowlr_dict,
    'pt1000-uniform-short' : pt1000_short_unif_dict,
    'pt1000-uniform-shorter' : pt1000_shorter_unif_dict,
    'pt1000-powerall' : pt1000_powerall_dict,
    'pt1000-sumafter-pca' : pt1000_sumafter_pca_dict,
    'pt1000-sumafter-mutual_information' : pt1000_sumafter_mutinf_dict,
    'pt1000-short' : pt1000_short_dict,
    'pt1000-shorter' : pt1000_shorter_dict,
    'pt1000-long' : pt1000_long_dict,
    'pt1000-3ens-randinit-avg-top1-reseed' : ens3_randinit_avg_top1_reseed_dict,
    'pt1000-10ens-bagged-avg' : ens_bagged_avg_dict,
    'pt1000-10ens-randinit-avg' : ens_randinit_avg_dict,
    'pt1000-10ens-bagged-avg-top2' : ens_bagged_avg_top2_dict,
    'pt1000-10ens-bagged-avg-top2-powerall' : ens_bagged_avg_top2_powerall_dict,
    'pt1000-10ens-randinit-avg-top2' : ens_randinit_avg_top2_dict,
    'pt1000-10ens-bagged-avg-top2-reseed' : ens_bagged_avg_top2_reseed_dict,
    'pt1000-10ens-bagged-avg-top2-reseed-ss10000' : ens_bagged_avg_top2_reseed_ss10000_dict,
    'pt1000-10ens-bagged-avg-top2-unif-reseed-ss50000' : ens_bagged_avg_top2_unif_reseed_ss50000_dict,
    'pt1000-10ens-bagged-avg-top4-unif-reseed-ss50000' : ens_bagged_avg_top4_unif_reseed_ss50000_dict,
    'pt1000-10ens-bagged-avg-top2-unif-reseed' : ens_bagged_avg_top2_unif_reseed_dict,
    'pt1000-10ens-bagged-avg-top2-unif-reseed-ss10000' : ens_bagged_avg_top2_unif_reseed_ss10000_dict,
    'pt1000-10ens-randinit-avg-top2-reseed' : ens_randinit_avg_top2_reseed_dict,
    'pt100-10ens-randinit-avg-top2-reseed-sumafter-pca' : ens_randinit_avg_top2_pt100_reseed_sumafter_pca_dict,
    'pt1000-10ens-randinit-avg-top2-reseed-sumafter-pca' : ens_randinit_avg_top2_reseed_sumafter_pca_dict,
    'pt1000-10ens-randinit-avg-top1-reseed-sumafter-pca' : ens_randinit_avg_top1_reseed_sumafter_pca_dict,
    'pt100-10ens-randinit-avg-top2-reseed-pca' : ens_randinit_avg_top2_pt100_reseed_pca_dict,
    'pt1000-10ens-randinit-avg-top2-reseed-pca' : ens_randinit_avg_top2_reseed_pca_dict,
    'pt1000-10ens-randinit-avg-top1-reseed-pca' : ens_randinit_avg_top1_reseed_pca_dict,
    'pt100-10ens-randinit-avg-top2-reseed-sumafter-mutual_information' : ens_randinit_avg_top2_pt100_reseed_sumafter_mutinf_dict,
    'pt1000-10ens-randinit-avg-top2-reseed-sumafter-mutual_information' : ens_randinit_avg_top2_reseed_sumafter_mutinf_dict,
    'pt1000-10ens-randinit-avg-top1-reseed-sumafter-mutual_information' : ens_randinit_avg_top1_reseed_sumafter_mutinf_dict,
    'pt100-10ens-randinit-avg-top2-reseed-mutual_information' : ens_randinit_avg_top2_pt100_reseed_mutinf_dict,
    'pt1000-10ens-randinit-avg-top2-reseed-mutual_information' : ens_randinit_avg_top2_reseed_mutinf_dict,
    'pt1000-10ens-randinit-avg-top1-reseed-mutual_information' : ens_randinit_avg_top1_reseed_mutinf_dict,
    'pt1000-10ens-randinit-avg-top2-reseed-lowlr-short' : ens_randinit_avg_top2_reseed_lowlr_short_dict,
    'pt1000-10ens-randinit-avg-top2-reseed-100cl-long' : ens_randinit_avg_top2_reseed_100cl_long_dict,
    'pt1000-10ens-randinit-avg-top2-reseed-100cl-long-pca' : ens_randinit_avg_top2_reseed_100cl_long_pca_dict,
    'pt1000-10ens-randinit-avg-top2-reseed-100cl-long-sumafter-pca' : ens_randinit_avg_top2_reseed_100cl_long_sumafter_pca_dict,
    'pt1000-10ens-randinit-avg-top2-reseed-100cl-long-mutual_information' : ens_randinit_avg_top2_reseed_100cl_long_mutinf_dict,
    'pt1000-10ens-randinit-avg-top2-reseed-100cl-long-sumafter-mutual_information' : ens_randinit_avg_top2_reseed_100cl_long_sumafter_mutinf_dict,
    'pt1000-10ens-randinit-avg-top2-unif-reseed' : ens_randinit_avg_top2_unif_reseed_dict,
    'pt1000-10ens-randinit-avg-top2-unif-reseed-ss10000' : ens_randinit_avg_top2_unif_reseed_ss10000_dict,
    'pt100-10ens-randinit-avg-top2-unif-reseed-pca' : ens_randinit_avg_top2_unif_pt100_reseed_pca_dict,
    'pt1000-10ens-randinit-avg-top2-unif-reseed-pca' : ens_randinit_avg_top2_unif_reseed_pca_dict,
    # 'pt1000-10ens-randinit-avg-top1-unif-reseed-pca' : ens_randinit_avg_top1_unif_reseed_pca_dict,
    'pt100-10ens-randinit-avg-top2-unif-reseed-sumafter-pca' : ens_randinit_avg_top2_unif_pt100_reseed_sumafter_pca_dict,
    'pt1000-10ens-randinit-avg-top2-unif-reseed-sumafter-pca' : ens_randinit_avg_top2_unif_reseed_sumafter_pca_dict,
    'pt100-10ens-randinit-avg-top2-unif-reseed-mutual_information' : ens_randinit_avg_top2_unif_pt100_reseed_mutinf_dict,
    'pt1000-10ens-randinit-avg-top2-unif-reseed-mutual_information' : ens_randinit_avg_top2_unif_reseed_mutinf_dict,
    'pt100-10ens-randinit-avg-top2-unif-reseed-sumafter-mutual_information' : ens_randinit_avg_top2_unif_pt100_reseed_sumafter_mutinf_dict,
    'pt1000-10ens-randinit-avg-top2-unif-reseed-sumafter-mutual_information' : ens_randinit_avg_top2_unif_reseed_sumafter_mutinf_dict,
    'pt1000-10ens-randinit-avg-top2-unif-reseed-lowlr-short' : ens_randinit_avg_top2_unif_reseed_lowlr_short_dict,
    'pt1000-10ens-randinit-avg-top3-unif-reseed-highlr-short' : ens_randinit_avg_top3_unif_reseed_highlr_short_dict,
    'pt1000-10ens-randinit-avg-top3-unif-reseed-sumafter-pca-highlr-short' : ens_randinit_avg_top3_unif_reseed_sumafter_pca_highlr_short,
    'pt1000-10ens-randinit-avg-top1-unif-reseed-stopearly' : ens_randinit_avg_top1_unif_reseed_stopearly_dict,
    'pt1000-10ens-randinit-avg-top2-unif-reseed-stopearly' : ens_randinit_avg_top2_unif_reseed_stopearly_dict,
    'pt1000-10ens-randinit-avg-top1-unif-reseed-pca-stopearly' : ens_randinit_avg_top1_unif_reseed_stopearly_pca_dict,
    'pt1000-10ens-randinit-avg-top1-unif-reseed-mutual_information-stopearly' : ens_randinit_avg_top1_unif_reseed_stopearly_mutinf_dict,
    'pt1000-10ens-randinit-avg-top1-unif-reseed-highlr-short' : ens_randinit_avg_top1_unif_reseed_highlr_short_dict,
    'pt1000-10ens-randinit-avg-top1-unif-reseed-highlr-short-powerall' : ens_randinit_avg_top1_unif_reseed_highlr_short_powerall_dict,
    'pt1000-10ens-randinit-avg-top1-unif-reseed-mutual_information-highlr-short-powerall' : ens_randinit_avg_top1_unif_reseed_highlr_short_powerall_mutinf_dict,
    'pt1000-10ens-randinit-avg-top2-unif-reseed-100cl-long' : ens_randinit_avg_top2_unif_reseed_100cl_long_dict,
    'pt1000-10ens-randinit-avg-top2-unif-reseed-100cl-long-pca' : ens_randinit_avg_top2_unif_reseed_100cl_long_pca_dict,
    'pt1000-10ens-randinit-avg-top2-unif-reseed-100cl-long-sumafter-pca' : ens_randinit_avg_top2_unif_reseed_100cl_long_sumafter_pca_dict,
    'pt1000-10ens-randinit-avg-top2-unif-reseed-100cl-long-mutual_information' : ens_randinit_avg_top2_unif_reseed_100cl_long_mutinf_dict,
    'pt1000-10ens-randinit-avg-top2-unif-reseed-100cl-long-sumafter-mutual_information' : ens_randinit_avg_top2_unif_reseed_100cl_long_sumafter_mutinf_dict,
    'pt1000-10ens-randinit-avg-top4-unif-reseed-ss10000' : ens_randinit_avg_top4_unif_reseed_ss10000_dict,
    'pt2000' : pt2000_dict,
    'pt2000-5ens-randinit-avg-top1-reseed-highlr-longep' : ens_randinit_avg_top1_reseed_large_highlr_longep_dict,
    'pt2000-5ens-randinit-avg-top1-reseed-lowlr-longep-noes' : ens_randinit_avg_top1_reseed_large_lowlr_longep_dict_noes,
    'pt2000-10ens-randinit-avg-top2-reseed-lowlr-longep-noes' : ens_randinit_avg_top2_reseed_large_lowlr_longep_dict_noes,
}

def get_all_tasks():
    return all_tasks