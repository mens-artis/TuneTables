<br/>
<p align="center"><img src="img/tunetables_logo.png" width=700 /></p>

----
![Crates.io](https://img.shields.io/crates/l/Ap?color=orange)

We introduce TuneTables, a tabular classification algorithm that overcomes the limitations of prior-data fitted networks to achieve strong performance on large datasets.

While TabPFN achieves very strong performance on small tabular datasets, its current limitations include fewer than 1000 datapoints, fewer than 100 features, and fewer than 10 class labels. In this work, we overcome these limitations and substantially improve the performance of PFNs by developing context optimization techniques; specifically, we propose TuneTables, a novel prompt-tuning strategy. TuneTables scales TabPFN to be competitive with state-of-the-art tabular classification methods on larger datasets, while having additional benefits as well: (1) substantially lower inference time than TabPFN, (2) can be used as an interpretability tool, and (3) can mitigate biases by optimizing a fairness objective.

<p align="center"><img src="img/tunetables_overview.png" width=700 /></p>

This codebase extends the excellent public repository [TabPFN]([xxx](https://github.com/automl/tabpfn)), by Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter.

## Table of Contents
1. [Hardware Requirements](#hardware-requirements)
2. [Installation](#installation)
3. [Datasets](#datasets)
4. [Running Experiments](#running-experiments)

## Hardware Requirements

* Any modern Linux workstation should be compatible with the packages called by TuneTables.
* We recommend running TuneTables on a GPU. In our experiments, we use a single NVIDIA L4 with 24GB VRAM.

## Installation

1. Clone TabPFN-pt (TuneTables) repository to your local instance.
2. Initialize a clean environment of your preference, E.G.:

```bash
conda create --name "TuneTables" python=3.10
```

3. From the TabPFN-pt directory, run --

```bash
pip install . 
```

All commands should be run from within the `tabpfn` directory.

```
cd tabpfn
```

## Datasets

TuneTables was tested on over 40 datasets sourced from [OpenML](https://www.openml.org/). We recommend downloading some of the datasets listed in order to get familiar with our software.

### Dataset Preparation

In order to prepare your dataset(s) for TuneTables, we recommend following the dataset preparation steps outlined in the [TabZilla](https://github.com/naszilla/tabzilla?tab=readme-ov-file#datasets) repository, which we used in our experiments. We also include two preprocessed datasets, `openml__elevators__3711` and `openml__har__14970` in `TabPFN-pt\sample_datasets`, for testing purposes.

### List of OpenML Datasets

An extended list of datasets we use for experiments are included below.

```
openml__Bioresponse__9910
openml__blood-transfusion-service-center__10101
openml__breast-cancer__145799
openml__BNG(labor)__2137
openml__BNG(vote)__212
openml__christine__168908
openml__Click_prediction_small__7294
openml__climate-model-simulation-crashes__146819
openml__colic__27
openml__connect-4__146195
openml__dilbert__168909
openml__dresses-sales__125920
openml__eeg-eye-state__14951
openml__elevators__3711
openml__har__14970
openml__higgs__146606
openml__poker-hand__9890
openml__riccardo__168338
openml__robert__168332
openml__volkert__168331
openml__balance-scale__11
openml__cmc__23
openml__cylinder-bands__14954
openml__heart-c__48
openml__kc1__3917
openml__Agrawal1__146093
openml__airlines__189354
openml__albert__189356
```

## Running Experiments

### Datasets

Dataset names are stored in plaintext files, newline separated. 

```
openml__airlines__189354
openml__albert__189356
```

Examples of task and dataset lists can be found in the `tabpfn/metadata` directory.

### Tasks

TuneTables jobs are organized into `tasks`, which are then executed in batches over a list of `datasets`. Examples of task and dataset lists can be found in the `tabpfn/metadata` directory. Here are some example tasks which can be included in a TuneTables batch.

```bash
ft #fine tune TabPFN end-to-end
pt10 #learn a prompt with an embedding of dimensions (10, ndim)
pt1000-10ens-randinit-avg-top2-reseed #learn an ensemble of 10 prompts, each randomly initialized, with reseeded data
```

The complete list of tuning tasks can be found in `tabpfn/batch/all_tasks.py`.

### Zero-Shot Tasks

It is also possible to run zero-shot TabPFN tasks via TuneTables, across a range of context sizes, ensemble sizes, feature subsampling methods. Here are some example zero-shot tasks which can be included in a TuneTables batch.

```bash
zs-pca_white-32 #ensemble size 32, feature selection via principal component analysis with whitening
zs-sparse_random_projection-16 #ensemble size 16, feature selection via sparse_random_projection
zs-mutual_information-8 #ensemble size 8, feature selection via mutual information
```

### Jobs

As mentioned earlier, TuneTables `jobs` are organized into `tasks`, which are then executed in batches over a list of `datasets`. Jobs are invoked via `batch\run_tabpfn_job.py`.

Since prompt tuning requires a pretrained and frozen transformer, we need to `--resume` from a previous checkpoint. A TabPFN checkpoint is included in the repository: `tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt`.

Also required is the `--base_path` where the datasets you wish to evaluate are stored.

Here is an example of how you might run a job in TuneTables:

```bash
python3 batch/run_tabpfn_job.py --resume PATH/TO/CHECKPOINT --base_path PATH/TO/DATASETS --datasets metadata/test_datasets.txt --tasks metadata/test_tasks.txt
```

### Special Flags

The `run_tabpfn_job` script accepts several special flags. We describe their function here.

`--splits`: A space-separated list of splits to evaluate for each dataset, `0` by default. Our results are usually reported on an average: `--splits 0 1 2`.

`--real_data_qty`: Controls how much of the dataset is passed as context to the model during validation (`C / CT` setting only)

`--bptt`: Controls the length of the sequence passed to the model during training. When the task type is uniform, `bptt` is processed as a constant. When the task type is non-uniform, `bptt` is processed as the argmax of a random variable.

