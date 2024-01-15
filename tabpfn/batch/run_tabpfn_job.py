import subprocess
import os
import argparse
from tqdm.auto import tqdm
from all_tasks import get_all_tasks
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description='Run TabPFN')
parser.add_argument('--base_path', type=str, default='/cmlscratch/vcherepa/TabPFN-pt/tabpfn/data', help='Path to TabPFN-pt dataset directory')
parser.add_argument('--datasets', type=str, default='/cmlscratch/vcherepa/TabPFN-pt/tabpfn/metadata/subset_datasets.txt', help='Path to datasets text file')
parser.add_argument('--tasks', type=str, default='/cmlscratch/vcherepa/TabPFN-pt/tabpfn/metadata/subset_tasks.txt', help='Tasks to run')
parser.add_argument('--bptt', type=int, default=-1, help='bptt batch size')
parser.add_argument('--splits', nargs='+', type=int, default=[0], help='Splits to run')
parser.add_argument('--shuffle_every_epoch', action='store_true', help='Whether to shuffle the order of the data every epoch (can help when bptt is large).')
parser.add_argument('--run_optuna', action='store_true', help='Whether to run optuna hyperparameter search.')

args = parser.parse_args()

with open(args.datasets) as f:
    datasets = f.readlines()

with open(args.tasks) as f:
    tasks = f.readlines()

all_tasks = get_all_tasks()

if args.run_optuna:
    base_cmd = 'run_optuna_n.py'
else:
    base_cmd = 'train_loop.py'

for dataset in tqdm(datasets):
    print("Starting dataset: ", dataset.strip())
    dataset_path = os.path.join(args.base_path, dataset.strip())
    log_dir = './logs/' + dataset.strip()
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    for split in args.splits:
        for task in tasks:
            # Get task name
            task = task.strip()
            task_str = task
            if args.run_optuna:
                task_str += '_optuna'
            if args.bptt > -1:
                task_str += '_bptt_' + str(args.bptt)
            if args.shuffle_every_epoch:
                task_str += '_shuffleep_'
            task_str += '_split_' + str(split)
            if task.startswith('zs'):
                ensemble_size = int(task.split('-')[-1])
                command = ['python', base_cmd, 
                           '--data_path', dataset_path,
                           '--feature_subset_method', 'pca',
                           '--split', str(split), 
                           '--wandb_group', dataset.strip() + "_" + task_str, 
                           '--zs-eval-ensemble', str(ensemble_size)]
            else:
                # Get task args
                npp = False
                npad = False
                if '-npad' in task:
                    npad = True
                    task = task.replace('-npad', '')
                if '-nopreproc' in task:
                    npp = True
                    task = task.replace('-nopreproc', '')
                next_task = all_tasks[task]
                if npp:
                    try:
                        next_task.pop('do_preprocess')
                    except:
                        pass
                    task_str += '_nopreproc'
                if npad:
                    try:
                        next_task.pop('pad_features')
                    except:
                        pass
                    task_str += '_npad'
                addl_args = []
                for k, v in next_task.items():
                    addl_args.append("--" + k)
                    val = str(v)
                    if val != '':
                        addl_args.append(val)
                command = ['python', base_cmd, '--data_path', dataset_path, '--split', str(split), '--wandb_group', dataset.strip() + "_" + task_str] + addl_args
            if args.bptt > -1:
                command.append("--bptt")
                command.append(str(args.bptt))     
            if args.shuffle_every_epoch:
                command.append("--shuffle_every_epoch")           
            print("Running command:", ' '.join(command))
            subprocess.call(command)
            new_outputs = Path('logs').glob('_multiclass*')
            updated_outputs = []
            for output in new_outputs:
                new_name = output.name.replace('_multiclass', task_str)
                new_path = os.path.join(output.parent, new_name)
                os.rename(output, new_path)
                updated_outputs.append(new_path)
            for output in updated_outputs:
                shutil.move(output, log_dir)
