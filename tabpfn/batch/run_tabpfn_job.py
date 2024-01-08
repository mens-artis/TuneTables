import subprocess
import os
import argparse
from tqdm.auto import tqdm
from all_tasks import get_all_tasks
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description='Run TabPFN')
parser.add_argument('--base_path', type=str, default='/home/benfeuer/TabPFN-pt/tabpfn/data', help='Path to TabPFN-pt dataset directory')
parser.add_argument('--datasets', type=str, default='/home/benfeuer/TabPFN-pt/tabpfn/metadata/subset.txt', help='Path to datasets text file')
parser.add_argument('--tasks', type=str, default='/home/benfeuer/TabPFN-pt/tabpfn/metadata/subset_tasks.txt', help='Tasks to run')
parser.add_argument('--bptt', type=int, default=0, help='bptt batch size')
parser.add_argument('--splits', nargs='+', type=int, default=[0], help='Splits to run')

args = parser.parse_args()

with open(args.datasets) as f:
    datasets = f.readlines()

with open(args.tasks) as f:
    tasks = f.readlines()

all_tasks = get_all_tasks()

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
            if args.bptt > 0:
                task_str += '_bptt_' + str(args.bptt)
            task_str += '_split_' + str(split)
            if task.startswith('zs'):
                ensemble_size = int(task.split('-')[-1])
                command = ['python', 'train_loop.py', '--data_path', dataset_path, '--split', str(split), '--wandb_group', dataset.strip() + "_" + task_str, '--zs-eval-ensemble', str(ensemble_size)]
                print("Running command:", ' '.join(command))
            else:
                # Get task args
                next_task = all_tasks[task]
                addl_args = []
                for k, v in next_task.items():
                    addl_args.append("--" + k)
                    val = str(v)
                    if val != '':
                        addl_args.append(val)
                if args.bptt > 0:
                    addl_args.append("--bptt")
                    addl_args.append(str(args.bptt))
                command = ['python', 'train_loop.py', '--data_path', dataset_path, '--split', str(split), '--wandb_group', dataset.strip() + "_" + task_str] + addl_args
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
