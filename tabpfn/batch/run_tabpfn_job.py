import subprocess
import asyncio
import os
import time
import datetime
import argparse
import json
import shutil
from pathlib import Path

from tqdm.auto import tqdm

from all_tasks import get_all_tasks

MAX_CLASSES = 10
MAX_FEATURES = 100
MAX_SAMPLES = 3000

async def run_command(cmd):
    # Start the subprocess
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    # Wait for the command to finish
    stdout, stderr = await process.communicate()

    return process.returncode, stdout, stderr

def main_f(args):

    def run_tunetables(dataset_path, task, split, log_dir, args, base_cmd, gcp_txt):
        #TODO: Full impl
        if args.gcp_run:
            raise NotImplementedError("GCP run not yet supported for tunetables task, please run each task individually and aggregate results.")
        args.real_data_qty = MAX_SAMPLES
        metadata = json.load(open(os.path.join(dataset_path, 'metadata.json')))
        n_classes = metadata['num_classes']
        n_features = metadata['num_features']
        n_samples = metadata['num_instances']
        all_res = {}
        all_res_d = {}
        if n_features > MAX_FEATURES:
            print("Sweeping feature subselection methods.")
            #NOTE: Other options: zs-isomap-32, zs-ica-32, zs-random-32, zs-sparse_random_projection-32
            tt_tasks = ['zs-pca_white-16', 'zs-mutual_information-16']
            for task in tt_tasks:
                res = run_single_job(dataset_path, task, split, log_dir, args, base_cmd, gcp_txt)
                all_res[task] = res["Val_Accuracy"]
                all_res_d[task] = res
            best_task = max(all_res, key=all_res.get)
            feat_sel_method = "-" + best_task.split('-')[1]
        else:
            feat_sel_method = ''
        if n_classes > 25:
            raise NotImplementedError("Please add a task to all_tasks for the correct number of classes (modify task pt1000-10ens-randinit-avg-top2-unif-reseed-25cl-long).")
        if n_classes > MAX_CLASSES and n_classes < 25:
            tt_tasks = [f'pt1000-10ens-randinit-avg-top2-reseed-25cl-long', f'pt1000-10ens-randinit-avg-top2-unif-reseed-25cl-long']
        if n_samples <= MAX_SAMPLES:
            if feat_sel_method == '':
                feat_sel_method = 'random'
            tt_tasks = ['zs-{feat_sel_method}-2', 'zs-{feat_sel_method}-16', 'zs-{feat_sel_method}-32']
        for task in tt_tasks:
            if all_res_d.get(task, None) is not None:
                continue
            args.bptt = args.bptt_backup
            if 'unif' in task:
                args.bptt_backup = args.bptt
                args.bptt = 128
            res = run_single_job(dataset_path, task, split, log_dir, args, base_cmd, gcp_txt)
            all_res_d[task] = res
            all_res[task] = max(res.get("Val_Accuracy", 0.0), res.get("Val_nc_Accuracy", 0.0), res.get("Ens_Val_Accuracy", 0.0), res.get("Ens_Val_Accuracy_NC", 0.0))

    def run_single_job(dataset_path, task, split, log_dir, args, base_cmd, gcp_txt):
            # Get task name
            task = task.strip()
            task_str = task
            if args.run_optuna:
                task_str += '_optuna'
            if args.bptt > -1:
                task_str += '_bptt_' + str(args.bptt)
            if args.shuffle_every_epoch:
                task_str += '_shuffleep_'
            task_str += '_rdq_' + str(args.real_data_qty)
            task_str += '_split_' + str(split)
            if task.startswith('zs'):
                ensemble_size = int(task.split('-')[-1])
                subset_ft_method = task.split('-')[-2]
                command = ['python', base_cmd, 
                        '--data_path', dataset_path,
                        '--subset_features_method', subset_ft_method,
                        '--split', str(split),
                        '--real_data_qty', str(args.real_data_qty),
                        '--zs-eval-ensemble', str(ensemble_size)]
                if args.wandb_log:
                    command = command + [
                        '--wandb_log',
                        '--wandb_group', "\"" + dataset.strip() + "_" + task_str + "_" + subset_ft_method + "\"", 
                    ]
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
                if not args.wandb_log:
                    next_task.pop('wandb_log')
                if args.wandb_project != '':
                    next_task['wandb_project'] = args.wandb_project
                if args.resume != '':
                    next_task['resume'] = args.resume
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
                if args.gcp_run:
                    command = ['python', base_cmd, '--data_path \"' + dataset_path + "\"", '--split', str(split), '--real_data_qty', str(args.real_data_qty), '--wandb_group', "\"" + dataset.strip() + "_" + task_str + "\""] + addl_args
                else:
                    command = ['python', base_cmd, '--data_path', dataset_path, '--split', str(split), '--real_data_qty', str(args.real_data_qty), '--wandb_group', "\"" + dataset.strip() + "_" + task_str +  "\""] + addl_args
                if args.run_optuna:
                    if args.wandb_project == '':
                        command = command + ["--wandb_project", args.wandb_project]
                    else:
                        command = command + ["--wandb_project", "tabpfn-pt-optuna"]
            if args.bptt > -1:
                command.append("--bptt")
                command.append(str(args.bptt))     
            if args.shuffle_every_epoch:
                command.append("--shuffle_every_epoch")           
            print("Running command:", ' '.join(command))
            if args.gcp_run:
                gcp_txt += "\'" + ' '.join(command) + '\'\n'
            else:
                returncode, stdout, stderr = asyncio.run(run_command(' '.join(command)))
                stdout = stdout.decode()
                print("Stderr:", stderr.decode())
                if args.print_stdout:
                    print("Stdout:", stdout)
                # Initialize an empty dictionary to hold the parsed output
                output_dict = {}
                # Define the marker indicating the start of the JSON output
                json_start_marker = "^RESULTS\n"
                # Check if the marker is in the stdout
                if json_start_marker in stdout:
                    # Extract the JSON string part. Assume the JSON starts immediately after the marker
                    json_str = stdout.split(json_start_marker, 1)[1]
                    
                    # Attempt to parse the JSON string into a Python dictionary
                    try:
                        output_dict = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        output_dict = {}
                # Parse and relocate logs
                new_outputs = Path('logs').glob('_multiclass*')
                updated_outputs = []
                for output in new_outputs:
                    new_name = output.name.replace('_multiclass', task_str)
                    new_path = os.path.join(output.parent, new_name)
                    os.rename(output, new_path)
                    updated_outputs.append(new_path)
                for output in updated_outputs:
                    shutil.move(output, log_dir)
                return output_dict

    with open(args.datasets) as f:
        datasets = f.readlines()

    with open(args.tasks) as f:
        tasks = f.readlines()

    all_tasks = get_all_tasks()

    if args.run_optuna:
        base_cmd = 'run_optuna.py'
    else:
        base_cmd = 'train_loop.py'

    gcp_txt = "run_commands=(\n"

    for dataset in tqdm(datasets):
        print("Starting dataset: ", dataset.strip())
        dataset_path = "\"" + os.path.join(args.base_path, dataset.strip()) + '\"'
        #sanitize name
        # dataset_path = dataset_path.replace(r'(', r'\(').replace(r')', r'\)')
        # print("Dataset path:", dataset_path)
        log_dir = './logs/' + dataset.strip()
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        for split in args.splits:
            for task in tasks:
                if task == 'tunetables':
                    res = run_tunetables(dataset_path, task, split, log_dir, args, base_cmd, gcp_txt)
                else:
                    res = run_single_job(dataset_path, task, split, log_dir, args, base_cmd, gcp_txt)
                print("Results:", res)
    if args.gcp_run:
        task_str = "tunetables_gcp_" + dataset.strip() + "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        gcp_txt += ")"
        with open("run_commands.sh", "w") as f:
            f.write(gcp_txt)
        start_time = time.time()
        print("Starting GCP run.")
        returncode, stdout, stderr = asyncio.run(run_command('bash batch/run_gcp_expt.sh'))
        print("GCP run finished in", time.time() - start_time, "seconds.")
        if args.print_stdout:
            print("Stdout:")
            print(stdout.decode())
            print("Stderr:")
            print(stderr.decode())
        target_dir = os.path.join(log_dir, task_str)
        os.makedirs(target_dir, exist_ok=True)
        with open(os.path.join(target_dir, "stdout.txt"), "w") as f:
            f.write(stdout.decode())
        with open(os.path.join(target_dir, "stderr.txt"), "w") as f:
            f.write(stderr.decode())
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run TabPFN')
    parser.add_argument('--base_path', type=str, default='/home/benfeuer/TabPFN-pt/tabpfn/data', help='Path to TabPFN-pt dataset directory')
    parser.add_argument('--datasets', type=str, default='/home/benfeuer/TabPFN-pt/tabpfn/metadata/subset.txt', help='Path to datasets text file')
    parser.add_argument('--tasks', type=str, default='/home/benfeuer/TabPFN-pt/tabpfn/metadata/subset_tasks.txt', help='Tasks to run')
    parser.add_argument('--resume', type=str, default='/home/benfeuer/TabPFN-pt/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt', help='TabPFN checkpoint to resume from')
    parser.add_argument('--bptt', type=int, default=-1, help='bptt batch size')
    parser.add_argument('--splits', nargs='+', type=int, default=[0], help='Splits to run')
    parser.add_argument('--shuffle_every_epoch', action='store_true', help='Whether to shuffle the order of the data every epoch (can help when bptt is large).')
    parser.add_argument('--run_optuna', action='store_true', help='Whether to run optuna hyperparameter search.')
    parser.add_argument('--real_data_qty', type=int, default=0, help='Number of real data points to use for fitting.')
    parser.add_argument('--gcp_run', action='store_true', help='Whether to launch the job on a GCP instance.')
    parser.add_argument('--wandb_log', action='store_true', help='Whether to log to wandb.')
    parser.add_argument('--wandb_project', type=str, default='', help='Project name for wandb logging')
    parser.add_argument('--print_stdout', action='store_true', help='Whether to print stdout from each run.')
    args = parser.parse_args()
    main_f(args)
