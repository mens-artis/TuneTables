#!/bin/bash

MAX_PROCESSES=7

#defines run_experiment_gpu
source ./batch/utils.sh

# set -e

source run_commands.sh

trap "delete_instances" EXIT
INSTANCE_LIST=()

num_experiments=0

for cmd in "${run_commands[@]}"; do
  instance_name="tunetables-$(date +%Y%m%d%H%M%S)"
  INSTANCE_LIST+=("${instance_name}")
  num_experiments=$((num_experiments+1))
  run_experiment_gpu "$cmd" "$instance_name" >> "runlog-$(date +%Y%m%d%H%M%S).txt" 2>&1 &
  sleep 60
  # if we have started MAX_PROCESSES experiments, wait for them to finish
  wait_until_processes_finish $MAX_PROCESSES
done

echo "still waiting for processes to finish..."
wait
echo "done."