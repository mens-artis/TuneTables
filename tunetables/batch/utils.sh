run_experiment_gpu() {

  source ./config/gcp_vars.sh

  run_command="$1"
  instance_name="$2"

  # set a return trap to delete the instance when this function returns
  trap "echo deleting instance ${instance_name}...; printf \"Y\" | gcloud compute instances delete ${instance_name} --zone=${zone} --project=${project}" RETURN

  # maximum number of attempts at creating gcloud instance and ssh
  MAX_TRIES=3

  echo "launching instance ${instance_name}..."

  COUNT=1
  while [ $COUNT -le $MAX_TRIES ]; do

    # attempt to create instance

    gcloud compute instances create $instance_name --zone=$zone \
    --project=$project --image-family=$image_family \
    --machine-type=${machine_type} \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --service-account $service_account \
    --metadata=^,@^ssh-keys=nc3468:ssh-rsa\ \
AAAAB3NzaC1yc2EAAAADAQABAAABAQDfhoLPr6ZoSSL9epL7N0YQuJ9nD\+JB5CmK/f3NTX0vmOAHT51Dlmb/9G7AocqykQ0lwyaZ4fdrx4LL6hz20HjZUh1FGaDKkLgBdXdThY/YuPijKroO5sLivbAHRY2XHyva5fIdBIzGUJ1K4taILKwUWiG\+tMm7w1UCA6evZGgCh5olGcENa8A7yB2dqxLikpjTX2HdzaAPxDgIFJ6K2aBFenyIRWeofHm\+Ng/xAySPn3nTxUn/S1QYb2/hOWKDocVK4g1BJJqqTcmbzS3lYs6Wa30kdA4ChJpvJt9fX/ArNR9KKrGUSafHcMOao\+jfayau4zuI5dfpgiV1BOxDNxfL\ nc3468@nyu.edu$'\n'penfever:ecdsa-sha2-nistp256\ AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBFRhIujJC2W1TnWm9386COjviNKoZFVjFwZdbXvTWgV/Wc7Om6gryvbC5XodzD/0j29cQ/ycYd7aZ/yhsr3m/Eo=\ google-ssh\ \{\"userName\":\"penfever@gmail.com\",\"expireOn\":\"2024-01-04T16:21:58\+0000\"\}$'\n'penfever:ssh-rsa\ AAAAB3NzaC1yc2EAAAADAQABAAABAQCjS/Lv93SFJTOsgsY4/WWcKe4TmrYrNLHyZG0vatWel6G5NyYS3jGEt8Y2uVt86FH50gGfo\+KxVhlTehw4p\+xVYeqxY1xOKq7vqpHGpkNiJzTTYwSW9Yt\+cSQKJdAfx/BGibVUiPzquYxLFP0LbdQvy47drtARgVykLQadDABd2w5iUSoLpWlukUTuxjefXWxnWQqf8/Zo3I1d0WUDw4327LBhItN2NtysdLXv2QWoYsrYjo6oNbiuxvb3DuuTD4oAWdxCleQSpxPyzWMI4orst7LKsIfjnzdJEYGuK6OT5k8awquA/ysOj4GmA3bv1jc3CbU1cujX/uLF0yFfxXyx\ google-ssh\ \{\"userName\":\"penfever@gmail.com\",\"expireOn\":\"2024-01-04T16:22:13\+0000\"\} \
    --no-restart-on-failure \
    --maintenance-policy TERMINATE \
    --boot-disk-size=200GB \
    --scopes=https://www.googleapis.com/auth/devstorage.read_write \
    --accelerator=count=1,type=nvidia-l4

    # keep this for later
    INSTANCE_RETURN_CODE=$?

    if [ $INSTANCE_RETURN_CODE -ne 0 ]; then
      # failed to create instance
      let COUNT=COUNT+1
      echo "failed to create instance during attempt ${COUNT}... (exit code: ${INSTANCE_RETURN_CODE})"
      if [[ $COUNT -ge $(( $MAX_TRIES + 1 )) ]]; then
        echo "too many create-instance attempts. giving up."
        exit 1
      fi
      echo "trying again in 30 seconds..."
      sleep 30
    else
      # success!
      break
    fi
  done
  echo "successfully created instance: ${instance_name}"

  sleep 10
  # ssh and run the experiment. steps:
  # 1. set environment variables used by script tabzilla_experiment.sh
  # 2. chmod the experiment script
  # 3. run the experiment script
  instance_repo_dir=/home/benfeuer/TuneTables
  # instance_script=${instance_repo_dir}/scripts/run_experiment_on_instance.sh

  COUNT=1
  MAX_TRIES_SSH=2
  while [ $COUNT -le $MAX_TRIES_SSH ]; do

    # attempt to run tunetables experiment
    if [[ $run_command == *"tunetables"* ]]; then
      # get substrings

      delimiter="_dataset_"
      replacement_delimiter=$'\x1F' # Use a character unlikely to be in the string
      delimiter2="_args_"
      replacement_delimiter2=$'\x0F'

      # Replace the delimiter with the replacement delimiter and then split
      modified_string="${run_command//$delimiter/$replacement_delimiter}"
      IFS="$replacement_delimiter" read -ra parts <<< "$modified_string"
      task_str="${parts[0]}"
      rem_str="${parts[1]}"
      modified_string2="${rem_str//$delimiter2/$replacement_delimiter2}"
      IFS="$replacement_delimiter2" read -ra parts <<< "$modified_string2"
      dataset_str="${parts[0]}"
      args_str="${parts[1]}"
      run_cmd="python3 batch/run_tt_job.py ${args_str} --datasets './metadata/dataset.txt' --tasks './metadata/task.txt'"

      echo "running tunetables experiment with task: ${task_str}, dataset: ${dataset_str}, args: ${args_str}"

      gcloud compute ssh --ssh-flag="-A" ${instance_name} --zone=${zone} --project=${project} \
        --command="\
        sudo /opt/deeplearning/install-driver.sh; \
        cd ${instance_repo_dir}; \
        source /home/bf996/.bashrc; \
        cd ${instance_repo_dir}/tunetables; \
        sudo echo "${task_str}" >> metadata/task.txt; \
        printf '%s\n' \"${dataset_str}\" >> metadata/dataset.txt; \
        ${run_cmd}; \
        "
    else
      # attempt to run standard experiment
      gcloud compute ssh --ssh-flag="-A" ${instance_name} --zone=${zone} --project=${project} \
        --command="\
        sudo /opt/deeplearning/install-driver.sh; \
        cd ${instance_repo_dir}; \
        source /home/bf996/.bashrc; \
        cd ${instance_repo_dir}/tunetables; \
        ${run_command}; \
        "
    fi

    #${run_command}
    SSH_RETURN_CODE=$?
    echo "ssh return code: ${SSH_RETURN_CODE}"
    # was this instance preempted?
    echo gcloud compute operations list \
      --filter="operationType=compute.instances.preempted AND targetLink:instances/${instance_name}"


    if [ $SSH_RETURN_CODE -ne 0 ]; then
      # failed to run experiment
      let COUNT=COUNT+1
      echo "failed to run experiment during attempt ${COUNT}... (exit code: ${SSH_RETURN_CODE})"
      if [[ $COUNT -ge $(( $MAX_TRIES_SSH + 1 )) ]]; then
        echo "too many SSH attempts. giving up and deleting instance."
        printf "Y" | gcloud compute instances delete ${instance_name} --zone=${zone} --project=${project}
        exit 1
      fi
      echo "trying again in 30 seconds..."
      sleep 30
    else
      # success!
      break
    fi
  done
  echo "successfully ran experiment"

  # we don't need to delete the instance here, because we set a return trap
}

delete_instances() {

  source ./config/gcp_vars.sh
  # deletes all instances in global variable INSTANCE_LIST
  echo "attempting to delete all instances..."
  for i in "${INSTANCE_LIST[@]}";
    do
        echo "deleting instance: $i"
        printf "Y" | gcloud compute instances delete $i --zone=${zone} --project=${project}
    done
}

wait_until_processes_finish() {
  # only takes one arg: the maximum number of processes that can be running
  # print a '.' every 60 iterations
  counter=0
  while [ `gcloud compute instances list --filter='status=RUNNING' | wc -l` -gt $1 ]; do
    sleep 1
    counter=$((counter+1))
    if (($counter % 60 == 0))
    then
      echo -n "."     # no trailing newline
    fi
  done
  echo "no more than $1 jobs are running. moving on."
}
