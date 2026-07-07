#!/bin/bash
#
# automates the launch of the container + ray.train kernel
# zenith uses a split-fabric approach where the control plane (Ray/GSC/ethernet)
# is separate from the data plane (training/RCCL/hsi)
#

# arg checking
if [ $# -lt 2 ]; then
    echo "Usage: $0 <directory holding the models> <directory holding datasets>"
    exit 1
fi
MODELPATH="$1"          # directory holding our LLMs
DATASETSPATH="$2"	# directory holding our datasets for training

# arg checking on model path
if [ ! -d "${MODELPATH}" ]; then
    echo "Error: Model path directory ${MODELPATH} does not exist"
    exit 1
fi

# get rid of any container logs that might be laying around from a previous run
rm -f container.*.log

# number of nodes assigned to this job
RESOURCEINFO=`flux resource info`
NNODES=`echo ${RESOURCEINFO} | awk '{print $1}'`
NGPUS=`echo ${RESOURCEINFO} | awk '{print $5}'`
NCPUS=`echo ${RESOURCEINFO} | awk '{print $3}'`
GPUSPERNODE=$((NGPUS / NNODES))
CPUSPERNODE=$((NCPUS / NNODES))

# get the list of nodes running within this flux job context
MYJOBID=`env | grep -w FLUX_JOB_ID | awk -F '=' '{print $2}'`
MYJOBHOSTS=`flux job hostpids ${MYJOBID} | sed 's/:[^:]*,/ /g' | sed 's/:[^:]*/ /g'`
NUMWORKERS=`echo ${MYJOBHOSTS} | wc -w`
NUMWORKERS=$((NUMWORKERS - 1))
# the first one in the list will be designated the head
HEADNODE=`echo ${MYJOBHOSTS} | awk '{print $1}'`
MYNODE=`/usr/bin/hostname`

# if it's a single node job, the single ray instance is head
if [ ${NNODES} == 1 ]; then
  HEADNODE=${MYNODE}
  NODE_TYPE="head"
  # HSI IP address
  ifconfig hsi0 | grep inet | awk '{print $2}' > container.headnode_hsi_address
  # vllm needs to listen on the ethernet device for API serving
  ifconfig enp129s0 | grep inet | awk '{print $2}' > container.headnode_eth_address
else
  # the first node of the flux job alloc is the "head", all others are "workers" for communication purposes
  if [ ${NNODES} -gt 1 ] && [ ${MYNODE} == ${HEADNODE} ]; then
    NODE_TYPE="head"
    # all nodes need to know the HSI IP address of the head
    ifconfig hsi0 | grep inet | awk '{print $2}' > container.headnode_hsi_address
    # vllm needs to listen on the ethernet device for API serving
    ifconfig enp129s0 | grep inet | awk '{print $2}' > container.headnode_eth_address
  else
    NODE_TYPE="worker"
  fi
fi

# echo to stdout if we are the head node
function headecho() {
  if [ ${MYNODE} == ${HEADNODE} ]; then
    stdbuf -oL echo "$@"
  fi
}

# echo to stdout if we are a worker node
function workerecho() {
  if [ ${MYNODE} != ${HEADNODE} ]; then
    stdbuf -oL echo "$@"
  fi
}

# current rev
ZENITH_VERSION=`cat VERSION`
headecho "Launching Zenith version ${ZENITH_VERSION}"
# output ASCII art logo
headecho ''
headecho '      ___           ___           ___                                     ___     '
headecho '     /\__\         /\__\         /\  \                                   /\  \    '
headecho '    /::|  |       /:/ _/_        \:\  \       ___           ___          \:\  \   '
headecho '   /:/:|  |      /:/ /\__\        \:\  \     /\__\         /\__\          \:\  \  '
headecho '  /:/|:|  |__   /:/ /:/ _/_   _____\:\  \   /:/__/        /:/  /      ___ /::\  \ '
headecho ' /:/ |:| /\__\ /:/_/:/ /\__\ /::::::::\__\ /::\  \       /:/__/      /\  /:/\:\__\'
headecho ' \/__|:|/:/  / \:\/:/ /:/  / \:\~~\~~\/__/ \/\:\  \__   /::\  \      \:\/:/  \/__/'
headecho '     |:/:/  /   \::/_/:/  /   \:\  \        ~~\:\/\__\ /:/\:\  \      \::/__/     '
headecho '     |::/  /     \:\/:/  /     \:\  \          \::/  / \/__\:\  \      \:\  \     '
headecho '     |:/  /       \::/  /       \:\__\         /:/  /       \:\__\      \:\__\    '
headecho '     |/__/         \/__/         \/__/         \/__/         \/__/       \/__/    '
headecho ''
sleep 5		# wait for a bit to flush

# output the job resource info
headecho "${NNODES} total nodes, headnode is ${HEADNODE}, mynode is ${MYNODE} which is the head node"
workerecho "${NNODES} total nodes, headnode is ${HEADNODE}, mynode is ${MYNODE} which is 1 of ${NUMWORKERS} workers"
sleep 10   # wait just a tad for all workers to get into the run state

# ensure a race condition doesn't exist whereby the head node hasn't yet written out it's interface IP
if [ ${NODE_TYPE} == "worker" ]; then
  # wait for the head node to write the address files
  MAX_RETRIES=30
  RETRIES=0
  while [ ! -f container.headnode_eth_address ] && [ $RETRIES -lt $MAX_RETRIES ]; do
    sleep 2
    RETRIES=$((RETRIES + 1))
  done

  if [ ! -f container.headnode_eth_address ]; then
    echo "Error: Timed out waiting for container.headnode_eth_address"
    exit 1
  fi
fi

# launch the head/worker containers across the nodes for multinode training run
echo "starting up container of type ${NODE_TYPE} on ${MYNODE} in the background"
headecho "NOTE: it can take 5-10 minutes to copy over the image blobs"
HEADNODE_HSI_ADDRESS=`cat container.headnode_hsi_address`
HEADNODE_ETH_ADDRESS=`cat container.headnode_eth_address`

# launch the container on each node (head + workers)
if [ ${NNODES} == 1 ]; then
  state_token="single"
else
  state_token="multi"
fi
# ray containers will communicate over the ethernet device, but RCCL training will occur over slingshot
/bin/bash run_container.sh ${HEADNODE_ETH_ADDRESS} --${NODE_TYPE} ${MODELPATH} ${DATASETSPATH} ${state_token} > container.${MYNODE}.log 2>&1 &

# on the ray cluster, the head node has a virtual state with access to all of the GPUs on the allocation
# first, check to make sure that all containers+ray have come up on their respective nodes
if [ ${MYNODE} == ${HEADNODE} ]; then

  # give containers a bit of a chance to get running before we go into a loop to check on them
  echo "*** Starting up all containers to form the ray virtual cluster spanning the job allocation nodes ***"
  sleep 20

  # wait for ray to start on all workers
  ALL_STARTED=0
  COUNTER=1 ; MAXCOUNTER=30
  while [ ${ALL_STARTED} -eq 0 ]; do

    # exit if we hit the max attempts
    if [ ${COUNTER} == ${MAXCOUNTER} ]; then
      echo "*** container startup failed, exiting after ${MAXCOUNTER} attempts ***"
      exit 1
    fi

    # loop over all hosts
    STARTED_SUM=0
    for i in `seq 1 ${NNODES}`
    do
      HOST_ELEMENT=`echo $MYJOBHOSTS | cut -d " " -f $i`
      HOST_FILENAME="container.${HOST_ELEMENT}.log"
      # definitely not running if log file hasn't even been created yet
      if [ -e ${HOST_FILENAME} ]; then
        GREP_STRING=`grep "Ray runtime started" ${HOST_FILENAME} > /dev/null ; echo $?`
        if [ ${GREP_STRING} == 0 ]; then
          HOST_IS_FINISHED=1
        else
          HOST_IS_FINISHED=0
        fi
      else
        HOST_IS_FINISHED=0
      fi
      # accumulate until we account for every container, up to NNODES
      STARTED_SUM=$((STARTED_SUM + ${HOST_IS_FINISHED}))
    done

    if [ ${STARTED_SUM} -eq ${NNODES} ]; then
      ALL_STARTED=1
      echo "*** All workers have successfully started ray ***"
      sleep 5
    else
      ALL_STARTED=0
      NUMSECS=60
      echo "Runtime check counter ${COUNTER} of ${MAXCOUNTER}: ${STARTED_SUM} out of ${NNODES} ray instances are running"
      echo "Runtime check counter ${COUNTER} of ${MAXCOUNTER}: HEAD is waiting ${NUMSECS} secs before checking worker startup status again"
      sleep $NUMSECS
    fi

    # increment the attempt counter
    COUNTER=$((COUNTER + 1))

  done

  # health check stage
  headecho "*** Running series of health checks before launching training ***"

  # run a test of raw cxi device communication first
  headecho "*** Checking that HPE slingshot NICs are up and communicating ***"
  if [ ${MYNODE} == ${HEADNODE} ]; then
    # output basic cxi info of the devices first
    podman exec node /opt/libfabric/bin/fi_info -p cxi
    # start the pingpong cxi server on the head
    #podman exec node /opt/libfabric/bin/fi_pingpong -p cxi -d hsi0 -I 1000 -S 1000000
  else
    # start the pingpong cxi client on each worker
    #podman exec node /opt/libfabric/bin/fi_pingpong -p cxi -d hsi0 -I 1000 -S 1000000 ${HEADNODE_HSI_ADDRESS}
    echo
  fi

  # flash attention 2 ROCm check
  headecho "*** Checking that container flash attention is using ROCm FA2 ***"
  HEALTH_FA2_KERNEL=ray_health_fa2.py
  podman cp ${HEALTH_FA2_KERNEL} node:/app/${HEALTH_FA2_KERNEL}
  podman exec node python ${HEALTH_FA2_KERNEL}

  # run a somewhat intense RCCL comms test to ensure ray communication working properly over fabric
  HEALTH_RCCL_KERNEL=ray_health_rccl.py
  podman cp ${HEALTH_RCCL_KERNEL} node:/app/${HEALTH_RCCL_KERNEL}
  headecho "*** Running 1 GB per GPU all-reduce over the compute domain to stress test the RCCL fabric, launching ${HEALTH_RCCL_KERNEL} on ${NNODES} nodes with ${CPUSPERNODE} CPUs per node and ${GPUSPERNODE} GPUs per node ***"
  podman exec --env NCCL_DEBUG="INFO" --env NCCL_DEBUG_SUBSYS="INIT,ENV,NET" node python ${HEALTH_RCCL_KERNEL} ${NNODES} ${CPUSPERNODE} ${GPUSPERNODE}

  # exercise a small kernel across all ray nodes to test overall health
  HEALTH_OVERALL_KERNEL=ray_health_overall.py
  podman cp ${HEALTH_OVERALL_KERNEL} node:/app/${HEALTH_OVERALL_KERNEL}
  headecho "*** Launching tiny test kernel over the compute domain to test the overall health of the Ray cluster, launching ${HEALTH_OVERALL_KERNEL} on ${NNODES} nodes with ${CPUSPERNODE} CPUs per node and ${GPUSPERNODE} GPUs per node ***"
  podman exec node python ${HEALTH_OVERALL_KERNEL} ${NNODES} ${CPUSPERNODE} ${GPUSPERNODE}

  headecho "*** All RCCL comm pattern health checks passed, moving to dataset preprocessing now ***"

  # data preprocessing stage
  PREPROCESS_CONFIG=ministral-3-14B-cpt-science.preprocess.yaml

  # training stage
  TRAIN_CONFIG=ministral-3-14B-cpt-science.train.yaml
  #TRAIN_CONFIG=ministral-3-14B-fft-hedp.yaml
  #TRAIN_CONFIG=gpt-oss-20b-fft-hedp.yaml
  #TRAIN_CONFIG=mistral-small-4-119B-fft-hedp.yaml

  # cp our training config to the container working filesystem
  TRAIN_CONFIG_FILENAME=$(basename "${TRAIN_CONFIG}")
  PREPROCESS_CONFIG_FILENAME=$(basename "${PREPROCESS_CONFIG}")
  for f in *.yaml *.json; do podman cp "$f" node:/app/; done

  # determione the job size specific values
  TRAIN_RAY_NWORKERS=$((NNODES * GPUSPERNODE))
  TRAIN_NGPUS_PER_WORKER=1
  TRAIN_NCPUS_PER_WORKER=$((CPUSPERNODE / GPUSPERNODE))
  PREPROCESS_NCPUS_DATASETS=$((NNODES * CPUSPERNODE / 12))
  headecho "Setting ${TRAIN_RAY_NWORKERS} Ray workers in axolotl config, with:"
  headecho "   ${TRAIN_NGPUS_PER_WORKER} GPUs and ${TRAIN_NCPUS_PER_WORKER} CPUs assigned per worker"
  headecho "   ${PREPROCESS_NCPUS_DATASETS} CPUs assigned to dataset tokenization across the entire Ray worker allocation"

  # append job specific values to preprocessing config
  PREPROCESS_DATASET_PREP_FILENAME="/app/datasets/zenith_last_run_dataprep.`date +%Y%m%d`"
  podman exec node /bin/bash -c "echo 'dataset_num_proc: ${PREPROCESS_NCPUS_DATASETS}' >> /app/${PREPROCESS_CONFIG_FILENAME}"
  podman exec node /bin/bash -c "echo 'dataloader_num_workers: 2' >> /app/${PREPROCESS_CONFIG_FILENAME}"
  podman exec node /bin/bash -c "echo 'dataloader_prefetch_factor: 2' >> /app/${PREPROCESS_CONFIG_FILENAME}"
  podman exec node /bin/bash -c "echo 'dataset_prepared_path: ${PREPROCESS_DATASET_PREP_FILENAME}' >> /app/${PREPROCESS_CONFIG_FILENAME}"

  # append the dataset info and ray worker resources for training
  podman exec node /bin/bash -c "echo 'dataset_prepared_path: ${PREPROCESS_DATASET_PREP_FILENAME}' >> /app/${TRAIN_CONFIG_FILENAME}"
  podman exec node /bin/bash -c "echo 'dataset_num_proc: ${PREPROCESS_NCPUS_DATASETS}' >> /app/${TRAIN_CONFIG_FILENAME}"
  # XXX JB
  podman exec node /bin/bash -c "echo 'dataloader_num_workers: 0' >> /app/${TRAIN_CONFIG_FILENAME}"
  #podman exec node /bin/bash -c "echo 'dataloader_prefetch_factor: 2' >> /app/${TRAIN_CONFIG_FILENAME}"
  podman exec node /bin/bash -c "echo 'dataset_prepared_path: ${PREPROCESS_DATASET_PREP_FILENAME}' >> /app/${TRAIN_CONFIG_FILENAME}"
  podman exec node /bin/bash -c "echo 'datasets:' >> /app/${TRAIN_CONFIG_FILENAME}"
  podman exec node /bin/bash -c "echo '  - path: ${PREPROCESS_DATASET_PREP_FILENAME}' >> /app/${TRAIN_CONFIG_FILENAME}"
  podman exec node /bin/bash -c "echo '    type: completion' >> /app/${TRAIN_CONFIG_FILENAME}"
  podman exec node /bin/bash -c "echo '    ds_type: arrow' >> /app/${TRAIN_CONFIG_FILENAME}"
  podman exec node /bin/bash -c "echo '' >> /app/${TRAIN_CONFIG_FILENAME}"
  podman exec node /bin/bash -c "echo 'use_ray: true' >> /app/${TRAIN_CONFIG_FILENAME}"
  podman exec node /bin/bash -c "echo 'ray_num_workers: ${TRAIN_RAY_NWORKERS}' >> /app/${TRAIN_CONFIG_FILENAME}"
  podman exec node /bin/bash -c "echo 'resources_per_worker:' >> /app/${TRAIN_CONFIG_FILENAME}"
  podman exec node /bin/bash -c "echo '  GPU: ${TRAIN_NGPUS_PER_WORKER}' >> /app/${TRAIN_CONFIG_FILENAME}"
  podman exec node /bin/bash -c "echo '  CPU: ${TRAIN_NCPUS_PER_WORKER}' >> /app/${TRAIN_CONFIG_FILENAME}"

  # preprocess the dataset on the CPUs
  headecho "*** Preprocessing the dataset, following commands in ${TRAIN_CONFIG_FILENAME} ***"
  # launch the training with extreme verbosity
  headecho "*** Launching large-scale training kernel with deep diagnostics ***"
  podman exec node python -m axolotl.cli.preprocess ministral-3-14B-cpt-science.preprocess.yaml --world_size 1 --local_rank 0

  # launch the training
  headecho "*** Launching large-scale training kernel from ${TRAIN_CONFIG_FILENAME} on ${NNODES} nodes with ${CPUSPERNODE} CPUs per node and ${GPUSPERNODE} GPUs per node ***"
  # debug line
  #podman exec node python -m axolotl.cli.train ${TRAIN_CONFIG_FILENAME} --overwrite_output_dir=true --report_to=none --debug
  podman exec node python -m axolotl.cli.train ${TRAIN_CONFIG_FILENAME} --overwrite_output_dir=true --report_to=none


fi

