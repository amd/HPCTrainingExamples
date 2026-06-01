#!/bin/bash
#
# starts up ray within a container, spanning large multinode training run
# ray communicates across the virtual cluster over ethernet,
# RCCL communicates over the slingshot high speed fabric
#

# arg checking
if [ $# != 5 ]; then
    echo "Usage: $0 <head_ipaddr> --head|--worker </path/to/models> </path/to/datasets> <multistate>"
    exit 1
fi

# host addr, node type and directory holding our LLMs
HEAD_NODE_ADDRESS="$1"  # IP of the machine designated as the head (do NOT list worker machine IPs here)
NODE_TYPE="$2"          # should be --head or --worker
MODELPATH="$3"          # directory holding the model weights
DATASETSPATH="$4"	# directory holding the datasets for training
MULTISTATE="$5"		# whether numnodes=1 or >1

# arg checking on node type
if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

# arg checking on model path
if [ ! -d "${MODELPATH}" ]; then
    echo "Error: Model path directory ${MODELPATH} does not exist"
    exit 1
fi

# arg checking on datasets path
if [ ! -d "${DATASETSPATH}" ]; then
    echo "Error: Datasets path directory ${DATASETSPATH} does not exist"
    exit 1
fi

# whether it's a single or multinode run
if [ "${MULTISTATE}" != "single" ] && [ "${MULTISTATE}" != "multi" ]; then
    echo "Error: multistate must be either single or multi"
    exit 1
fi


# kill off any partial container that might be left behind from a failed prior process
podman rm -af
buildah rm -a
podman rm -af
sleep 30

# define a function to cleanup on EXIT signal
cleanup() {
    podman stop node || true
    podman rm --force node || true
}
trap cleanup EXIT

# get the IP address of the HPE slingshot device
MY_HSI_ADDRESS=`ifconfig hsi0 | grep inet | awk '{print $2}'`
# IP address of the ethernet device (use for ray comms)
MY_ETH_ADDRESS=`ifconfig enp129s0 | grep inet | awk '{print $2}'`


# command setup for head or worker node
SYS_CONFIG='{"state_api_timeout_s":60,"num_heartbeats_timeout":600,"health_check_timeout_ms":60000}'
RAY_START_CMD="ray start --block --disable-usage-stats --log-style=pretty --object-store-memory 10000000000"
if [ "${NODE_TYPE}" == "--head" ]; then
    if [ "${MULTISTATE}" == "single" ]; then
      # if it's a single node run, allow all gpus to be used on head (beware ray head memory usage though)
      RAY_START_CMD+=" --head --node-ip-address=${MY_ETH_ADDRESS} --port=6379"
    else
      # allow only a single training instance to run on the head node to answer API requests
      #RAY_START_CMD+=" --head --node-ip-address=${MY_HSI_ADDRESS} --port=6379  --num-cpus=0 --num-gpus=1"
      # as of ray v2.44.0, the mem usage on the head node is no longer excessive: allow 4 workers to run there
      RAY_START_CMD+=" --head --node-ip-address=${MY_ETH_ADDRESS} --port=6379"
    fi
else
    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379 --node-ip-address=${MY_ETH_ADDRESS}"
fi

# local node host name
MYHOSTNAME=`hostname`

# copy the container image on the rabbit local storage node
IMAGE=train_image.tar
RABBIT_DIR=/l/ssd
RABBIT_IMAGE=${RABBIT_DIR}/${IMAGE}
#echo "transferring container image ${IMAGE} to the rabbit local storage device on ${MYHOSTNAME} ..."
#flux run -n 96 dcp ${IMAGE} ${RABBIT_DIR}
#echo "...completed"

# clear out any old containers to avoid rootless uid mapping problems
podman system prune --force
sleep 30

# start up ray within the container
echo "starting up container on ${MYHOSTNAME} with ray command line ${RAY_START_CMD}"
podman run \
   -it \
   --log-level=error \
   --name=node \
   --network=host \
   --group-add=video \
   --ipc=host \
   --cap-add=SYS_PTRACE \
   --security-opt seccomp=unconfined \
   --device /dev/kfd \
   --device /dev/dri \
   --device /dev/cxi0 \
   --device /dev/cxi1 \
   --device /dev/cxi2 \
   --device /dev/cxi3 \
   --device /dev/cxi_sbl \
   --privileged  \
   --env RAY_system_config="$SYS_CONFIG" \
   --env RAY_ADDRESS_HOST=${MY_ETH_ADDRESS} \
   --env RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 \
   --env RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
   --env RAY_EXPERIMENTAL_NO_SET_CUDA_VISIBLE_DEVICES=1 \
   --env NCCL_DMABUF_ENABLE=0 \
   --env TORCH_NCCL_BLOCKING_WAIT=0 \
   --env TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
   --env TORCH_NCCL_ENABLE_MONITORING=0 \
   --env TRITON_CACHE_DIR=/tmp/.triton_cache \
   --env TORCH_EXTENSIONS_DIR=/tmp/.torch_extensions \
   --env HF_HOME=/tmp/.cache/huggingface \
   --env HIP_VISIBLE_DEVICES=0,1,2,3 \
   --env ROCR_VISIBLE_DEVICES=0,1,2,3 \
   --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
   --env OMP_NUM_THREADS=1 \
   --env MKL_NUM_THREADS=1 \
   --env HIPBLASLT_DISABLE=1 \
   --env TORCH_BLAS_PREFER_HIPBLASLT=0 \
   --env NCCL_NET_GDR_LEVEL=0 \
   --env PYTORCH_TUNABLEOP_ENABLED=0 \
   --env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
   --env NCCL_P2P_DISABLE=1 \
   --env HSA_XNACK=0 \
   --env ROCBLAS_USE_HIPBLASLT=0 \
   --env NCCL_SOCKET_IFNAME=enp129s0 \
   -v $MODELPATH:/app/models \
   -v $DATASETSPATH:/app/datasets \
   docker-archive:${IMAGE} \
   ${RAY_START_CMD}


# XXX JB
# NCCL_DMABUF_ENABLE=0 might need to be flipped for multinode training
#
#   --env HIP_VISIBLE_DEVICES=0,1,2,3 \
#   --env ROCR_VISIBLE_DEVICES=0,1,2,3 \
#   --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
#
#   --env NCCL_SOCKET_IFNAME=hsi0 \
#   --env RAY_EXPERIMENTAL_NO_SET_CUDA_VISIBLE_DEVICES=1 \
