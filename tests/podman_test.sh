#!/bin/bash
module load rocm podman
podman run --device=/dev/dri --device=/dev/kfd  --network=host --ipc=host --group-add keep-groups -v $HOME:/workdir --workdir /workdir docker://rocm/dev-ubuntu-22.04:6.4.1 rocminfo

# to launch a shell session inside a container
#podman run -it --device=/dev/dri --device=/dev/kfd  --network=host --ipc=host --group-add keep-groups -v $HOME:/workdir --workdir /workdir docker://rocm/dev-ubuntu-22.04:6.1.2-complete
