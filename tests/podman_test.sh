#!/bin/bash

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   # Rootless podman setup for this Cray node:
   #  - XDG_RUNTIME_DIR must be a valid local dir (NOT /localnvme, which is absent on compute nodes)
   #  - container storage must be on a local filesystem (NFS $HOME cannot do overlay xattrs)
   export XDG_RUNTIME_DIR=/run/user/$(id -u)
   [ -d "$XDG_RUNTIME_DIR" ] || loginctl enable-linger "$(whoami)" 2>/dev/null
   
   # Prefer node-local NVMe if present, else local tmpfs
   if [ -d /localnvme ]; then LOCAL_BASE=/localnvme/$USER; else LOCAL_BASE=/tmp/$USER; fi
   PODMAN_STORE=${LOCAL_BASE}/podman/store
   PODMAN_RUNROOT=${LOCAL_BASE}/podman/run
   mkdir -p "$PODMAN_STORE" "$PODMAN_RUNROOT"
   
   # overlay.ignore_chown_errors works around compute nodes that lack /etc/subuid entries
   PODMAN_STORAGE_ARGS="--root ${PODMAN_STORE} --runroot ${PODMAN_RUNROOT} \
     --storage-driver overlay \
     --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs \
     --storage-opt overlay.ignore_chown_errors=true"
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
fi

podman ${PODMAN_STORAGE_ARGS} run --device=/dev/dri --device=/dev/kfd  --network=host --ipc=host --group-add keep-groups -v $HOME:/workdir --workdir /workdir docker://rocm/dev-ubuntu-22.04:6.4.1 rocminfo

# to launch a shell session inside a container
#podman run -it --device=/dev/dri --device=/dev/kfd  --network=host --ipc=host --group-add keep-groups -v $HOME:/workdir --workdir /workdir docker://rocm/dev-ubuntu-22.04:6.1.2-complete
