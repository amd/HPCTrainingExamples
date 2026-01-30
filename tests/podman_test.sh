#!/bin/bash

if [[ "`printenv |grep -w CRAY |wc -l`" -lt 1 ]]; then
   module list 2>&1 | grep -q -w "rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
   module load amdflang-new >& /dev/null
   if [ "$?" == "1" ]; then
      module load amdclang
   fi
fi

podman run --device=/dev/dri --device=/dev/kfd  --network=host --ipc=host --group-add keep-groups -v $HOME:/workdir --workdir /workdir docker://rocm/dev-ubuntu-22.04:6.4.1 rocminfo

# to launch a shell session inside a container
#podman run -it --device=/dev/dri --device=/dev/kfd  --network=host --ipc=host --group-add keep-groups -v $HOME:/workdir --workdir /workdir docker://rocm/dev-ubuntu-22.04:6.1.2-complete
