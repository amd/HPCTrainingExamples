# Installs rocprof-sys and rocprof-compute inside the ROCm PyTorch Apptainer container (see README).
# Source it (don't execute it) from version2_pytorch_fused/ after creating ./venv-pt:
#   source ./setup_profilers_apptainer.sh
# The installation only happens once; source it again in every new shell to set ROCM_PATH and LD_LIBRARY_PATH.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "Please source this script: source ./setup_profilers_apptainer.sh" >&2
    exit 1
fi

_setup_profilers() {
    if [ ! -d ./venv-pt ]; then
        echo "ERROR: ./venv-pt not found. Run this from version2_pytorch_fused/ after the container setup in the README." >&2
        return 1
    fi
    [ "$VIRTUAL_ENV" = "$PWD/venv-pt" ] || source ./venv-pt/bin/activate

    local C=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core
    local P=venv-pt/lib/python3.12/site-packages/_rocm_profiler

    if [ ! -d $P ]; then
        pip3 install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[profiler]==10.0.0" || return 1
    fi

    # ROCm tree linking into the container's ROCm
    if [ ! -e venv-pt/rocm/lib/libhsa-amd-aqlprofile64.so ]; then
        mkdir -p venv-pt/rocm/lib
        ln -s $C/{.info,bin,etc,include,libexec,share} venv-pt/rocm/
        ln -s $C/lib/* venv-pt/rocm/lib/
        ln -s libamdhip64.so.7 venv-pt/rocm/lib/libamdhip64.so
        ln -s libhsa-amd-aqlprofile64.so.1 venv-pt/rocm/lib/libhsa-amd-aqlprofile64.so
    fi

    # rocprof-compute analyze needs exactly pinned packages that would downgrade the container's numpy/pandas.
    if [ ! -x ./venv-analyze/bin/pip ]; then
        python3 -m venv ./venv-analyze
        ./venv-analyze/bin/pip install -r $P/libexec/rocprofiler-compute/requirements.txt || return 1
    fi

    case ":$LD_LIBRARY_PATH:" in *":$PWD/venv-pt/rocm/lib:"*) ;; *)
        export LD_LIBRARY_PATH=$PWD/venv-pt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} ;;
    esac
    export ROCM_PATH=$PWD/venv-pt/rocm
    echo "Profilers ready: ROCM_PATH=$ROCM_PATH"
}
_setup_profilers
unset -f _setup_profilers
