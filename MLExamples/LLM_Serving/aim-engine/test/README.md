# Testing AIM on a throwaway cluster

`aim_engine_test.sh` stands up a disposable `kind` (Kubernetes on Docker) cluster with real AMD GPU
passthrough so we can exercise the whole flow, including inference, before
deploying for real. It runs the sibling [`deploy/`](../deploy/README.md) scripts:
it mounts them into the node at `/aim-engine` and, in auto-run mode, calls
`aim_engine_setup.sh` directly. Everything lives in a working directory removed
on exit. It needs a GPU host with `docker` and the `/dev/kfd` and `/dev/dri`
devices; `kind`, `kubectl`, and `helm` are fetched if absent. On a shared HPC
cluster we rarely have such a host directly: we obtain one through the scheduler,
for example a Slurm allocation on a GPU node (`salloc -N1 --gpus=1`, or `srun`)
where `docker` and the devices are available, and we run the script from inside
that allocation.

The harness provides only the foundation a real cluster already has, leaving the
prerequisite and operator installs to the `deploy/` scripts: it creates the
cluster, deploys the ROCm device plugin (advertising `amd.com/gpu`), injects the
accelerator labels a GPU Operator would set, and installs an in-cluster NFS
provisioner for the ReadWriteMany StorageClass the operator's cache needs. That
is why a run here is the closest rehearsal of a real deploy: the scripts and the
operator and inference paths are identical, and only the cluster, the GPU
advertisement and labels, and shared storage are substituted.

## Modes

```bash
./aim_engine_test.sh                      # bring up kind + GPU, then drop into a sandbox shell
./aim_engine_test.sh --auto-run 1         # install, idempotency, and inference checks, then clean up
./aim_engine_test.sh --base-image-only 1  # skip the operator; only the base-image serve check
./aim_engine_test.sh --container-only 1   # no Kubernetes; run the AIM container directly
```

The `--auto-run 1` path runs three checks in sequence: it installs the
prerequisites and the operator, then re-runs the same install to confirm it is
idempotent, and finally serves a model and issues a real inference request. The
idempotency check matters because deploy scripts get re-run in practice (a
retried job, a repeated command): an idempotent install detects that everything
is already present and exits success without reinstalling or erroring, so a
second run is a no-op rather than a failure. The `--replace 1` flag is the
deliberate escape hatch for when we do want to reinstall.

The default (`--auto-run 0`) drops us into a shell in the `kind` node, with the
`deploy/` scripts at `/aim-engine`, `kubectl` and `helm` on `PATH`, and
`KUBECONFIG` set. We run any level exactly as in production, then exit to tear it
all down:

```bash
./aim_deploy.sh --level 4
kubectl get aimservice,inferenceservice,pods -n default
```

## Hosts without Docker or cgroup v2

`kind` needs a container runtime to host its node. The harness prefers Docker and
falls back to Podman when Docker is absent, which on a shared HPC node is the
usual case. That fallback runs into a `cgroup` limit: `kind`'s rootless mode, the
only way to run it without a privileged daemon, requires `cgroup` v2, whereas a
root-level Docker daemon can host a node on `cgroup` v1. So where we have only
rootless Podman on a `cgroup` v1 host, `kind` cannot create a cluster at all.

For that case, `--container-only` skips `kind`, `kubectl`, and `helm` and runs
the AIM container directly with GPU passthrough. A single container is far less
demanding than a Kubernetes node: it needs neither a cluster nor `cgroup` v2, so
rootless Podman on a `cgroup` v1 host runs it without sudo. This validates the
AIM microservice layer alone, a GPU serving a model, not the operator or the
Kubernetes scripts. By default it leaves the model serving and drops us into a
shell inside the container, a throwaway sandbox reachable at
`http://localhost:8000`; `--auto-run 1` runs the checks and cleans up instead.

Once a run here is green, deploy for real per
[`deploy/README.md`](../deploy/README.md); for an overview see the
[top-level README](../README.md).
