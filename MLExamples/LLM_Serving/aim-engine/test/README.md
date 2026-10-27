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
cluster; deploys the ROCm device plugin, the component that makes the node's GPUs
schedulable by advertising them to Kubernetes as the `amd.com/gpu` resource
(Slurm's GRES, what `--gpus` requests); injects the accelerator labels a GPU
Operator would set; and installs an in-cluster NFS provisioner for a
ReadWriteMany StorageClass, on-demand storage many pods can mount read-write at
once (like a shared Lustre or GPFS filesystem), which the operator's cache needs.
That
is why a run here is the closest rehearsal of a real deploy: the scripts and the
operator and inference paths are identical, and only the cluster, the GPU
advertisement and labels, and shared storage are substituted.

## Modes

```bash
./aim_engine_test.sh                      # bring up the throwaway cluster, then open a sandbox shell to drive it by hand
./aim_engine_test.sh --auto-run 1         # run the whole install-and-inference sequence automatically, then clean up
./aim_engine_test.sh --base-image-only 1  # quick check: serve a model straight from the AIM container, no operator
./aim_engine_test.sh --container-only 1   # no Kubernetes at all: run the AIM container on its own
```

Two terms recur below. An operator is a program we install into the cluster that
runs continuously and automates what a person would otherwise do by hand. The
contrast with Slurm is the useful part: with Slurm we write an `sbatch` script and
submit a job that runs to completion, whereas here we hand the operator a short
declarative request (an `AIMService`, meaning "serve this model") and it does the
rest, pulling the model, starting the serving pods (the containers running on the
nodes), wiring up networking, and setting resource limits, then keeps the service
up and restarts anything that fails, all without us watching. Installing the AIM
Engine operator also pulls in a stack of supporting add-ons. The base image, by
contrast, is that same AIM model container run on its own, as a plain pod, with
nothing orchestrating it.

The modes differ in how much they set up and therefore what we can exercise. The
no-flag default is the full rehearsal and a superset of the rest: it also lays
down what the operator needs (the GPU labels and shared storage), then drops us
into a sandbox shell where we run any level by hand, either the operator-less
base-image serve (`aim_deploy.sh --level 1`) or the full operator path
(`--level 2` through `--level 4`), with inference and inspection in between.
`--base-image-only` is the narrow, fast case: it brings up only what a bare serve
needs (a GPU-advertising cluster, without the operator's labels or shared storage)
and limits us to the base-image serve check, so we confirm a GPU serves a model
from the AIM image but cannot drive the operator or `AIMService` path here.
`--container-only` goes further and drops Kubernetes entirely.

The `--auto-run 1` path runs three checks in sequence: it installs the
prerequisites and the operator, then re-runs the same install to confirm it is
idempotent, and finally serves a model and issues a real inference request. The
idempotency check matters because deploy scripts get re-run in practice (a
retried job, a repeated command): an idempotent install detects that everything
is already present and exits success without reinstalling or erroring, so a
second run is a no-op rather than a failure. The `--replace 1` flag is the
deliberate escape hatch for when we do want to reinstall.

Inside that default sandbox shell, the `deploy/` scripts are at `/aim-engine`,
`kubectl` (the cluster's command-line client) and `helm` are on `PATH`, and
`KUBECONFIG` points it at this cluster, so we run the levels exactly as in
production; `exit` tears it all down:

```bash
./aim_deploy.sh --level 4
kubectl get aimservice,inferenceservice,pods -n default
```

## Hosts where rootless `kind` cannot start a cluster

`kind` needs a container runtime to host its node. The harness prefers Docker and
falls back to Podman when Docker is absent, which on a shared HPC node is the
usual case. That fallback runs into a `cgroup` limit: `kind`'s rootless mode, the
only way to run it without a privileged daemon, requires `cgroup` v2, whereas a
root-level Docker daemon can host a node on `cgroup` v1. So where we have only
rootless Podman on a `cgroup` v1 host, `kind` cannot create a cluster at all.

A second block appears even on `cgroup` v2 when we run under a scheduler. A Slurm
allocation places our shell in the job step's own `cgroup` (`slurmstepd.scope`),
which lacks the systemd `Delegate=yes` property and the delegated `cpu` and `io`
controllers that `kind`'s rootless provider requires, so cluster creation fails
with a `Delegate=yes` error even though `cgroup` v2 is present. That delegation
can only be set on `user@.service`, not on the Slurm step that actually hosts our
processes, so enabling it does not reach the Podman we launch inside the
allocation.

For either case, `--container-only` skips `kind`, `kubectl`, and `helm` and runs
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
