# AIM Engine deployment and testing

These scripts install and serve models with AMD's AIM (AMD Inference
Microservices) Engine, a Kubernetes (k8s) operator for serving models on AMD
GPUs. We deploy a cluster-scoped operator (Custom Resource Definitions plus a
Helm chart) through `kubectl` and `helm`; there is no Lmod module. The scripts
target a cluster that already exists and do not provision one: provisioning from
bare metal is the job of the AMD Enterprise AI reference stack (see References),
of which AIM Engine is one component.

An AIM is the inference container that serves one model and self-selects
optimized runtime parameters for the hardware; AIM Engine is the operator that
manages those containers at scale. We run an AIM directly at level 1 and drive
AIM Engine at levels 2 to 4.

## Layout

- [`deploy/`](deploy/README.md): the `aim_deploy.sh` entrypoint, its three
  building-block scripts, and a per-level usage guide. This is what we run
  against a real cluster, from a login node or a laptop holding its kubeconfig.
- [`test/`](test/README.md): `aim_engine_test.sh`, which stands up a disposable
  `kind` cluster with real AMD GPU passthrough, runs the `deploy/` scripts on it
  end to end, and discards everything on exit.

The [`rocBudAI/aim-engine`](../../rocBudAI/aim-engine/README.md) example
consumes this service. It installs the
[rocBudAI](https://github.com/AMD-HPC/rocBudAI) profiling agent as a thin client
of a model served by `deploy/`, instead of a local ollama daemon, so the model
runs remotely on a trusted, in-boundary cluster.

## Test first

We recommend a harness run before deploying for real. It is the closest
rehearsal available, because it runs the same `deploy/` scripts: in interactive
mode it hands us the same `aim_deploy.sh --level N` commands we run in
production. It only substitutes the foundation a real cluster already provides:

- the cluster itself (`kind` with `/dev/kfd` and `/dev/dri` passthrough);
- the `amd.com/gpu` advertisement and accelerator labels a GPU Operator would
  stamp;
- a ReadWriteMany StorageClass (an in-cluster NFS provisioner) for shared storage.

A green run therefore confirms the scripts, the operator install and GPU-backed
inference. The caveat is scale: it is a single `kind` node, so it validates flow
correctness, not capacity. See [`test/README.md`](test/README.md)
for the modes, including a Kubernetes-free path for hosts without Docker or
`cgroup` v2.

## Prerequisites

- Deploy: `kubectl` and `helm` on `PATH`, and a cluster whose GPU nodes run the
  AMD GPU Operator. Installing cluster-scoped components needs cluster-admin. A
  kubeconfig that authenticates through OIDC also needs the kubelogin plugin (see
  [`deploy/README.md`](deploy/README.md)).
- Test: a GPU host with `docker` and the `/dev/kfd` and `/dev/dri` devices;
  `kind`, `kubectl`, and `helm` are fetched as user-local binaries if absent.

## References

- [AMD Enterprise AI reference stacks overview](https://enterprise-ai.docs.amd.com/en/latest/reference-stacks.html)
- [AMD Enterprise AI reference stack, on-premises installation](https://enterprise-ai.docs.amd.com/en/latest/platform-infrastructure/on-premises-installation.html)
- [Cluster Bloom](https://github.com/silogen/cluster-bloom)
- [Cluster Forge](https://github.com/silogen/cluster-forge)
