# Testing AIM on a throwaway cluster

`aim_engine_test.sh` stands up a disposable `kind` cluster with real AMD GPU
passthrough so we can exercise the whole flow, including inference, before
deploying for real. It runs the sibling [`deploy/`](../deploy/README.md) scripts:
it mounts them into the node at `/aim-engine` and, in auto-run mode, calls
`aim_engine_setup.sh` directly. Everything lives in a working directory removed
on exit. It needs a GPU host with `docker` and the `/dev/kfd` and `/dev/dri`
devices; `kind`, `kubectl`, and `helm` are fetched if absent.

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

The default (`--auto-run 0`) drops us into a shell in the `kind` node, with the
`deploy/` scripts at `/aim-engine`, `kubectl` and `helm` on `PATH`, and
`KUBECONFIG` set. We run any level exactly as in production, then exit to tear it
all down:

```bash
./aim_deploy.sh --level 4
kubectl get aimservice,inferenceservice,pods -n default
```

## Hosts without Docker or cgroup v2

`kind` needs a container runtime that can host a node, and its rootless Podman
path needs `cgroup` v2, so a `cgroup` v1 host without root cannot create a
cluster (a `kind` limitation). For that case, and for anyone without sudo,
`--container-only` skips `kind`, `kubectl`, and `helm` and runs the AIM container
directly with GPU passthrough: it validates the microservice layer only, not the
operator or the Kubernetes scripts, but needs no cluster, `cgroup` v2, or sudo.
Its default leaves the model serving and drops us into a shell inside the
container (a throwaway sandbox reachable at `http://localhost:8000`);
`--auto-run 1` runs the checks and cleans up.

Once a run here is green, deploy for real per
[`deploy/README.md`](../deploy/README.md); for an overview see the
[top-level README](../README.md).
