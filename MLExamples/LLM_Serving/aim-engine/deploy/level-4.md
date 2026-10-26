# Level 4: install prerequisites and the operator, then serve

Level 4 is the turnkey path for a cluster we own: it installs the seven add-ons,
installs AIM Engine, and serves, assuming only that the GPU nodes run the AMD GPU
Operator. It needs cluster-admin, since it creates cluster-scoped resources for
both.

Set `KUBECONFIG` and `NAMESPACE` first (see
[Supplying a kubeconfig](README.md#supplying-a-kubeconfig) and
[Choosing a namespace](README.md#choosing-a-namespace)), then run the level:

```bash
export KUBECONFIG=/path/to/your/kubeconfig
export NAMESPACE=<your-namespace>
./aim_deploy.sh --level 4 --namespace "$NAMESPACE"
# serve a specific image (gated images also need HF_TOKEN exported):
./aim_deploy.sh --level 4 --namespace "$NAMESPACE" --model-image amdenterpriseai/aim-qwen-qwen3-32b:0.13.0
```

This runs `aim_engine_setup.sh --install-prereqs 1` (which runs
`aim_prereqs_setup.sh` for the add-ons, then installs the CRDs and operator) and
applies the same `AIMService` as level 2. The serve, verify, inference,
example-app, and cleanup steps are identical to [level 2](level-2.md).

The [test harness](../test/README.md) runs this exact flow on a throwaway `kind`
cluster, so a run there is the closest rehearsal of a real level-4 deploy.
