# perf_events security & access control

> Part of the [CG profiler guides](README.md). Companion to [Linux perf](perf.md).
> The runnable demo is [`CG-CPU/perf_security_demo.sh`](../../CG-CPU/perf_security_demo.sh);
> the full plan is [chapter 9](../09-perf-security-demo.md).

`perf` (see [perf.md](perf.md)) reads hardware performance counters,
execution-context registers, and sampled addresses — data that can leak sensitive
information from *other* processes. The kernel therefore gates `perf_events` behind
an access-control model you must understand to (a) know why a command is denied and
(b) request the right, least-privilege fix from your admins.

## The `perf_event_paranoid` scope ladder

An unprivileged user's scope is set by `/proc/sys/kernel/perf_event_paranoid`:

| Value | What an unprivileged user may do |
|:-----:|----------------------------------|
| **-1** | Everything; per-cpu `perf_event_mlock_kb` limit ignored (**least secure**) |
| **>= 0** | Per-process **and** system-wide (`-a`); excludes raw/ftrace tracepoints |
| **>= 1** | Per-process **only** (no system-wide); user + kernel events |
| **>= 2** | Per-process, **user-space events only** (no kernel-space sampling) |

On this cluster the **exclusive compute nodes run `paranoid = -1`** (full access —
verified), while the **login node is restricted** (`paranoid = 4`). That is why the
rule throughout these guides is *profile inside an allocation*. `perf stat`/`perf
record` on your own `cg_cpu` (per-process) work at any level `<= 2`; `perf stat -a`
(system-wide, e.g. uncore/IMC memory-bandwidth counters) needs `<= 0`.

## CAP_PERFMON — the least-privilege alternative

Lowering `paranoid` system-wide is blunt. The secure alternative is the
**`CAP_PERFMON`** capability (Linux ≥ 5.9), which grants perf scope *without* full
root and is preferred over the legacy `CAP_SYS_ADMIN`:

```bash
groupadd perf_users
chgrp perf_users "$(command -v perf)" && chmod o-rwx "$(command -v perf)"
setcap "cap_perfmon,cap_sys_ptrace,cap_syslog=ep" "$(command -v perf)"
getcap "$(command -v perf)"   # cap_sys_ptrace,cap_syslog,cap_perfmon+ep
```

Members of `perf_users` then profile at any paranoid level. (`cap_syslog` allows
resolving kernel symbols via `/proc/kallsyms`; `cap_sys_ptrace` is not needed on
≥ 5.9 but is harmless; add `cap_ipc_lock` for `perf top`.) Where `setcap` is not
possible (e.g. `nosuid` filesystem) the guide's `capsh`/`sudo` "privileged perf
shell" achieves the same via the ambient capability set.

The [demo](../09-perf-security-demo.md) automates this as a **least-privilege
overlay on one ROCm release**:
[`perf_users_setup.sh`](../../CG-CPU/perf_users_setup.sh) creates the group, adds
users, and gates (`chgrp`+`chmod o-rwx`) + `setcap`s the profilers of a single
release, while **leaving `perf_event_paranoid` open** so profilers from *other*
releases keep working unchanged. A real-world wrinkle it handles: the local `perf`
ELF is on **xfs** (`setcap` works), but ROCm is on **NFS**, which cannot hold
file-capability xattrs — there group members get `CAP_PERFMON` via the `capsh`
privileged shell instead. To **prove** the overlay,
[`perf_paranoid_test.sh`](../../CG-CPU/perf_paranoid_test.sh) *raises*
`perf_event_paranoid` and shows an ordinary `perf_users` member still lands a
system-wide `perf stat -a` via `CAP_PERFMON`, then restores the original value.

## Resource limits that bite multi-rank runs

Two per-process resource limits (not security *scope*, but they cause the same
"cannot proceed" symptom) matter when profiling all MPI ranks at once:

- **`perf_event_mlock_kb`** (here `516` KiB, *per cpu*) — the ring-buffer budget.
  The first `perf` process can grab it all and starve the other ranks. Cap each
  rank's buffer with `perf record --mmap-pages=N` (or `perf top -m N`).
- **`RLIMIT_NOFILE`** (`ulimit -n`) — perf opens ≥ `events × cpus` file descriptors;
  large event lists on many cores can exhaust it.

`CAP_IPC_LOCK` lifts the `mlock` limit for privileged perf users.

## Why this does not block on the ROCm version

`perf_events` security is a **kernel** feature — orthogonal to ROCm. The
[demo](../09-perf-security-demo.md) therefore pins a **single** `rocm` module (only
to build `cg_cpu`) and does **not** sweep ROCm versions: the paranoid ladder,
`CAP_PERFMON`, and the resource limits behave identically under every ROCm module on
a given kernel. Because the setup **leaves `perf_event_paranoid` open**, you can
apply the `perf_users`/`CAP_PERFMON` overlay to **one release's** profilers and every
*other* release keeps working untouched. (Contrast the *GPU* ATT decoder in
[rocprofv3 §4](rocprofv3.md#4-instruction-level-advanced-thread-trace-att), which
genuinely needs `rocm >= 7.12`.)

## See also

- [Linux perf](perf.md) — the profiler this governs
- [chapter 9](../09-perf-security-demo.md) — the full demo and setup scripts
