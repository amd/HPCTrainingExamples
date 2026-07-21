# 9. perf_events security — implementation & demo plan

This chapter turns the [perf_events security reference](profilers/perf-security.md)
(a distillation of the
[kernel perf-security guide](https://docs.kernel.org/admin-guide/perf-security.html))
into something you can **run** on the CG example.

The demo is a **least-privilege access model**, not a "lower the paranoid knob"
hack. Concretely:

1. Create a **`perf_users`** group and add the users who are allowed to profile.
2. **Gate the profiling executables of ONE ROCm release** (plus the local `perf`
   ELF) to that group — `chgrp perf_users` + `chmod o-rwx` — and stamp
   **`CAP_PERFMON`** on them (`setcap`) where the filesystem supports file
   capabilities.
3. **Leave `perf_event_paranoid` open** (unchanged). Profilers from *other* ROCm
   releases are not capability-stamped, so they must keep relying on the open
   paranoid setting — tightening it would break them. The `perf_users` +
   `CAP_PERFMON` setup is a least-privilege **overlay on one release**, not a
   global change.

Two deliverables implement this:

| Script | Role | Privilege |
|--------|------|-----------|
| [`CG-CPU/perf_users_setup.sh`](../CG-CPU/perf_users_setup.sh) | **Admin**: create group, add users, gate + `setcap` one ROCm release's profilers | root (`--apply`); **dry-run by default** |
| [`CG-CPU/perf_security_demo.sh`](../CG-CPU/perf_security_demo.sh) | **User**: verify the posture (paranoid, group membership, caps) and exercise the permitted perf scopes on `cg_cpu` | none (read-only) |
| [`CG-CPU/perf_paranoid_test.sh`](../CG-CPU/perf_paranoid_test.sh) | **Proof**: show an ordinary user can still profile under a *raised* `perf_event_paranoid` via `CAP_PERFMON`. Two modes: `verify` (no root; reads the prolog-set level) and `raise` (self-contained `sudo`, auto-restores) | none / `sudo` |
| [`CG-CPU/slurm/perf_paranoid.prolog.sh`](../CG-CPU/slurm/perf_paranoid.prolog.sh) | **Admin (SLURM)**: raise `perf_event_paranoid` per job when the user passes `--comment=paranoid[=N]` | installed as root; runs as root at job start |
| [`CG-CPU/slurm/perf_paranoid.epilog.sh`](../CG-CPU/slurm/perf_paranoid.epilog.sh) | **Admin (SLURM)**: restore the original `perf_event_paranoid` at job end | installed as root; runs as root at job end |

## Goals

1. Grant profiling rights to a **named group**, not the whole node, via the
   kernel's `CAP_PERFMON` least-privilege capability (preferred over
   `CAP_SYS_ADMIN` and over blanket `paranoid=-1`).
2. Pilot the secure setup on **one ROCm release** while **every other release
   keeps working unchanged**.
3. Keep it safe to demonstrate: the admin script changes nothing unless run with
   `--apply` as root; the user script is entirely read-only.

## Can we demo on one ROCm version without blocking on others? — Yes, by design

This is the whole point of leaving paranoid open:

- `perf_events` security is a **kernel** feature, identical under every `rocm/*`
  module. So the group/capability mechanism is version-independent.
- We stamp `CAP_PERFMON` on the profilers of **one** release (`rocm/7.2.3` here, a
  stable pick — note it predates and therefore lacks the ATT decoder, underscoring
  that perf security is independent of that GPU feature) and gate them to
  `perf_users`.
- Because `perf_event_paranoid` is **left open**, the **other** ROCm releases —
  whose profilers are *not* stamped — continue to work exactly as before. Nothing
  about them changes.

So you can roll the secure posture out **one release at a time** with zero risk to
users on the other releases. The only genuinely ROCm-version-gated item anywhere
in this chapter is the *GPU* ATT/SQTT decoder (§A.4/§H.4, needs `rocm >= 7.12`),
which is unrelated to perf security.

> **What actually depends on the kernel (not ROCm).** `CAP_PERFMON` needs Linux
> ≥ 5.9; this cluster is on 6.8. And **file capabilities need a supporting
> filesystem**: the local `perf` ELF lives on **xfs** (`setcap` works), but ROCm
> is on **NFS**, which cannot hold `security.capability` xattrs — so ROCm-release
> profilers are *gated* to the group but get `CAP_PERFMON` via the guide's `capsh`
> privileged shell (or simply via the open paranoid). The setup script detects the
> filesystem per target and does the right thing.

## Implementation plan

**Status: implemented and dry-run-verified** (kernel 6.8, `rocm/7.2.3`,
`/nfsapps` = NFS, `perf` ELF on xfs). The admin script
[`perf_users_setup.sh`](../CG-CPU/perf_users_setup.sh) performs:

1. **Group + membership.** `groupadd -f perf_users`; `usermod -aG perf_users <u>`
   for each `--users`. (Members must re-login for the group to take effect.)
2. **Target discovery.** The local `perf` **ELF** (resolved through the Ubuntu
   `/usr/bin/perf` wrapper to `/usr/lib/linux-tools/$(uname -r)/perf`) plus the
   `--profilers` found under `--rocm/bin` (default: `rocprofv3`, `rocprof-compute`,
   `rocprof-sys-run`, `rocprof-sys-sample`, `rocprof-sys-instrument`).
3. **Gate + capability-stamp, per filesystem.** For each target:
   `chgrp perf_users` + `chmod o-rwx` (restrict to the group), then — only where
   `stat -f`/`findmnt` show a cap-capable, non-`nosuid` FS — `setcap
   cap_perfmon,cap_syslog=ep`. On NFS it prints the `capsh`/open-paranoid fallback
   instead of a broken `setcap`.
4. **Leave paranoid alone.** It prints the current `perf_event_paranoid` and an
   explicit note that it is **not** modified, so other releases keep working.

The **proof** that the overlay actually works is a separate step,
[`perf_paranoid_test.sh`](../CG-CPU/perf_paranoid_test.sh), run *once* under `sudo`
inside an **exclusive** allocation. It:

1. Saves the current `perf_event_paranoid` and installs a restore trap.
2. **Raises** it (default `2`; try `3` for stricter) — i.e. tightens security.
3. Runs perf **as the ordinary `$SUDO_USER`** (via `runuser`, not as root):
   a per-process probe and a system-wide `perf stat -a` probe.
4. **Verdict:** if the system-wide probe passes at the raised level *and* the perf
   ELF carries `CAP_PERFMON`, the overlay is proven — an unprivileged user profiled
   a tightened node purely by capability, not by root. If it is denied, it tells
   you to run `perf_users_setup.sh --apply` first.
5. **Restores** the original `perf_event_paranoid` on exit (trap covers INT/TERM).

This is the crux of the demo: it shows the least-privilege grant is real and
survives a security tightening — while, back at the normal open setting, every
other ROCm release is unaffected.

### Preferred: raise paranoid in the SLURM prolog/epilog (`--comment=paranoid`)

Rather than hand each tester `sudo` to poke the sysctl, do the privileged change
**once, in the job prolog**, gated on a job comment. The user opts in per job with
`--comment=paranoid[=N]`; the prolog raises `perf_event_paranoid`, the epilog
restores it. Two drop-ins implement this (install them as root):

- [`slurm/perf_paranoid.prolog.sh`](../CG-CPU/slurm/perf_paranoid.prolog.sh)
- [`slurm/perf_paranoid.epilog.sh`](../CG-CPU/slurm/perf_paranoid.epilog.sh)

**How the comment maps to a level** (deployed form is `paranoid<num>`, no `=`;
parsing verified against the §M scope ladder):

| `--comment=` | perf_event_paranoid | Effect on an unprivileged user |
|--------------|:-------------------:|--------------------------------|
| `paranoid`   | **2** (default)     | per-process, user-space only; system-wide `-a` needs `CAP_PERFMON` |
| `paranoid1`  | 1                   | per-process (user+kernel); no system-wide |
| `paranoid2`  | 2                   | per-process, user-space only |
| `paranoid3`  | 3                   | vendor-strict: essentially nothing without `CAP_PERFMON` |
| `paranoid-1` | -1                  | fully open (only useful to *reset* on a restricted node) |
| *(none)*     | *unchanged*         | job untouched — no-op |

`paranoid=N` and bare `paranoid` are also accepted. The comment may be embedded in
a list (`foo,paranoid2,bar`); out-of-range values are rejected and logged;
`paranoidish`/`notparanoid` do **not** match. The prolog
**stashes** the pre-job value (`/run/perf_paranoid.$SLURM_JOB_ID`) so the epilog
restores it exactly; if the stash is lost it falls back to
`PERF_PARANOID_DEFAULT` (default `-1`, the open value these exclusive compute
nodes normally run). An optional **raise-only** guard (commented out in the
prolog) refuses to *lower* security via `--comment`.

**Install (as root, `sudo`):** point `slurm.conf` at them (or drop into your
existing `prolog.d`/`epilog.d`):

```conf
# /etc/slurm/slurm.conf   (then: scontrol reconfigure)
Prolog=/etc/slurm/perf_paranoid.prolog.sh
Epilog=/etc/slurm/perf_paranoid.epilog.sh
```

With that in place testers need **no sudo**: `--comment=paranoidN` raises it, and
`perf_paranoid_test.sh` (verify mode) confirms profiling still works. The
self-contained `sudo ./perf_paranoid_test.sh N` remains as a fallback where the
prolog is not installed.

> **Verified** (`ppac-pl1-s24-16`, kernel 6.8): `srun --exclusive
> --comment=paranoid2` brought the node up at `perf_event_paranoid = 2`;
> per-process profiling worked (perf auto-scoped to `:u`), system-wide `-a` was
> denied (no `CAP_PERFMON` yet), and `rocm/7.2.3`'s `rocprofv3`/`rocprof-compute`
> ran unaffected. One gotcha: `SLURM_JOB_COMMENT` is **not exported into the job
> step**, so the prolog reads the comment via its `scontrol show job` fallback —
> that fallback is essential, don't drop it.

### Do the ROCm 7.2.3 profilers still work at paranoid=2? — Yes (verified)

The GPU profilers use rocprofiler-sdk / the KFD, **not** `perf_events`, so
`perf_event_paranoid` does not gate them. Verified end-to-end on `cg_gpu`
(gfx942, MI300A) inside `--comment=paranoid2` with
[`CG-GPU/prof_all_723_test.sh`](../CG-GPU/prof_all_723_test.sh):

| Profiler (ROCm 7.2.3) | Result at paranoid=2 | Notes |
|-----------------------|----------------------|-------|
| `rocprofv3 --sys-trace` | **PASS** | clean |
| `rocprofv3 --pmc` (HW counters) | **PASS** | clean |
| `rocprof-compute profile` | **PASS** | counters + empirical roofline both OK (self-contained Nuitka build via the `rocm` module) |
| `rocprof-sys-sample` | **PASS** (trace written) | writes the full Perfetto trace, then exits 134 on a glibc **double-free at teardown** (rocprofiler-systems v1.3.0 bug) — unrelated to paranoid |
| `rocprof-sys-run` | **PASS** (trace written) | same teardown double-free |

**None** emitted a `perf_event_paranoid` message. Two setup gotchas the script
handles (neither is about paranoid):

1. **GPUs need `--gres`.** An `--exclusive` job alone hides `/dev/dri` in the
   cgroup, so `rocminfo` sees 0 GPUs and every GPU app aborts with "No HIP capable
   device". Request `--gres=gpu:4` (or `:1`).
2. **`module load rocm/<ver>` pulls in the patches automatically.** The
   `rocm/<ver>` modulefile prepends `rocm-patches-<ver>/rocprof-compute/bin`, so
   `rocprof-compute` resolves to the self-contained **Nuitka single-file
   executable** (bundling pandas/dash/matplotlib/…), and `ROCM_PATH`/`LD_LIBRARY_PATH`
   are set correctly (the roofline microbench's bundled helper then finds
   `libamdhip64.so.7`). No manual `PATH` surgery is needed; if the site *also* ships
   a dedicated `rocm_patches` module, load it too.
   **Gotcha:** never pipe or command-substitute the `module` command
   (e.g. `module load rocm/7.2.3 | tail`). A pipe runs the `module` shell function
   in a **subshell**, so its `eval`'d environment changes (`PATH`, `ROCM_PATH`,
   `LD_LIBRARY_PATH`) are discarded — and then the patches path never lands on
   `PATH` and `rocprof-compute` falls back to the plain Python script. Call `module`
   plainly. Set `ROCM_VER` to target another release.

Design rules:

- **Dry-run by default** — prints every command, changes nothing; `--apply`
  requires root (guarded).
- **Filesystem-aware** — never emits a `setcap` that would fail on NFS.
- **Least privilege** — `cap_perfmon,cap_syslog` only (not `cap_sys_admin`;
  `cap_syslog` is for `/proc/kallsyms` kernel-symbol resolution). Add
  `cap_ipc_lock` only if you need `perf top`.
- **Idempotent-ish** — `groupadd -f`, `usermod -aG`, and re-`setcap` are safe to
  re-run.

### Verified dry-run (excerpt)

```
1) Create group and add users
    groupadd -f 'perf_users'
    usermod -aG 'perf_users' 'alice'
2) Gate + capability-stamp the profiling executables
  target: /usr/lib/linux-tools/6.8.0-…/perf
    chgrp 'perf_users' '…/perf'
    chmod o-rwx '…/perf'
    setcap 'cap_perfmon,cap_syslog=ep' '…/perf'      # xfs -> works
  target: /nfsapps/…/rocm-7.2.3/bin/rocprofv3
    chgrp 'perf_users' '…/rocprofv3'
    chmod o-rwx '…/rocprofv3'
  # NFS: no file-capability support -> use capsh shell or open paranoid
3) perf_event_paranoid — LEFT UNCHANGED on purpose
  # current perf_event_paranoid = -1 (NOT modified)
```

## Demo plan (how to present it)

```bash
# 0. (Admin, once) install the SLURM prolog/epilog so testers need no sudo:
sudo cp CG-CPU/slurm/perf_paranoid.prolog.sh /etc/slurm/
sudo cp CG-CPU/slurm/perf_paranoid.epilog.sh /etc/slurm/
#   set Prolog=/etc/slurm/perf_paranoid.prolog.sh, Epilog=... ; scontrol reconfigure

# 1. Get an exclusive node AND ask the prolog to tighten paranoid for this job:
salloc -p PPAC_MI300A_SPX --exclusive -t 00:20:00 --comment=paranoid2
module load rocm/7.2.3 openmpi
cd CG-CPU && make CXXFLAGS="-O3 -g -std=c++17"

# 2. Show the overlay plan (dry-run, no changes), then apply it once:
./perf_users_setup.sh --users "$USER" --rocm "$ROCM_PATH"
sudo ./perf_users_setup.sh --apply --users "$USER" --rocm "$ROCM_PATH"

# 3. From a fresh login (so group membership is live), verify + exercise:
./perf_security_demo.sh

# 4. PROVE it (no sudo): the prolog already set paranoid=2 for this job.
./perf_paranoid_test.sh                # verify mode: profiles as you via CAP_PERFMON
#   Fallback where the prolog is NOT installed:
#   sudo ./perf_paranoid_test.sh 2     # raise/restore it yourself
```

Talking points:

1. **Step 1** shows the *whole* privileged change up front — a group, some
   `usermod`s, and per-file `chgrp`/`setcap`. Nothing global; `perf_event_paranoid`
   is explicitly left untouched.
2. **xfs vs NFS** — the `perf` ELF gets a real `CAP_PERFMON`; the NFS-resident
   ROCm profilers are gated to the group and fall back to `capsh`/open paranoid.
   This is the practical reality on most clusters and the demo names it.
3. **Step 3** (`perf_security_demo.sh`) prints `perf_users member = yes` and
   `getcap (perf ELF) = …cap_perfmon+ep`, then runs per-process and system-wide
   perf on `cg_cpu` — the memory-bound CPU signature (ties to §K).
4. **Step 4** (`perf_paranoid_test.sh`) is the proof: at the tightened level, an
   ordinary user still lands a system-wide `perf stat -a` because the perf ELF
   carries `CAP_PERFMON`. Preferably the **prolog** raised paranoid (via
   `--comment=paranoid2`), so the tester needs no root at all; the self-contained
   `sudo` mode is a fallback and runs perf as `$SUDO_USER`, never as root.
5. **The version-independence payoff**: switch to any *other* `rocm` module and its
   profilers still work — because paranoid stayed open — even though only the
   7.2.3 profilers carry the new group/cap posture.

## Validation checklist

- [x] `perf_users_setup.sh`: `bash -n` clean; dry-run verified end-to-end.
- [x] Filesystem detection correct: xfs `perf` ELF → `setcap`; NFS ROCm → gate +
      fallback note.
- [x] Never modifies `perf_event_paranoid`; prints current value + rationale.
- [x] `--apply` refuses to run as non-root.
- [x] `perf_security_demo.sh` reports group membership + perf-ELF caps; read-only.
- [x] `perf_paranoid_test.sh`: `bash -n` clean; **verify mode** runs with no root
      (verified live on the login node at `paranoid=4` — both probes correctly
      DENIED, sysctl untouched); raise mode's non-root path is the `sudo` fallback.
- [x] Prolog/epilog: `bash -n` clean; comment parser verified for
      `paranoidN`/`paranoid=N`/bare/embedded/out-of-range/`paranoidish` (no false match).
- [x] **End-to-end verified** on `ppac-pl1-s24-16` (kernel 6.8.0-134): an exclusive
      `srun --comment=paranoid2` job came up with `perf_event_paranoid = 2` (login =
      4), per-process `perf stat ./cg_cpu`-style profiling worked (auto-scoped to
      `:u`), system-wide `-a` was correctly denied (no `CAP_PERFMON`), and
      `rocm/7.2.3`'s `rocprofv3`/`rocprof-compute` ran fine (unaffected by paranoid).
      NB: `SLURM_JOB_COMMENT` is **unset inside the job step**, so the prolog's
      `scontrol show job` fallback is what reads the comment — keep it.
- [ ] (Manual, needs admin) after `perf_users_setup.sh --apply`, re-run the same
      job and confirm the system-wide `-a` probe also PASSes via `CAP_PERFMON`.
- [ ] (Manual, needs admin) `--apply` on a real node, then re-login and confirm
      `perf_security_demo.sh` shows `perf_users member = yes` and the perf ELF caps.

## See also

- [perf_events security & access control](profilers/perf-security.md) — the reference.
- [Linux perf](profilers/perf.md) — the profiler this secures.
- [Kernel perf-security guide](https://docs.kernel.org/admin-guide/perf-security.html) — the upstream source (Privileged Perf users groups, CAP_PERFMON, resource control).
