- [Exercise 6 (mdb): one prompt for all ranks](#orgf406c18)
  - [6d - install mdb into a venv](#org6fabc54)
  - [6e - launch mdb across all ranks, then attach](#org81e02cd)
  - [6f - drive every rank from one prompt](#org331f797)


<a id="orgf406c18"></a>

# Exercise 6 (mdb): one prompt for all ranks

`mdb` (<https://github.com/TomMelt/mdb>, MIT) is an MPI-aware front-end that runs a gdb on every rank and drives them from a SINGLE prompt: the interactive counterpart to the per-rank batch wrapper (which writes N separate logs). This exercise is self-contained: install mdb, then debug the same `mpi_demo` across all ranks.

-   Prerequisites: Python >= 3.10, `gdb` on PATH (mdb's `-b gdb` backend launches the `gdb` binary by name; `rocgdb` is a drop-in only if it is what `gdb` resolves to), and an MPI stack (OpenMPI, Cray MPI, or mpich >= 4.3.2).


<a id="org6fabc54"></a>

## 6d - install mdb into a venv

```
$ git clone https://github.com/TomMelt/mdb.git
$ python3 -m venv .mdb
$ source .mdb/bin/activate
$ cd mdb/
$ pip install '.[termgraph]'       # quote it (your shell treats .[...] as a glob); adds ASCII plots
$ cd ..
```

-   If SSL certificate generation fails ("Failed to generate SSL certificate"), point mdb at the system openssl and retry: `export MDB_OPENSSL=/usr/bin/openssl`.
-   `deactivate` leaves the venv; `source .mdb/bin/activate` re-enters it in a new shell.


<a id="org81e02cd"></a>

## 6e - launch mdb across all ranks, then attach

-   Build the program if you have not already: `make -C ../common mpi`.
-   Launch 4 ranks under gdb: mdb drives the MPI job itself and prints a host and port.

```
$ mdb launch -b gdb -n 4 -t ../common/mpi_demo
#   ... prints the host and port to attach to ...
```

-   In a SECOND terminal, activate the same venv and attach to that host/port:

```
$ source .mdb/bin/activate
$ mdb attach -h <host> -p <port>
(mdb)
```


<a id="org331f797"></a>

## 6f - drive every rank from one prompt

```
(mdb) command break mpi_demo.c:17   # set the breakpoint on EVERY rank
(mdb) command run                   # run every rank to it
(mdb) command print value           # value = rank*rank, shown for ALL ranks at once
(mdb) command 0,2 print value       # ... or only ranks 0 and 2
(mdb) broadcast start               # from now on every line goes to the selected ranks
(mdb) print rank
(mdb) broadcast stop
(mdb) command continue              # let them all finish
```

-   One prompt, one command, all ranks: contrast with the wrapper's N `gdb_out_<rank>.txt`.
-   For the curious: `plot value` draws the per-rank values as an ASCII chart, and a whole session can be scripted non-interactively with `mdb attach ... -x session.mdb`.
