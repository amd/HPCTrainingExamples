- [Exercise 6 (batch): non-interactive rocgdb, then MPI](#org41bc356)
  - [6a - batch-debug a non-MPI program, two ways](#org97f0989)
  - [6b - debug an MPI job with the per-rank wrapper](#org7f3af55)
  - [6c - change what the wrapper does](#org869da76)


<a id="org41bc356"></a>

# Exercise 6 (batch): non-interactive rocgdb, then MPI

The two wrapper scripts live in THIS directory. Run rocgdb from here; the compiled programs are in `../common/` (built by the setup step, including `make mpi`).


<a id="org97f0989"></a>

## 6a - batch-debug a non-MPI program, two ways

-   On the command line with `-ex` (one flag per command); catch a crash and print a trace:

```
$ rocgdb -batch -ex run -ex bt --args ../common/faulty
```

-   Or from a command *file* with `-x` (the basis of the MPI wrapper):

```
$ cat > cmds.txt <<'EOF'
break sum_array
run
print n
backtrace
EOF
$ rocgdb -batch -x cmds.txt --args ../common/demo
```

-   `-ex` and `-x` are interchangeable; everything must be set *before* `run`.


<a id="org7f3af55"></a>

## 6b - debug an MPI job with the per-rank wrapper

```
$ mpirun -np 4 ./gdb_mpi_wrapper_all_ranks.sh ../common/mpi_demo
#   (or: srun -n 4 ./gdb_mpi_wrapper_all_ranks.sh ../common/mpi_demo)
$ tail -n +1 gdb_out_*.txt                     # one log per rank, read from the end
```

-   Each rank writes its own `gdb_commands_<rank>.txt` and `gdb_out_<rank>.txt` (no interleaving). The log captures both rocgdb's output *and* the program's own stdout.
-   As shipped, each rank just `run`s and dumps a full backtrace.


<a id="org869da76"></a>

## 6c - change what the wrapper does

-   Edit `gdb_mpi_wrapper_all_ranks.sh`. In the block that builds the command file (the `echo "..."` lines), add a breakpoint before `run` and a print after it:

```
echo "break mpi_demo.c:17"     # add BEFORE the  echo "run"  line
echo "run"                     #   (line 17 is AFTER value is set on line 16)
echo "print value"             # add AFTER it: value = rank*rank on each rank
echo "continue"
```

-   Re-run as in 6b and check each `gdb_out_<rank>.txt`: rank *r* should report `value = r*r` at the breakpoint, then run to completion.
