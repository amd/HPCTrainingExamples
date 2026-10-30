# Exercise 1 (running): launch, control the start, and attach

The programs are in `../common/` (build them first, see `../common/README`). Run rocgdb from this directory; look at `../common/demo.c` alongside.


## 1a - launch under rocgdb: run / start / next / continue

```
$ rocgdb --args ../common/demo
(gdb) run          # runs freely to the end: prints "sum = 31, ..."
(gdb) start        # re-launch, stop at the first line of main
(gdb) next         # step over one source line (stays in main)
(gdb) next         # ... again; watch it walk down main
(gdb) continue     # let it run to the end
```

-   `run` lets the program go; `start` is `run` plus a one-shot breakpoint at `main`.
-   `next` executes one line at a time *without* descending into calls.
-   If rocgdb asks `Enable debuginfod for this session? (y or [n])`, answer `n` for this course. To stop it asking in future, put `export DEBUGINFOD_URLS=` in your shell startup - an empty value disables the automatic download of system debug info.


## 1b - attach to an already-running process

-   Start `demo` in its long idle loop in the background, then attach to it:

```
$ ../common/demo --spin &      # backgrounded; note the PID (printed as [1] <pid>, also $!)
$ rocgdb -p $!                    # attach to that PID
(gdb) backtrace                # you are somewhere inside the sleep, deep in libc
```

-   You will NOT be sitting in `main`: the innermost frame is a libc sleep routine (e.g. `__GI___clock_nanosleep`). Read *up* the trace to the *outermost* (highest- numbered) frame - that one is in `main`, at one of demo.c lines 39-40, almost certainly line 40 (the `usleep` call, where essentially all the time is spent).
-   Getting the prompt back and cleaning up - three ways:
    -   `continue` resumes the endless loop and never returns to the prompt; press `Ctrl-c` to interrupt it. So `continue` cannot be your *last* step here.
    -   `detach` leaves the process running - you must then stop it from the shell (`kill %1`, or `kill $!`).
    -   `kill` (at the rocgdb prompt) ends the process immediately: no `continue` or `detach` afterwards, and no shell kill needed - it is already gone.
