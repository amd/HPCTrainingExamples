# Exercise 2 (stepping): next, step, and until

Programs are in `../common/`; run rocgdb from this directory. Key lines in `demo.c`:

-   the call `sum_array(data, 8)` is line 32
-   the loop body `total = add(...)` is line 22
-   `return total` is line 25; the never-taken `printf("overflow!")` is line 24.
-   Parts 2a-2c reuse ONE rocgdb session - do not quit rocgdb between them.


## 2a - next OVER a call

```
$ rocgdb ../common/demo
(gdb) break demo.c:32      # the sum_array(...) call
(gdb) run
(gdb) next                 # runs the WHOLE call, stops on the next line of main
```

-   For the curious: `nexti` does the same at the machine-instruction level.


## 2b - step INTO a call

-   Stay in the SAME session - the breakpoint at line 32 is still set:

```
(gdb) run                  # re-runs and stops again at line 32
                           #   (if you had quit after 2a there would be no breakpoint,
                           #    and "run" would go straight to the end - "step" then invalid)
(gdb) step                 # 1: into sum_array (line 20)
(gdb) step                 # 2: line 21
(gdb) step                 # 3: line 22
(gdb) step                 # 4: into add (line 12) - four steps to get here
(gdb) bt                   # see add <- sum_array <- main
```

-   For the curious: `stepi` steps a single instruction.


## 2c - step and a LIBRARY function

-   Same session again. Line 31 calls `strlen` (libc); add a breakpoint and step:

```
(gdb) break demo.c:31
(gdb) run                  # stops at line 31 (the strlen call)
(gdb) step                 # usually steps OVER strlen: you land on line 32
```

-   `step` only stops in code that has line information. Your program has it (`-ggdb`), but the C library normally does not, so `step` treats `strlen` the way `next` does. That is what you see on most clusters, Hunter included.
-   Where the C library's debug symbols are installed (a `libc6-dbg` or `glibc-debuginfo` package, or fetched through debuginfod), `step` does descend: you land in the library's implementation, e.g. `__strlen_evex`, usually hand-written assembly shown with a "No such file" warning. `finish` (run until the current function returns) brings you back to `main`.
-   To follow the call without those symbols, step single instructions instead:

```
(gdb) run                  # back to line 31 (answer y to restart)
(gdb) stepi                # three single instructions: the third enters
(gdb) stepi                #   strlen@plt, the stub through which your program
(gdb) stepi                #   calls a shared-library function
(gdb) bt                   # strlen@plt <- main
(gdb) finish               # runs the call to completion: back in main, line 31
```


## 2d - until: run out of a loop

-   Clear the leftover breakpoints from 2a/2c first (or quit and start rocgdb afresh):

```
(gdb) delete
(gdb) tbreak demo.c:22     # temporary breakpoint in the loop body
(gdb) run
(gdb) until                # steps to the loop test (line 21)
(gdb) until                # runs ALL remaining iterations, stops past the loop (line 23)
(gdb) print total          # 31 - the loop finished
```

-   Contrast with `next`, which would stop on every one of the 8 iterations.


## 2e - until <line>: reached vs. run out of frame

-   Two outcomes of `until LOCATION` (stop at LOCATION, *or* the frame returns first). Each block below is a fresh `run` (answer `y` if rocgdb asks to restart):

```
(gdb) tbreak demo.c:22
(gdb) run
(gdb) until 25             # line 25 IS reached -> stops there (return total)
```

```
(gdb) tbreak demo.c:22
(gdb) run
(gdb) until 24             # line 24 is never executed (un-taken branch)
                           # -> sum_array returns; you land back in main (line 32)
```
