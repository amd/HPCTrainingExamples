- [Exercise 2 (stepping): next, step, and until](#org1ee2af2)
  - [2a - next OVER a call](#org9c64c6b)
  - [2b - step INTO a call](#org8677b43)
  - [2c - step and a LIBRARY function](#org300db2d)
  - [2d - until: run out of a loop](#org1547c8c)
  - [2e - until <line>: reached vs. run out of frame](#orge507148)


<a id="org1ee2af2"></a>

# Exercise 2 (stepping): next, step, and until

Programs are in `../common/`; run rocgdb from this directory. Key lines in `demo.c`: the call `sum_array(data, 8)` is line 32; the loop body `total = add(...)` is line 18; `return total` is line 25; the never-taken `printf("overflow!")` is line 24.

-   Parts 2a-2c reuse ONE rocgdb session: do not quit rocgdb between them.


<a id="org9c64c6b"></a>

## 2a - next OVER a call

```
$ rocgdb ../common/demo
(gdb) break demo.c:32      # the sum_array(...) call
(gdb) run
(gdb) next                 # runs the WHOLE call, stops on the next line of main
```

-   For the curious: `nexti` does the same at the machine-instruction level.


<a id="org8677b43"></a>

## 2b - step INTO a call

-   Stay in the SAME session: the breakpoint at line 32 is still set:

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


<a id="org300db2d"></a>

## 2c - step and a LIBRARY function

-   Same session again. Line 27 calls `strlen` (libc); add a breakpoint and step:

```
(gdb) break demo.c:31
(gdb) run                  # stops at line 31 (the strlen call)
(gdb) step                 # descends INTO libc's strlen (e.g. __strlen_evex, no source)
(gdb) finish               # returns you to your own code (main, line 31)
```

-   What `step` does at a library call depends on the build: with line info (as here) it drops into the library's implementation, often hand-written assembly shown with a "No such file" warning; without line info it just steps over. Either way, `finish` (run until the current function returns), `until`, or a breakpoint back in your own code gets you out.


<a id="org1547c8c"></a>

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


<a id="orge507148"></a>

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
