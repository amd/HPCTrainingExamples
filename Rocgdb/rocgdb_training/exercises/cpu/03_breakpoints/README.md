- [Exercise 3 (breakpoints): set, list, and toggle](#org0491417)
  - [3a - set breakpoints several ways, then list them](#org568500f)
  - [3b - a one-shot (temporary) breakpoint](#org9b84883)
  - [3c - disable a loop breakpoint once you have seen enough](#org91394dc)


<a id="org0491417"></a>

# Exercise 3 (breakpoints): set, list, and toggle

Programs are in `../common/`; run rocgdb from this directory.


<a id="org568500f"></a>

## 3a - set breakpoints several ways, then list them

```
$ rocgdb ../common/demo
(gdb) break sum_array        # by function
(gdb) break add              # another function
(gdb) break demo.c:22        # by line (the loop body)
(gdb) break demo.c:25        # by file:line (the return)
(gdb) info breakpoints       # see them all, with their numbers and hit counts
(gdb) delete                 # clear them
(gdb) info breakpoints       # confirm: "No breakpoints or watchpoints."
```


<a id="org9b84883"></a>

## 3b - a one-shot (temporary) breakpoint

```
(gdb) tbreak main
(gdb) run                    # stops at the first line of main...
(gdb) info breakpoints       # ...and the tbreak is already gone (fires once)
```


<a id="org91394dc"></a>

## 3c - disable a loop breakpoint once you have seen enough

-   A breakpoint in the loop fires on every iteration. Watch a couple, then switch it off and let the program finish - a realistic move once the loop is clear:

```
(gdb) break demo.c:22        # note the number rocgdb prints (call it N)
(gdb) run                    # stops, i = 0
(gdb) print i
(gdb) continue               # stops again, i = 1
(gdb) print i
(gdb) disable N              # N from the "break" line above (or "info breakpoints")
(gdb) continue               # now runs freely to the end
(gdb) info breakpoints       # N is still listed, just "Enb = n" (dormant, not deleted)
```

-   `enable N` turns it back on; `delete N` removes it for good.
