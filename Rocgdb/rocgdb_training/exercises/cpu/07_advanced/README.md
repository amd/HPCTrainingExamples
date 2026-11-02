# Exercise 7 (advanced): conditions, ignore, watch, display

Programs are in `../common/`; run rocgdb from this directory. All parts share one session; `delete` clears breakpoints between them.


## 7a - a conditional breakpoint

```
$ rocgdb ../common/demo
(gdb) break demo.c:22 if i == 5    # only stop on the 6th iteration
(gdb) run
(gdb) print i                      # 5
(gdb) print total                  # 14 (3 + 1 + 4 + 1 + 5)
```


## 7b - ignore a breakpoint N times

```
(gdb) delete
(gdb) break demo.c:22              # note its number N
(gdb) ignore N 5                   # skip the next 5 hits
(gdb) run                          # stops on the 6th hit, i = 5
(gdb) print i
```


## 7c - watch a value change

```
(gdb) delete
(gdb) break sum_array
(gdb) run
(gdb) watch total                  # stop whenever total changes
(gdb) continue                     # 0 -> 3
(gdb) continue                     # 3 -> 4 ...
```


## 7d - awatch an array element (not the 0th)

In this exercise, the use of `-location` is necessary to avoid rocgdb stopping each time the base address of `data` is accessed.

```
(gdb) delete
(gdb) break demo.c:20              # sum_array entry (data is in scope)
(gdb) run
(gdb) awatch -location data[3]     # stop on any READ or WRITE of element 3
(gdb) continue                     # fires when the loop reaches i = 3 (data[3] = 1)
```


## 7e - display a value while stepping

```
(gdb) delete
(gdb) tbreak demo.c:22
(gdb) run
(gdb) display total                # re-print total at every stop
(gdb) next
(gdb) next                         # total is shown automatically each time
```

-   `display` is ideal for a value that changes every iteration - set it once and just step.
