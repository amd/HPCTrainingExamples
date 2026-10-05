- [Exercise 7 (advanced): conditions, ignore, watch, display](#orgff13a99)
  - [7a - a conditional breakpoint](#orgac3e103)
  - [7b - ignore a breakpoint N times](#orgec9420a)
  - [7c - watch a value change](#orgcb89375)
  - [7d - awatch an array element (not the 0th)](#org4f8e68a)
  - [7e - display a value while stepping](#org5151c8a)


<a id="orgff13a99"></a>

# Exercise 7 (advanced): conditions, ignore, watch, display

Programs are in `../common/`; run rocgdb from this directory. All parts share one session; `delete` clears breakpoints between them.


<a id="orgac3e103"></a>

## 7a - a conditional breakpoint

```
$ rocgdb ../common/demo
(gdb) break demo.c:22 if i == 5    # only stop on the 6th iteration
(gdb) run
(gdb) print i                      # 5
(gdb) print total                  # 14 (3 + 1 + 4 + 1 + 5)
```


<a id="orgec9420a"></a>

## 7b - ignore a breakpoint N times

```
(gdb) delete
(gdb) break demo.c:22              # note its number N
(gdb) ignore N 5                   # skip the next 5 hits
(gdb) run                          # stops on the 6th hit, i = 5
(gdb) print i
```


<a id="orgcb89375"></a>

## 7c - watch a value change

```
(gdb) delete
(gdb) break sum_array
(gdb) run
(gdb) watch total                  # stop whenever total changes
(gdb) continue                     # 0 -> 3
(gdb) continue                     # 3 -> 4 ...
```


<a id="org4f8e68a"></a>

## 7d - awatch an array element (not the 0th)

```
(gdb) delete
(gdb) break demo.c:20              # sum_array entry (data is in scope)
(gdb) run
(gdb) awatch data[3]              # stop on any READ or WRITE of element 3
(gdb) continue                     # fires when the loop reaches i = 3 (data[3] = 1)
```


<a id="org5151c8a"></a>

## 7e - display a value while stepping

```
(gdb) delete
(gdb) tbreak demo.c:22
(gdb) run
(gdb) display total                # re-print total at every stop
(gdb) next
(gdb) next                         # total is shown automatically each time
```

-   `display` is ideal for a value that changes every iteration: set it once and just step.
