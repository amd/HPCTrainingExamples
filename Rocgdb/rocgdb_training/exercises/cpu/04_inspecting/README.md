- [Exercise 4 (inspecting): stack, variables, memory - and a wrong line](#orga09dd1b)
  - [4a - a backtrace without a crash (demo)](#org2d106e8)
  - [4b - move around the stack: frame / up / down](#orgb455f7f)
  - [4c - print and examine memory](#org628e204)
  - [4d - print with a SIDE EFFECT (careful!)](#orgb8e4370)
  - [4e - x remembers the format: a size gotcha (floats)](#org149dfe9)
  - [4f - a breakpoint on the wrong line (fusion, built at -O2)](#orgd090ce0)
  - [4g - fix it: rebuild without optimization](#org95839cf)


<a id="orga09dd1b"></a>

# Exercise 4 (inspecting): stack, variables, memory - and a wrong line

Programs are in `../common/`; run rocgdb from this directory. This exercise uses three programs (`demo`, `floats`, `fusion`); quit and restart rocgdb when you switch, as marked.


<a id="org2d106e8"></a>

## 4a - a backtrace without a crash (demo)

```
$ rocgdb ../common/demo
(gdb) break add            # demo.c:12, deep in the call chain
(gdb) run
(gdb) bt                   # #0 add  <-  #1 sum_array  <-  #2 main
```


<a id="orgb455f7f"></a>

## 4b - move around the stack: frame / up / down

-   Same stop as 4a (do not restart):

```
(gdb) frame 1              # select sum_array's frame
(gdb) print i              # its loop counter
(gdb) print total          # its running sum
(gdb) up                   # move to main
(gdb) print data           # main's array: {3, 1, 4, 1, 5, 9, 2, 6}
(gdb) down                 # back towards add
```


<a id="org628e204"></a>

## 4c - print and examine memory

-   Still the same stop:

```
(gdb) frame 2              # main's frame
(gdb) print data[3]        # one element
(gdb) x/8dw data           # dump 8 decimal words starting at data
```


<a id="orgb8e4370"></a>

## 4d - print with a SIDE EFFECT (careful!)

-   Same session, but a fresh breakpoint - clear the old one first:

```
(gdb) delete
(gdb) break demo.c:22 if i == 3
(gdb) run
(gdb) print total          # 8 so far (3 + 1 + 4)
(gdb) print i = 7          # skip to the last element - a real write into the program
(gdb) continue
```

-   The program now prints `sum = 14`, not `31`: your assignment changed the result.


<a id="org149dfe9"></a>

## 4e - x remembers the format: a size gotcha (floats)

-   Quit rocgdb (`quit`) and start it on `floats`:

```
$ rocgdb ../common/floats
(gdb) break floats.c:17
(gdb) run
(gdb) x/3fw fa             # 3 floats, word size (4) -> 1.5 2.5 3.5  (correct)
(gdb) x/3fg da             # 3 doubles, giant size (8) -> 1.5 2.5 3.5  (correct)
(gdb) x/3f  fa             # no size -> reuses "g" (8) on 4-byte floats -> GARBAGE
```

-   Lesson: `x` keeps the last size/format. Always name the size that matches the type.


<a id="orgd090ce0"></a>

## 4f - a breakpoint on the wrong line (fusion, built at -O2)

-   Quit rocgdb and start it on `fusion` (built at `-O2` by the setup step):

```
$ rocgdb ../common/fusion
(gdb) break fusion.c:12
(gdb) break fusion.c:13
(gdb) break fusion.c:14
(gdb) info breakpoints     # all three show the SAME address
```

-   Ask for line 13 and rocgdb binds you to a shared instruction, not the line you meant.


<a id="org95839cf"></a>

## 4g - fix it: rebuild without optimization

```
$ amdclang -g -O0 -o ../common/fusion ../common/fusion.c
$ rocgdb ../common/fusion
(gdb) break fusion.c:12
(gdb) break fusion.c:13
(gdb) break fusion.c:14
(gdb) info breakpoints     # now three DISTINCT addresses, one per line
```

-   Moral: when a breakpoint lands somewhere surprising, suspect optimization first.
-   Note: any optimization (`-O1` and up) fuses these particular lines here: only `-O0` makes every line separately addressable.
-   Restore the shared `-O2` build afterwards with `make -C ../common fusion`.
