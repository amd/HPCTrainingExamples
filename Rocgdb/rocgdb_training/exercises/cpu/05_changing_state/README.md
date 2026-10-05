- [Exercise 5 (changing state): write memory and skip a line](#org007bb35)
  - [5a - write a variable and an array element](#org0a89246)
  - [5b - jump over a line](#org4533207)


<a id="org007bb35"></a>

# Exercise 5 (changing state): write memory and skip a line

Programs are in `../common/`; run rocgdb from this directory. In `floats.c`: `da` is a `double[3]`, `scale` is set on line 11 and multiplied on line 14.


<a id="org0a89246"></a>

## 5a - write a variable and an array element

```
$ rocgdb ../common/floats
(gdb) break floats.c:17
(gdb) run
(gdb) print scale                  # 20 (already multiplied by 10)
(gdb) set var scale = 100          # overwrite a scalar
(gdb) print scale
(gdb) print da                     # {1.5, 2.5, 3.5} - before we change it
(gdb) set {double}&da[1] = 9.0      # typed write straight to an address
(gdb) print da                     # {1.5, 9, 3.5}
```

-   `set {double}<addr> = ...` writes `sizeof(double)` bytes at the address: powerful and unchecked, so aim carefully.


<a id="org4533207"></a>

## 5b - jump over a line

-   `jump` moves the program counter; use it to skip the multiply on line 14:

```
(gdb) delete
(gdb) break floats.c:14            # the "scale = scale * 10.0" line
(gdb) run
(gdb) jump 15                      # resume at line 15, skipping line 14
```

-   Output shows `after: scale = 2` (the multiply never ran). No `continue` is needed: `jump` resumes execution, and with no breakpoint ahead the program finishes on its own.
-   `jump` does *not* adjust the stack: it is for deliberate experiments, easy to misuse.
