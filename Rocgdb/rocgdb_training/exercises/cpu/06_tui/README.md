- [Exercise 6 (TUI): redo a stepping session full-screen](#org2c98191)


<a id="org2c98191"></a>

# Exercise 6 (TUI): redo a stepping session full-screen

Programs are in `../common/`; run rocgdb from this directory. This repeats Exercise 2, but in the terminal UI so you can watch the source as you step.

```
$ rocgdb --tui ../common/demo   # or start rocgdb normally and press Ctrl-x a
(gdb) break sum_array
(gdb) run                    # the source pane highlights the current line
(gdb) next                   # watch the marker move; the code stays in view
(gdb) step                   # descend into add
(gdb) layout asm             # switch the top pane to disassembly...
(gdb) layout src             # ...and back to source
```

-   Keys: `Ctrl-x a` toggles the TUI on/off; `Ctrl-x o` moves focus between panes; the focused pane scrolls with the arrows / `PgUp` / `PgDn`; `Ctrl-L` redraws if program output scribbles over the screen.
-   The TUI is terminal-native and works over a plain SSH session: no X11 needed.
