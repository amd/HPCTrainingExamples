#!/bin/bash

# mdb install check: module resolves, 'mdb version' runs, subcommands listed.
# mdb drives gdb/lldb/rocgdb; the install check itself needs no GPU.

module -t list 2>&1 | grep -q "^rocm" || module load rocm

if ! module load mdb 2>/dev/null; then
   echo "Unable to locate a modulefile for 'mdb'"
   exit 0
fi

command -v mdb >/dev/null || { echo "FAIL: mdb not found"; exit 1; }
mdb version >/dev/null 2>&1 || { echo "FAIL: 'mdb version' failed"; exit 1; }
mdb --help 2>/dev/null | grep -Eq "attach|launch|version" || { echo "FAIL: mdb subcommands missing"; exit 1; }

echo "MDB Install Check: SUCCESS"
