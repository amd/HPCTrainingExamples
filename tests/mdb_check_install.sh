#!/bin/bash

# mdb install check: module resolves, 'mdb version' runs, subcommands listed.
# mdb drives gdb/lldb/rocgdb; the install check itself needs no GPU.

module -t list 2>&1 | grep -q "^rocm" || module load rocm

if ! module load mdb 2>/tmp/mdb_check.$$.err; then
   cat /tmp/mdb_check.$$.err
   rm -f /tmp/mdb_check.$$.err
   echo "Unable to locate a modulefile for 'mdb'"
   exit 0
fi
rm -f /tmp/mdb_check.$$.err

echo "=== mdb install check ==="
MDB_BIN=$(command -v mdb) || { echo "FAIL: mdb not found"; exit 1; }
echo "mdb: ${MDB_BIN}"

echo "+ mdb version"
mdb version || { echo "FAIL: 'mdb version' failed"; exit 1; }

echo "+ mdb --help"
MDB_HELP=$(mdb --help 2>&1)
echo "${MDB_HELP}"
echo "${MDB_HELP}" | grep -Eq "attach|launch|version" || { echo "FAIL: mdb subcommands missing"; exit 1; }

echo "MDB Install Check: SUCCESS"
