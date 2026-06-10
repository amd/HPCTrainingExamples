#!/bin/bash

# This test imports the IntelliKit packages in Python to verify that
# IntelliKit is installed and accessible via its module.
#
# IntelliKit is a monorepo of independent packages (accordo, kerncap,
# linex, metrix, nexus, rocm_mcp, uprof_mcp). Some packages compile C++
# at install time and may be unavailable on a given ROCm toolchain, so
# this test discovers which IntelliKit packages are actually installed
# under INTELLIKIT_HOME and imports exactly those (requiring at least
# one to be present and all present ones to import cleanly).
#
# NOTE: this test assumes IntelliKit has been installed according to the
# model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/tools/scripts/intellikit_setup.sh

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

module load intellikit

python3 - <<'PYEOF' 2>&1 && echo 'Success' || echo 'Failure'
import importlib
import os
import sys

candidates = ["accordo", "kerncap", "linex", "metrix", "nexus", "rocm_mcp", "uprof_mcp"]

home = os.environ.get("INTELLIKIT_HOME", "")
if not home or not os.path.isdir(home):
    print(f"INTELLIKIT_HOME not set or missing: {home!r}", file=sys.stderr)
    sys.exit(1)

# A package is "installed" if its top-level package dir or its dist-info
# metadata is present in INTELLIKIT_HOME (the pip --target install root).
def is_installed(name):
    if os.path.isdir(os.path.join(home, name)):
        return True
    for entry in os.listdir(home):
        if entry.startswith(name + "-") and entry.endswith(".dist-info"):
            return True
    return False

installed = [name for name in candidates if is_installed(name)]
if not installed:
    print(f"No IntelliKit packages found under {home}", file=sys.stderr)
    sys.exit(1)

failed = []
for name in installed:
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        failed.append(f"{name}: {exc}")

print("Found under INTELLIKIT_HOME:", ", ".join(installed))
if failed:
    print("Failed to import:", "; ".join(failed), file=sys.stderr)
    sys.exit(1)

print("Imported OK:", ", ".join(installed))
PYEOF
