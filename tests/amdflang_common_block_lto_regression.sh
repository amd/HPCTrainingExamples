#!/bin/bash

# LCOMPILER-142: under -flto, amdflang truncates the ELF st_size / .lbss size of
# a COMMON block larger than 2**32 bytes to size mod 2**32, while address codegen
# still uses the full 64-bit offset -> a later store segfaults with no
# diagnostic. The repro declares a 6,000,000,000-byte (0x1_65A0BC00) COMMON
# block `big` (symbol big_); this test asserts readelf reports the full size,
# not the low-32-bit truncation (0x65A0BC00). Static check; no GPU required.

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   [ -z "$FC" ] && export FC=`which ftn`
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
   module load amdflang-new >& /dev/null
   if [ "$?" == "1" ]; then
      module load amdclang
   fi
fi
if [ -z "$FC" ]; then
   for c in amdflang-new amdflang flang-new flang; do
      command -v $c >/dev/null 2>&1 && { FC=$c; break; }
   done
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
SRC_DIR=${REPO_DIR}/Fortran/LTO/LCOMPILER-142
BUILD_DIR=$(mktemp -d)
trap 'rm -rf ${BUILD_DIR}' EXIT
cp ${SRC_DIR}/* ${BUILD_DIR}/
cd ${BUILD_DIR}

${FC} -O3 -mcmodel=medium -flto -o repro repro.F || { echo "LCOMPILER142_FAIL: build failed"; exit 0; }

# st_size of the COMMON-block symbol big_ (hex 0x... or decimal both handled)
SZ=$(readelf -sW repro | awk '$NF=="big_"{print $3; exit}')
VAL=$(printf "%d" "${SZ}" 2>/dev/null)
EXPECTED=6000000000
echo "big_ st_size = ${SZ} (${VAL}), expected ${EXPECTED}"
if [ "${VAL}" = "${EXPECTED}" ]; then
   echo "LCOMPILER142_PASS"
else
   echo "LCOMPILER142_FAIL: COMMON-block size truncated under -flto"
fi
