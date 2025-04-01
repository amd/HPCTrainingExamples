--offload-arch=gfx942#!/bin/bash

git clone https://github.com/SOLLVE/sollve_vv.git

cd sollve_vv
module load amdclang
make CC=clang CXX=clang++ FC=flang OMP_VERSON=5.0 DEVICE_TYPE=amd LOG=1 make all report_html

AMDCLANG_PASSED=`grep "PASS. exit code: 0" results_report/results.csv |wc -l`
AMDCLANG_FAILED=`grep FAIL results_report/results.csv |wc -l`
AMDCLANG_TOTAL=`wc -l results_report/results.csv`

tar -czvf ../amdclang_results_report.tgz results_report

make tidy

module load aomp
make CC=clang CXX=clang++ FC=flang OMP_VERSON=5.0 DEVICE_TYPE=amd LOG=1 all report_html

AOMP_PASSED=`grep "PASS. exit code: 0" results_report/results.csv |wc -l`
AOMP_FAILED=`grep FAIL results_report/results.csv |wc -l`
AOMP_TOTAL=`wc -l results_report/results.csv`

echo "For amdclang, PASSED $AMDCLANG_PASSED FAILED $AMDCLANG_FAILED Total $AMDCLANG_TOTAL"
echo "For aomp, PASSED $AOMP_PASSED FAILED $AOMP_FAILED Total $AOMP_TOTAL"

tar -czvf ../aomp_results_report.tgz results_report

# copy back results_report.tgz and view either results.csv or results.html
