#!/bin/bash

# Regression check for the rocBLAS complex packed L2 (chpmv/ctpmv/ztpsv)
# unified-memory corruption bug on MI300A (gfx942, HSA_XNACK=1): handed plain
# system-allocator memory (no hipMalloc), these routines intermittently return
# ~1e34+ garbage, while the same calls on hipMalloc/hipMemcpy device buffers are
# always correct. The bug is stateful (needs a mix of packed routines at small
# sizes) and intermittent, so we launch the reproducer many times and FAIL if
# ANY launch returns wrong numbers; a device-buffer build (-DUSE_DEVICE) is run
# once as a control to tell the bug apart from a broken rocBLAS/environment.
# The reproducer C++ (from ~jopotyka/.../reproducer_for_frequent_testing) is
# embedded verbatim so this check is self-contained.
#
# Env: ROCBLAS_PACKED_USM_LAUNCHES (default 100), ROCBLAS_PACKED_USM_CONTROL (1).
# Verdict (CTest): "... PASSED"/"... FAILED"/"... SKIPPED" -> exit 0/1/77.

set -u

# --- rocm environment -------------------------------------------------------
if ! type module >/dev/null 2>&1; then
   [ -r /etc/profile.d/lmod.sh ]         && . /etc/profile.d/lmod.sh
   [ -r /usr/share/lmod/lmod/init/bash ] && . /usr/share/lmod/lmod/init/bash
fi
if type module >/dev/null 2>&1; then
   if module -t list 2>&1 | grep -q "^rocm"; then
      echo "rocm module already loaded, not reloading:" \
           "$(module -t list 2>&1 | grep '^rocm' | tr '\n' ' ')"
   else
      echo "rocm module is not loaded"
      echo "loading default rocm module"
      module load rocm
   fi
fi

# --- prerequisites: hipcc + rocBLAS (SKIP, not FAIL, if the toolchain is
#     simply not installed on this node) ------------------------------------
HIPCC=${HIPCC:-hipcc}
if ! command -v "${HIPCC}" >/dev/null 2>&1; then
   echo "ROCBLAS PACKED USM REGRESSION CHECK: SKIPPED (hipcc not on PATH; load a rocm module)"
   exit 77
fi
# ROCM_PATH: prefer the env the module exported, else derive <prefix> from
# the resolved hipcc path (<prefix>/bin/hipcc -> <prefix>).
if [ -z "${ROCM_PATH:-}" ]; then
   ROCM_PATH=$(dirname "$(dirname "$(command -v "${HIPCC}")")")
fi

# --- GPU / arch detection (SKIP if there is no GPU to run on) ---------------
GFX_ARCH=unknown; NGPU=0
if command -v rocminfo >/dev/null 2>&1; then
   RINFO=$(rocminfo 2>/dev/null)
   GFX_ARCH=$(echo "${RINFO}" | grep -m1 -E '^[[:space:]]*Name:[[:space:]]*gfx' \
      | sed -e 's/.*Name:[[:space:]]*//' -e 's/:.*//' -e 's/[[:space:]]*$//')
   NGPU=$(echo "${RINFO}" | grep -cE 'Name:[[:space:]]+gfx')
fi
GFX_ARCH=${GFX_ARCH:-unknown}
[ "${NGPU}" -ge 1 ] 2>/dev/null || NGPU=0

# --- self-diagnostic header -------------------------------------------------
LAUNCHES=${ROCBLAS_PACKED_USM_LAUNCHES:-100}
CONTROL=${ROCBLAS_PACKED_USM_CONTROL:-1}
echo
echo "--- rocBLAS packed-USM regression-check environment ---"
echo "  HIPCC        = ${HIPCC} ($(command -v "${HIPCC}"))"
echo "  ROCM_PATH    = ${ROCM_PATH}"
echo "  detected GPU arch = ${GFX_ARCH}"
echo "  visible GPUs      = ${NGPU}"
echo "  HSA_XNACK    = 1 (forced: unified memory is the bug precondition)"
echo "  LAUNCHES     = ${LAUNCHES}   CONTROL(device path) = ${CONTROL}"
if type module >/dev/null 2>&1; then
   echo "  loaded modules =" $(module -t list 2>&1 | tr '\n' ' ')
fi
echo "-------------------------------------------------------"
echo

if [ "${NGPU}" -lt 1 ]; then
   echo "ROCBLAS PACKED USM REGRESSION CHECK: SKIPPED (no GPU detected via rocminfo)"
   exit 77
fi

# --- scratch dir + embedded reproducer source -------------------------------
WORKDIR=$(mktemp -d -t rocblas_packed_usm_check_XXXXXXXXXX)
trap 'rm -rf "${WORKDIR}"' EXIT
SRC="${WORKDIR}/rocblas_packed_usm_bug.cpp"

cat > "${SRC}" <<'EOF'
// Self-contained C++ reproducer: rocBLAS complex packed Level-2 routines return
// garbage when called on UNIFIED / system-allocator memory (HSA_XNACK=1).
//
// Arrays are ordinary host allocations (std::vector -- system memory) handed
// straight to rocBLAS; no hipMalloc / hipMemcpy. On an APU with HSA_XNACK=1 the
// GPU page-migrates them on demand. Each result is checked against an independent
// in-file reference; every rocBLAS call is followed by hipDeviceSynchronize().
//
// Observed on MI300A (gfx942): the complex packed routines CHPMV / CTPMV / ZTPSV
// intermittently return ~1e34..1e37 garbage; the same calls staged through
// explicit hipMalloc/hipMemcpy device buffers are always correct. A single
// routine on its own does NOT reproduce -- the bug needs a *mix* of packed
// routines (a buffer freed by one reused uninitialised by the next) at small
// sizes (~500-700).
//
// Build/run (unified memory -- bug condition):
//   hipcc -O2 rocblas_packed_usm_bug.cpp -lrocblas -o repro_cpp
//   HSA_XNACK=1 ./repro_cpp            # -> RESULT: FAIL (often)
//   HSA_XNACK=0 ./repro_cpp            # -> unknown (XNACK-specificity test)
//
// Build/run (explicit device buffers -- control, should always pass):
//   hipcc -O2 -DUSE_DEVICE rocblas_packed_usm_bug.cpp -lrocblas -o repro_cpp_device
//   HSA_XNACK=1 ./repro_cpp_device     # -> RESULT: PASS

#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>

#include <complex>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using i8 = int64_t;
using cf = std::complex<float>;
using cd = std::complex<double>;

// rocBLAS enums (rocblas-types.h), matching the Fortran reproducer.
static const rocblas_fill      FILL_LOWER   = rocblas_fill_lower;        // 122
static const rocblas_operation OP_NONE      = rocblas_operation_none;    // 111
static const rocblas_diagonal  DIAG_NONUNIT = rocblas_diagonal_non_unit; // 131

#define CHECK_ROCBLAS(expr, msg)                                              \
   do {                                                                       \
      rocblas_status s_ = (expr);                                             \
      if (s_ != rocblas_status_success) {                                     \
         std::fprintf(stderr, "%s failed: %d\n", (msg), (int)s_);            \
         std::exit(2);                                                        \
      }                                                                       \
   } while (0)

#define CHECK_HIP(expr, msg)                                                  \
   do {                                                                       \
      hipError_t h_ = (expr);                                                 \
      if (h_ != hipSuccess) {                                                 \
         std::fprintf(stderr, "%s failed: %s\n", (msg), hipGetErrorString(h_));\
         std::exit(2);                                                        \
      }                                                                       \
   } while (0)

// lower-packed 'L' index of A(i,j), i>=j  (0-based i,j).
static inline i8 pidx(i8 i, i8 j, i8 n) {
   return j * n - j * (j - 1) / 2 + (i - j);
}

// fill value keyed on the *1-based* index k, matching the Fortran rv().
static inline double rv(i8 k) { return std::sin((double)k * 0.5) * 0.5; }

static rocblas_handle handle;
static int nfail = 0;

static void chk(const char* name, const char* prec, i8 n, double e, double tol) {
   const char* st = (e > tol) ? "FAIL" : "PASS";
   if (e > tol) ++nfail;
   std::printf("%8s  %2s   %6lld  %12.3e   %4s\n",
               name, prec, (long long)n, e, st);
}

// y := alpha*A*x + beta*y, A hermitian packed 'L'
static double run_chpmv(i8 n) {
   i8 np = n * (n + 1) / 2;
   std::vector<cf> ap(np), x(n), y(n), yref(n);
   for (i8 i = 1; i <= np; ++i) ap[i - 1] = cf((float)rv(i), (float)rv(i + 7));
   for (i8 j = 1; j <= n; ++j) {
      cf d = ap[pidx(j - 1, j - 1, n)];
      ap[pidx(j - 1, j - 1, n)] = cf(d.real(), 0.0f);
   }
   for (i8 i = 1; i <= n; ++i) {
      x[i - 1] = cf((float)rv(i + 1), (float)rv(i + 3));
      y[i - 1] = cf((float)rv(i + 2), (float)rv(i + 5));
   }
   cf alpha(1.1f, 0.3f), beta(0.5f, -0.2f);
   cd alphad(alpha), betad(beta);
   for (i8 i = 1; i <= n; ++i) {
      cd s(0.0, 0.0);
      for (i8 j = 1; j <= n; ++j) {
         cf aijf = (i >= j) ? ap[pidx(i - 1, j - 1, n)]
                            : std::conj(ap[pidx(j - 1, i - 1, n)]);
         s += cd(aijf) * cd(x[j - 1]);
      }
      yref[i - 1] = cf(alphad * s + betad * cd(y[i - 1]));
   }
#ifdef USE_DEVICE
   void *d_ap, *d_x, *d_y;
   CHECK_HIP(hipMalloc(&d_ap, np * sizeof(cf)), "hipMalloc ap");
   CHECK_HIP(hipMalloc(&d_x,  n  * sizeof(cf)), "hipMalloc x");
   CHECK_HIP(hipMalloc(&d_y,  n  * sizeof(cf)), "hipMalloc y");
   CHECK_HIP(hipMemcpy(d_ap, ap.data(), np * sizeof(cf), hipMemcpyDefault), "H2D ap");
   CHECK_HIP(hipMemcpy(d_x,  x.data(),  n * sizeof(cf), hipMemcpyDefault), "H2D x");
   CHECK_HIP(hipMemcpy(d_y,  y.data(),  n * sizeof(cf), hipMemcpyDefault), "H2D y");
   CHECK_ROCBLAS(rocblas_chpmv(handle, FILL_LOWER, (rocblas_int)n,
                  reinterpret_cast<rocblas_float_complex*>(&alpha),
                  reinterpret_cast<rocblas_float_complex*>(d_ap),
                  reinterpret_cast<rocblas_float_complex*>(d_x), 1,
                  reinterpret_cast<rocblas_float_complex*>(&beta),
                  reinterpret_cast<rocblas_float_complex*>(d_y), 1),
                 "chpmv");
   CHECK_HIP(hipDeviceSynchronize(), "sync");
   CHECK_HIP(hipMemcpy(y.data(), d_y, n * sizeof(cf), hipMemcpyDefault), "D2H y");
   hipFree(d_ap); hipFree(d_x); hipFree(d_y);
#else
   CHECK_ROCBLAS(rocblas_chpmv(handle, FILL_LOWER, (rocblas_int)n,
                  reinterpret_cast<rocblas_float_complex*>(&alpha),
                  reinterpret_cast<rocblas_float_complex*>(ap.data()),
                  reinterpret_cast<rocblas_float_complex*>(x.data()), 1,
                  reinterpret_cast<rocblas_float_complex*>(&beta),
                  reinterpret_cast<rocblas_float_complex*>(y.data()), 1),
                 "chpmv");
   CHECK_HIP(hipDeviceSynchronize(), "sync");
#endif
   double den = 0.0, num = 0.0;
   for (i8 i = 0; i < n; ++i) {
      den = std::max(den, (double)std::abs(yref[i]));
      num = std::max(num, (double)std::abs(y[i] - yref[i]));
   }
   if (den == 0.0) den = 1.0;
   return num / den;
}

// x := A*x, A lower-triangular packed 'L'
static double run_ctpmv(i8 n) {
   i8 np = n * (n + 1) / 2;
   std::vector<cf> ap(np), x(n), x0(n), xref(n);
   for (i8 i = 1; i <= np; ++i) ap[i - 1] = cf((float)rv(i), (float)rv(i + 7));
   for (i8 i = 1; i <= n; ++i) x0[i - 1] = cf((float)rv(i + 1), (float)rv(i + 3));
   for (i8 i = 1; i <= n; ++i) {
      cd s(0.0, 0.0);
      for (i8 j = 1; j <= i; ++j) s += cd(ap[pidx(i - 1, j - 1, n)]) * cd(x0[j - 1]);
      xref[i - 1] = cf(s);
   }
   x = x0;
#ifdef USE_DEVICE
   void *d_ap, *d_x;
   CHECK_HIP(hipMalloc(&d_ap, np * sizeof(cf)), "hipMalloc ap");
   CHECK_HIP(hipMalloc(&d_x,  n  * sizeof(cf)), "hipMalloc x");
   CHECK_HIP(hipMemcpy(d_ap, ap.data(), np * sizeof(cf), hipMemcpyDefault), "H2D ap");
   CHECK_HIP(hipMemcpy(d_x,  x.data(),  n * sizeof(cf), hipMemcpyDefault), "H2D x");
   CHECK_ROCBLAS(rocblas_ctpmv(handle, FILL_LOWER, OP_NONE, DIAG_NONUNIT,
                  (rocblas_int)n,
                  reinterpret_cast<rocblas_float_complex*>(d_ap),
                  reinterpret_cast<rocblas_float_complex*>(d_x), 1),
                 "ctpmv");
   CHECK_HIP(hipDeviceSynchronize(), "sync");
   CHECK_HIP(hipMemcpy(x.data(), d_x, n * sizeof(cf), hipMemcpyDefault), "D2H x");
   hipFree(d_ap); hipFree(d_x);
#else
   CHECK_ROCBLAS(rocblas_ctpmv(handle, FILL_LOWER, OP_NONE, DIAG_NONUNIT,
                  (rocblas_int)n,
                  reinterpret_cast<rocblas_float_complex*>(ap.data()),
                  reinterpret_cast<rocblas_float_complex*>(x.data()), 1),
                 "ctpmv");
   CHECK_HIP(hipDeviceSynchronize(), "sync");
#endif
   double den = 0.0, num = 0.0;
   for (i8 i = 0; i < n; ++i) {
      den = std::max(den, (double)std::abs(xref[i]));
      num = std::max(num, (double)std::abs(x[i] - xref[i]));
   }
   if (den == 0.0) den = 1.0;
   return num / den;
}

// solve A*x = b, A lower-triangular packed 'L', diagonally dominant
static double run_ztpsv(i8 n) {
   i8 np = n * (n + 1) / 2;
   std::vector<cd> ap(np), b(n), x(n), xref(n);
   for (i8 i = 1; i <= np; ++i) ap[i - 1] = cd(rv(i), rv(i + 7));
   for (i8 j = 1; j <= n; ++j) ap[pidx(j - 1, j - 1, n)] = cd((double)n, 0.0);
   for (i8 i = 1; i <= n; ++i) b[i - 1] = cd(rv(i + 1), rv(i + 3));
   for (i8 i = 1; i <= n; ++i) {
      cd s = b[i - 1];
      for (i8 j = 1; j <= i - 1; ++j) s -= ap[pidx(i - 1, j - 1, n)] * xref[j - 1];
      xref[i - 1] = s / ap[pidx(i - 1, i - 1, n)];
   }
   x = b;
#ifdef USE_DEVICE
   void *d_ap, *d_x;
   CHECK_HIP(hipMalloc(&d_ap, np * sizeof(cd)), "hipMalloc ap");
   CHECK_HIP(hipMalloc(&d_x,  n  * sizeof(cd)), "hipMalloc x");
   CHECK_HIP(hipMemcpy(d_ap, ap.data(), np * sizeof(cd), hipMemcpyDefault), "H2D ap");
   CHECK_HIP(hipMemcpy(d_x,  x.data(),  n * sizeof(cd), hipMemcpyDefault), "H2D x");
   CHECK_ROCBLAS(rocblas_ztpsv(handle, FILL_LOWER, OP_NONE, DIAG_NONUNIT,
                  (rocblas_int)n,
                  reinterpret_cast<rocblas_double_complex*>(d_ap),
                  reinterpret_cast<rocblas_double_complex*>(d_x), 1),
                 "ztpsv");
   CHECK_HIP(hipDeviceSynchronize(), "sync");
   CHECK_HIP(hipMemcpy(x.data(), d_x, n * sizeof(cd), hipMemcpyDefault), "D2H x");
   hipFree(d_ap); hipFree(d_x);
#else
   CHECK_ROCBLAS(rocblas_ztpsv(handle, FILL_LOWER, OP_NONE, DIAG_NONUNIT,
                  (rocblas_int)n,
                  reinterpret_cast<rocblas_double_complex*>(ap.data()),
                  reinterpret_cast<rocblas_double_complex*>(x.data()), 1),
                 "ztpsv");
   CHECK_HIP(hipDeviceSynchronize(), "sync");
#endif
   double den = 0.0, num = 0.0;
   for (i8 i = 0; i < n; ++i) {
      den = std::max(den, std::abs(xref[i]));
      num = std::max(num, std::abs(x[i] - xref[i]));
   }
   if (den == 0.0) den = 1.0;
   return num / den;
}

int main() {
   CHECK_ROCBLAS(rocblas_create_handle(&handle), "rocblas_create_handle");
   CHECK_ROCBLAS(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host),
                 "set_pointer_mode");

#ifdef USE_DEVICE
   std::printf("memory_mode: explicit hipMalloc/hipMemcpy (USE_DEVICE)\n");
#else
   std::printf("memory_mode: unified/system (no hipMalloc)\n");
#endif

   const i8 sizes[3] = {512, 600, 700};
   const int repeats = 5;

   std::printf("routine  prec   n      rel.err     status\n");
   for (int it = 0; it < repeats; ++it) {
      for (int is = 0; is < 3; ++is) {
         i8 n = sizes[is];
         chk("chpmv", "c", n, run_chpmv(n), 1.0e-3);
         chk("ctpmv", "c", n, run_ctpmv(n), 1.0e-3);
         chk("ztpsv", "z", n, run_ztpsv(n), 1.0e-9);
      }
   }

   std::printf("failures: %d\n", nfail);
   rocblas_destroy_handle(handle);
   if (nfail > 0) {
#ifdef USE_DEVICE
      std::printf("RESULT: FAIL (unexpected -- device buffer path should not fail)\n");
#else
      std::printf("RESULT: FAIL (rocBLAS packed L2 wrong on unified memory)\n");
#endif
      return 1;
   }
   std::printf("RESULT: PASS\n");
   return 0;
}
EOF

# --- build helper: distinguishes "rocBLAS not installed" (-> SKIP) from a
#     real compiler failure (-> FAIL). Extra args ($2..) go to hipcc. --------
build() {
   local out="$1"; shift
   "${HIPCC}" -O2 "${SRC}" "$@" \
      -I"${ROCM_PATH}/include" \
      -L"${ROCM_PATH}/lib" -lrocblas \
      -Wl,-rpath,"${ROCM_PATH}/lib" \
      -o "${out}" > "${out}.build.log" 2>&1
}

echo "=== Building unified-memory reproducer ==="
if ! build "${WORKDIR}/repro_usm"; then
   echo "hipcc build failed; build log:"
   sed 's/^/  /' "${WORKDIR}/repro_usm.build.log"
   if grep -qiE "rocblas/rocblas.h|cannot find -lrocblas|-lrocblas" "${WORKDIR}/repro_usm.build.log"; then
      echo "ROCBLAS PACKED USM REGRESSION CHECK: SKIPPED (rocBLAS headers/library not found)"
      exit 77
   fi
   echo "ROCBLAS PACKED USM REGRESSION CHECK: FAILED (build error)"
   exit 1
fi
echo "  built ${WORKDIR}/repro_usm"

# --- control (device-buffer) path: build + one run -------------------------
CTRL_STATE="not-run"
if [ "${CONTROL}" != "0" ]; then
   echo "=== Building device-buffer control (-DUSE_DEVICE) ==="
   if build "${WORKDIR}/repro_device" -DUSE_DEVICE; then
      echo "  built ${WORKDIR}/repro_device"
      echo "=== Running device-buffer control once (expected PASS) ==="
      HSA_XNACK=1 "${WORKDIR}/repro_device" > "${WORKDIR}/control.log" 2>&1
      case $? in
         0) CTRL_STATE="PASS";;
         1) CTRL_STATE="FAIL";;
         *) CTRL_STATE="APIERR";;
      esac
      grep -E "memory_mode:|RESULT:|failures:" "${WORKDIR}/control.log" | sed 's/^/  /'
   else
      echo "  control build failed (continuing; control is diagnostic only):"
      sed 's/^/  /' "${WORKDIR}/repro_device.build.log"
      CTRL_STATE="BUILD-FAIL"
   fi
   echo "  control (device path) verdict: ${CTRL_STATE}"
fi

# --- unified-memory launches: any wrong-numbers launch is a regression ------
echo
echo "=== Launching unified-memory reproducer x ${LAUNCHES} (HSA_XNACK=1) ==="
echo "  each launch: 3 sizes {512,600,700} x 5 repeats x {chpmv,ctpmv,ztpsv};"
echo "  tol chpmv/ctpmv<=1e-3, ztpsv<=1e-9; any call over tol => launch FAIL."
echo
pass=0; numfail=0; apierr=0; other=0; first_fail_log=""
for ((k = 1; k <= LAUNCHES; k++)); do
   HSA_XNACK=1 "${WORKDIR}/repro_usm" > "${WORKDIR}/run_${k}.log" 2>&1
   rc=$?
   case ${rc} in
      0) pass=$((pass + 1));    st="PASS";;
      1) numfail=$((numfail + 1)); st="FAIL(numeric)"
         [ -z "${first_fail_log}" ] && first_fail_log="${WORKDIR}/run_${k}.log";;
      2) apierr=$((apierr + 1)); st="APIERR";;
      *) other=$((other + 1));   st="rc=${rc}";;
   esac
   printf '  launch %3d/%3d : %s\n' "${k}" "${LAUNCHES}" "${st}"
   # Print launch 1's full table so a standalone run is informative on its own.
   [ "${k}" -eq 1 ] && sed 's/^/     /' "${WORKDIR}/run_1.log"
done

# Worst relative error per routine across EVERY launch -- the representative,
# at-a-glance health number. On a healthy/patched system these sit far below
# tolerance (chpmv/ctpmv ~1e-6, ztpsv ~1e-13); the bug drives them to ~1e34+.
read -r W_CHPMV W_CTPMV W_ZTPSV <<< "$(awk '
   /^[[:space:]]*chpmv[[:space:]]/{e=$4+0; if(e>c)c=e}
   /^[[:space:]]*ctpmv[[:space:]]/{e=$4+0; if(e>t)t=e}
   /^[[:space:]]*ztpsv[[:space:]]/{e=$4+0; if(e>z)z=e}
   END{printf "%.3e %.3e %.3e", c+0, t+0, z+0}' "${WORKDIR}"/run_*.log)"

echo
echo "=========================================================="
echo "  ROCBLAS PACKED USM REGRESSION CHECK summary"
echo "=========================================================="
echo "  arch=${GFX_ARCH}  launches=${LAUNCHES}"
echo "  launch verdicts : PASS=${pass}  FAIL(numeric)=${numfail}  APIERR=${apierr}  OTHER=${other}"
echo "  worst rel.err over all launches (representative pass criterion):"
echo "     chpmv  (tol 1e-3) = ${W_CHPMV}"
echo "     ctpmv  (tol 1e-3) = ${W_CTPMV}"
echo "     ztpsv  (tol 1e-9) = ${W_ZTPSV}"
echo "  device-path control = ${CTRL_STATE}"

# Evidence for the first numeric-failure launch, if any.
if [ -n "${first_fail_log}" ]; then
   echo "  --- first numeric-failure launch ---"
   grep -E "FAIL|RESULT:|failures:" "${first_fail_log}" | head -12 | sed 's/^/     /'
fi
echo "=========================================================="

# Precondition not met: every launch failed at the rocBLAS/HIP API level (e.g.
# this GPU cannot run rocBLAS on system-allocated / XNACK memory at all) -- that
# is an environment limitation, not the numeric corruption we test for.
if [ "${apierr}" -eq "${LAUNCHES}" ]; then
   echo "RESULT: verdict=SKIP (rocBLAS/HIP could not run on unified memory on ${GFX_ARCH})"
   echo "ROCBLAS PACKED USM REGRESSION CHECK: SKIPPED (unified-memory rocBLAS unusable here; not a regression signal)"
   exit 77
fi

if [ "${numfail}" -gt 0 ] || [ "${apierr}" -gt 0 ] || [ "${other}" -gt 0 ]; then
   echo "RESULT: verdict=FAIL arch=${GFX_ARCH} launches=${LAUNCHES} numeric_fail=${numfail} apierr=${apierr} other=${other} worst_rel_err[chpmv/ctpmv/ztpsv]=${W_CHPMV}/${W_CTPMV}/${W_ZTPSV} control=${CTRL_STATE}"
   if [ "${numfail}" -gt 0 ] && [ "${CTRL_STATE}" = "PASS" ]; then
      echo "  -> device-buffer control PASSED while the unified path FAILED: this is"
      echo "     the rocBLAS packed-L2 unified-memory corruption bug (regressed)."
   fi
   echo "ROCBLAS PACKED USM REGRESSION CHECK: FAILED (${numfail}/${LAUNCHES} launches returned wrong numbers; worst rel.err chpmv=${W_CHPMV} ctpmv=${W_CTPMV} ztpsv=${W_ZTPSV})"
   exit 1
fi

echo "RESULT: verdict=PASS arch=${GFX_ARCH} launches=${LAUNCHES} numeric_fail=0 worst_rel_err[chpmv/ctpmv/ztpsv]=${W_CHPMV}/${W_CTPMV}/${W_ZTPSV} control=${CTRL_STATE}"
echo "ROCBLAS PACKED USM REGRESSION CHECK: PASSED (${pass}/${LAUNCHES} launches correct, 0 numeric failures; worst rel.err chpmv=${W_CHPMV} ctpmv=${W_CTPMV} ztpsv=${W_ZTPSV})"
exit 0
