# ROCTx Markers in Fortran with HIPFort

This example demonstrates how to use roctx markers in Fortran applications. This allows you to annotate regions of your code for profiling with tools like `rocprofv3` and `rocprof-sys`.

The example provides **two versions**:
1. **hipfort version** (`roctx_demo_hipfort.F90`) - Uses the `hipfort_roctx` module from the HIPFort library
2. **standalone version** (`roctx_demo_standalone.F90`) - Self-contained with custom C bindings

## ROCTx API Overview

ROCTx provides functions for code annotation:
- `roctxRangePush(message)` - Start a named range (returns nesting level)
- `roctxRangePop()` - End the current range
- `roctxMark(message)` - Mark a single event

Ranges can also be nested. The ROCm profiling tools like `rocprofv3` can collect statistics of such ranges and visualize them in timeline traces.

## Using HIPFort (Recommended)

The simplest approach is to use the `hipfort_roctx` module provided by HIPFort directly:

```fortran
use hipfort_roctx
use iso_c_binding

integer :: ret

! Annotate initialization phase
ret = roctxRangePush("initialization"//c_null_char)
! ... initialization code ...
ret = roctxRangePop()

! Annotate computation with nested ranges
ret = roctxRangePush("computation"//c_null_char)
call roctxMark("starting kernel"//c_null_char)
! ... computation code ...
ret = roctxRangePop()
```

Note: Strings must be null-terminated with `//c_null_char` when using the hipfort bindings directly.

## Standalone Version (Without HIPFort)

If hipfort is not available, you can create your own C interface bindings. The standalone version includes a `roctx_mod` module that demonstrates this approach. The key is providing proper `iso_c_binding` interfaces:

```fortran
interface
  function roctxRangePushA(message) bind(C, name="roctxRangePushA") result(ret)
    import :: c_int, c_char
    character(kind=c_char), intent(in) :: message(*)
    integer(c_int) :: ret
  end function roctxRangePushA

  function roctxRangePop() bind(C, name="roctxRangePop") result(ret)
    import :: c_int
    integer(c_int) :: ret
  end function roctxRangePop
end interface
```

The standalone module also provides wrapper functions that handle null-termination automatically.

## Building

Load the required modules and build:

```bash
module load rocm
make
```

This creates two executables:
- `roctx_hipfort` - Uses hipfort's `roctx` bindings
- `roctx_standalone` - Self-contained with custom C bindings

## Running

Run the executables directly:

```bash
./roctx_hipfort
./roctx_standalone
```

## Profiling with rocprof

To see the roctx markers in a `rocprofv3` profile you can add the `--marker-trace` option:

```bash
rocprofv3 --marker-trace --output-format pftrace -- ./roctx_hipfort
```

This generates a Perfetto file `*.pftrace` that can be visualized with [Perfetto](https://ui.perfetto.dev/).
For more details on profiling with `rocprofv3` and roctx markers, see the [Rocprofv3 HIP exercises](https://github.com/amd/HPCTrainingExamples/blob/main/Rocprofv3/HIP/README.md).
