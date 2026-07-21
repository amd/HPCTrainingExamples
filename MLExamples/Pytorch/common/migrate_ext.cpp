// Zero-copy host->device "migrate" for MI300A-class APUs (unified/coherent HBM).
//
// On a discrete GPU, tensor.to('cuda') must hipMemcpy the bytes across PCIe/xGMI
// into a separate device allocation. On the MI300A APU the CPU and GPU share the
// *same* physical HBM, so that copy is physically unnecessary: we can hand the
// GPU the identical pointer and let the coherent fabric (or on-demand page
// migration under HSA_XNACK=1) serve the accesses. This turns an O(bytes) copy
// into an O(1) metadata operation -- "copy the pointer, not the data".
//
// Three primitives are exposed:
//   managed_empty(shape, dtype)  -> a CPU torch.Tensor whose storage is
//        hipMallocManaged() memory (valid on both host and device). Fill it on
//        the host like any CPU tensor.
//   migrate(cpu_tensor, prefetch)-> a CUDA torch.Tensor that *aliases* the same
//        bytes (no copy). If prefetch=True, hint the driver to migrate the pages
//        to the current device with hipMemPrefetchAsync (still no host->device
//        DMA of user data on a coherent APU; it just updates page residency).
//        Requires the source to be managed memory (use managed_empty()).
//   register_migrate(cpu_tensor) -> like migrate(), but works on ANY existing,
//        ordinary (pageable) CPU tensor: it hipHostRegister()s the pages for
//        device access and returns the mapped device pointer via
//        hipHostGetDevicePointer(). No pre-allocation in managed memory needed,
//        and still no data copy. The pages are hipHostUnregister()ed when the
//        returned CUDA tensor is freed.
//
// Correctness note: the returned CUDA tensor shares storage with the source, so
// writes on one side are visible on the other. For an input-staging buffer that
// is exactly what we want (host fills, GPU reads).
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <c10/hip/HIPStream.h>
#include <memory>
#include <vector>

#define HIP_CHECK(cmd)                                                     \
  do {                                                                     \
    hipError_t _e = (cmd);                                                 \
    TORCH_CHECK(_e == hipSuccess, "HIP error: ", hipGetErrorString(_e));  \
  } while (0)

static size_t nbytes(const std::vector<int64_t>& shape, c10::ScalarType dt) {
  size_t n = c10::elementSize(dt);
  for (auto s : shape) n *= static_cast<size_t>(s);
  return n;
}

// Allocate a host-usable tensor backed by hipMallocManaged so the exact same
// virtual address is dereferenceable from device kernels.
at::Tensor managed_empty(std::vector<int64_t> shape, c10::ScalarType dtype) {
  size_t bytes = nbytes(shape, dtype);
  void* ptr = nullptr;
  HIP_CHECK(hipMallocManaged(&ptr, bytes ? bytes : 1));
  auto opts = at::TensorOptions().dtype(dtype).device(at::kCPU);
  return at::from_blob(
      ptr, shape,
      [](void* p) { hipFree(p); },
      opts);
}

// Alias a (managed/coherent) CPU tensor as a CUDA tensor with no data copy.
at::Tensor migrate(at::Tensor cpu, bool prefetch) {
  TORCH_CHECK(cpu.is_cpu(), "migrate() expects a CPU tensor");
  TORCH_CHECK(cpu.is_contiguous(), "migrate() expects a contiguous tensor");
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  void* ptr = cpu.data_ptr();
  auto sizes = cpu.sizes().vec();
  auto strides = cpu.strides().vec();

  if (prefetch) {
    // Hint the driver to make the pages device-resident. On a coherent APU this
    // updates page residency/attribution rather than DMAing user data over a bus.
    size_t bytes = nbytes(sizes, cpu.scalar_type());
    hipStream_t s = c10::hip::getCurrentHIPStream(device).stream();
    // Best-effort: ignore failure (e.g. pageable, non-managed memory).
    hipMemPrefetchAsync(ptr, bytes, device, s);
  }

  // Keep the source storage alive for as long as the CUDA view exists.
  auto holder = std::make_shared<at::Tensor>(cpu);
  auto opts = at::TensorOptions().dtype(cpu.scalar_type()).device(at::kCUDA, device);
  return at::from_blob(
      ptr, sizes, strides,
      [holder](void*) mutable { holder.reset(); },
      opts);
}

// Alias ANY (pageable) CPU tensor as a CUDA tensor with no data copy, by
// registering its host pages for device access. This is the general-purpose
// variant: it does not require the source to have been allocated in managed
// memory, so it can migrate tensors produced by arbitrary code (e.g. a
// DataLoader batch) in place.
at::Tensor register_migrate(at::Tensor cpu) {
  TORCH_CHECK(cpu.is_cpu(), "register_migrate() expects a CPU tensor");
  TORCH_CHECK(cpu.is_contiguous(), "register_migrate() expects a contiguous tensor");
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  void* host = cpu.data_ptr();
  size_t bytes = static_cast<size_t>(cpu.numel()) * cpu.element_size();

  // Register the pageable host pages so the GPU can access them directly. If the
  // range is already registered (e.g. wrapped twice), reuse it and do not
  // unregister on free (the original owner will).
  hipError_t reg = hipHostRegister(host, bytes ? bytes : 1, hipHostRegisterMapped);
  bool we_registered = (reg == hipSuccess);
  if (!we_registered && reg != hipErrorHostMemoryAlreadyRegistered) {
    HIP_CHECK(reg);
  }
  void* dev = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(&dev, host, 0));

  auto holder = std::make_shared<at::Tensor>(cpu);
  auto opts = at::TensorOptions().dtype(cpu.scalar_type()).device(at::kCUDA, device);
  return at::from_blob(
      dev, cpu.sizes().vec(), cpu.strides().vec(),
      [holder, host, we_registered](void*) mutable {
        if (we_registered) hipHostUnregister(host);
        holder.reset();
      },
      opts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("managed_empty", &managed_empty, "Allocate a managed-memory CPU tensor",
        pybind11::arg("shape"), pybind11::arg("dtype"));
  m.def("migrate", &migrate, "Alias a managed CPU tensor as a CUDA tensor (no copy)",
        pybind11::arg("cpu"), pybind11::arg("prefetch") = true);
  m.def("register_migrate", &register_migrate,
        "Register + alias any pageable CPU tensor as a CUDA tensor (no copy)",
        pybind11::arg("cpu"));
}
