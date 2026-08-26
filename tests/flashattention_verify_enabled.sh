#!/bin/bash
#
# This test verifies that Flash Attention is not merely importable (see
# flashattention_check_import.sh for the bare `import flash_attn`) but is
# actually WIRED IN and picked up by the stack, and it doubles as a
# standalone diagnostic you can run on any PyTorch setup to see what is
# going on.
#
#   INFO ON THE OUTPUT PROVIDED BY THE TEST
#
#   (A) REPORTS, for the loaded build:
#       * torch / ROCm(HIP) / transformers versions and the GPU name;
#       * where flash_attn is imported from and which compiled extension
#           (flash_attn_2_cuda*.so -- named "cuda" even on ROCm) is mapped,
#           plus whether the build is a ROCm/HIP build;
#       * whether torch's own SDPA flash backend (aotriton) is available --
#           this is a SEPARATE "flash attention" from the flash_attn package
#           and is reported for information only (not gated).
#
#   (B) VERIFIES (these two gate the result):
#       * the compiled flash_attn kernel actually runs on the GPU and its
#           output matches a fp32 reference (proves the build is functional,
#           not just importable);
#       * transformers dispatches to attn_implementation="flash_attention_2"
#           without silently falling back (proves transformers picks it up).
#
#   The final line is "FLASH ATTENTION: OK ..." when both gated checks pass
#   (CTest gates on that keyword), "FLASH ATTENTION: FAIL - <reason>" when a
#   gated check fails, and the script emits "FLASH ATTENTION CHECK: SKIPPED"
#   (CTest skip, exit 77) when torch cannot be imported at all.
#
# NOTE: this test assumes Flash Attention has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh

set -u

# ---------------------------------------------------------------------------
# If rocm AND pytorch are already loaded (e.g. by the caller), use them as-is
# and do NOT reload. Otherwise load rocm (if needed) then pytorch. (Match the
# "rocm/" / "pytorch/" aliases so that e.g. "rocm-new/..." does not count.) If
# no module system is present (e.g. a plain venv) these calls just no-op and
# the venv's python3 is used.
# ---------------------------------------------------------------------------
if ! (module -t list 2>&1 | grep -q "^rocm/") || ! (module -t list 2>&1 | grep -q "^pytorch/"); then
  if ! module -t list 2>&1 | grep -q "^rocm/"; then
    echo "rocm module is not loaded"
    echo "loading default rocm module"
    module load rocm
  fi
  if ! module -t list 2>&1 | grep -q "^pytorch/"; then
    module load pytorch
  fi
fi

# ---------------------------------------------------------------------------
# Everything else is Python introspection of the loaded build. We capture the
# output so that if torch fails to import / aborts (environmental) we can emit
# SKIP instead of a bogus failure.
# ---------------------------------------------------------------------------
_OUT=$(mktemp)
trap 'rm -f "${_OUT}"' EXIT

python3 <<'EOF' 2>&1 | tee "${_OUT}"
import os
import re
import sys


def rule(t):
    print("\n" + "-" * 72 + f"\n{t}\n" + "-" * 72)


def loaded_from_maps(pattern):
    hits = set()
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                p = line.rsplit(" ", 1)[-1].strip()
                if p and re.search(pattern, os.path.basename(p), re.I):
                    hits.add(os.path.realpath(p))
    except FileNotFoundError:
        pass
    return sorted(hits)


# torch import failure is environmental -> SKIP (mirrors comm-backends test).
try:
    import torch
except Exception as e:
    print("could not import torch:", e)
    print("FLASH ATTENTION CHECK: SKIPPED")
    sys.exit(77)

is_rocm = getattr(torch.version, "hip", None) is not None

rule("Environment")
print("torch.__version__ :", torch.__version__)
print("torch.version.hip :", getattr(torch.version, "hip", None))
print("torch.version.cuda:", torch.version.cuda)
print("LOADEDMODULES     :", os.environ.get("LOADEDMODULES", "(not set)"))
if torch.cuda.is_available():
    print("GPU device count  :", torch.cuda.device_count())
    print("GPU device name   :", torch.cuda.get_device_name(0))
else:
    print("GPU               : NOT available (torch.cuda.is_available() == False)")

# Track the two gated outcomes; None = not determined.
kernel_ok = None
transformers_ok = None
fail_reason = None

# ---------------------------------------------------------------------------
rule("flash_attn package")
try:
    import flash_attn
    from flash_attn import flash_attn_func
    print("flash_attn version:", flash_attn.__version__)
    print("imported from     :", os.path.dirname(flash_attn.__file__))
    print("ROCm/HIP build    :", "yes" if is_rocm else "no (CUDA build)")
except Exception as e:
    print("import flash_attn FAILED:", e)
    flash_attn_func = None
    kernel_ok = False
    fail_reason = f"flash_attn not importable ({e})"

# ---------------------------------------------------------------------------
rule("flash_attn GPU kernel + correctness")
if flash_attn_func is None:
    print("skipped -- flash_attn did not import")
elif not torch.cuda.is_available():
    kernel_ok = False
    fail_reason = "no GPU available to run the flash_attn kernel"
    print("skipped -- no GPU available")
else:
    try:
        torch.manual_seed(0)
        dev, dt = "cuda", torch.bfloat16
        b, s, h, d = 2, 512, 8, 64
        q = torch.randn(b, s, h, d, device=dev, dtype=dt)
        k = torch.randn(b, s, h, d, device=dev, dtype=dt)
        v = torch.randn(b, s, h, d, device=dev, dtype=dt)
        out = flash_attn_func(q, k, v, causal=True)
        qf, kf, vf = (x.float().transpose(1, 2) for x in (q, k, v))
        scores = (qf @ kf.transpose(-1, -2)) / (d ** 0.5)
        mask = torch.triu(torch.ones(s, s, device=dev, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        ref = (scores.softmax(-1) @ vf).transpose(1, 2)
        err = (out.float() - ref).abs().max().item()
        mapped = loaded_from_maps(r"flash_attn")
        print("compiled ext mapped:", mapped if mapped else "(none found in /proc/self/maps)")
        print(f"kernel ran on GPU, shape={tuple(out.shape)}, max_err_vs_fp32={err:.4g}")
        kernel_ok = err < 2e-2
        if not kernel_ok:
            fail_reason = f"flash_attn output mismatch (max_err={err:.4g})"
        print("verdict           :", "OK" if kernel_ok else "MISMATCH")
    except Exception as e:
        kernel_ok = False
        fail_reason = f"flash_attn kernel execution failed ({e})"
        print("kernel execution FAILED:", e)

# ---------------------------------------------------------------------------
rule("transformers dispatch to flash_attention_2")
if not torch.cuda.is_available():
    transformers_ok = False
    fail_reason = fail_reason or "no GPU available to exercise the transformers FA2 path"
    print("skipped -- no GPU available")
else:
    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM
        print("transformers version:", transformers.__version__)
        cfg = AutoConfig.for_model("llama", hidden_size=256, num_hidden_layers=2,
                                   num_attention_heads=8, num_key_value_heads=8,
                                   intermediate_size=512, vocab_size=1000)
        model = AutoModelForCausalLM.from_config(
            cfg, attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16).to("cuda")
        impl = model.config._attn_implementation
        print("resolved _attn_implementation:", impl)
        ids = torch.randint(0, 1000, (1, 128), device="cuda")
        model(ids)
        print("forward pass through FA2 model: OK")
        transformers_ok = (impl == "flash_attention_2")
        if not transformers_ok:
            fail_reason = fail_reason or f"transformers fell back to '{impl}' instead of flash_attention_2"
        print("verdict             :", "OK" if transformers_ok else "FELL BACK")
    except Exception as e:
        transformers_ok = False
        fail_reason = fail_reason or f"transformers FA2 dispatch failed ({e})"
        print("transformers FA2 dispatch FAILED:", e)

# ---------------------------------------------------------------------------
# Informational only (NOT gated): torch's own SDPA flash backend is a separate
# "flash attention" (aotriton on ROCm), used by attn_implementation="sdpa".
rule("torch SDPA flash backend (informational, not gated)")
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if torch.cuda.is_available():
        qs = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            torch.nn.functional.scaled_dot_product_attention(qs, qs, qs, is_causal=True)
        print("SDPA FLASH_ATTENTION backend: available")
    else:
        print("SDPA FLASH_ATTENTION backend: not checked (no GPU)")
except Exception as e:
    print("SDPA FLASH_ATTENTION backend: unavailable ->", e)

# ---------------------------------------------------------------------------
rule("Summary")
print(f"  flash_attn kernel correctness : {kernel_ok}")
print(f"  transformers FA2 dispatch     : {transformers_ok}")
if kernel_ok and transformers_ok:
    print("FLASH ATTENTION: OK - flash_attn kernel verified on GPU and "
          "transformers dispatches to flash_attention_2")
    sys.exit(0)
else:
    print(f"FLASH ATTENTION: FAIL - {fail_reason}")
    sys.exit(1)
EOF

# ---------------------------------------------------------------------------
# CTest gates on the "FLASH ATTENTION: OK" keyword (PASS_REGULAR_EXPRESSION).
# A torch-import failure emits "FLASH ATTENTION CHECK: SKIPPED" -> CTest skip.
# ---------------------------------------------------------------------------
if grep -q "FLASH ATTENTION CHECK: SKIPPED" "${_OUT}"; then
  exit 77
elif grep -q "^FLASH ATTENTION: OK" "${_OUT}"; then
  exit 0
else
  exit 1
fi
