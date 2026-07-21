# ensure that FA2 is working

import torch, sys
import os

def run_unique_fa2_smoke_test():
    print('--- EXECUTING FA2 SMOKE TEST ---')
    try:
        import flash_attn
        from flash_attn import flash_attn_func

        # 1. HARD VERSION CHECK
        # FA3 and FA4 will report versions >= 3.0.0 or 4.0.0
        version = flash_attn.__version__
        if not version.startswith('2.'):
            print(f'FAIL: Version mismatch. Found FA {version}, expected 2.x')
            return False

        # 2. NAMESPACE VERIFICATION
        # Ensures we aren't using a 'kernels-community' wrapper
        origin = flash_attn.__file__
        if 'kernels' in origin or 'vllm' in origin:
            print(f'FAIL: Detected aliased/wrapped FA implementation at {origin}')
            return False

        # 3. EXECUTION TEST
        # We use a specific head dimension (128) and causal masking
        # which triggers the standard FA2 ROCm GEMM kernels.
        q = torch.randn((1, 128, 8, 128), dtype=torch.bfloat16, device='cuda')
        k = torch.randn((1, 128, 8, 128), dtype=torch.bfloat16, device='cuda')
        v = torch.randn((1, 128, 8, 128), dtype=torch.bfloat16, device='cuda')

        # We add 'window_size', which is handled via specific tiling in FA2
        # but often triggers different logic or errors in early FA3/FA4 previews
        out = flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        # 4. ARCHITECTURE COMPATIBILITY CHECK
        # FA2 ROCm specifically targets gfx90a/gfx942 for these dtypes.
        # If this doesn't throw a 'Not implemented' error, the FA2-specific
        # HSA/HIP binary is correctly mapped.

        print(f'FA2 KERNEL SUCCESSFUL (Version: {version}, Path: {origin})')
        return True

    except AttributeError:
        print('FAIL: flash_attn_func not found. This may be an incompatible FA3/FA4 structure.')
        return False
    except Exception as e:
        print(f'FA2 CRITICAL KERNEL FAILURE: {e}')
        return False

if __name__ == '__main__':
    sys.exit(0 if run_unique_fa2_smoke_test() else 1)


