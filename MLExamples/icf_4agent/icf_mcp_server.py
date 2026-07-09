from mcp.server.fastmcp import FastMCP
import subprocess
import sys
import os

mcp = FastMCP("ICF_Surrogate_Engine")

# Hard physical bounds — values outside these are unphysical or break the solver
PARAM_BOUNDS = {
    "R0":        (30e-6,  500e-6),   # stagnation radius: 30-500 um
    "v0":        (-800e3, -100e3),    # implosion velocity: 100-800 km/s (negative = inward)
    "T0":        (0.1,    10.0),      # initial hotspot temperature: 0.1-10 keV
    "M_sh":      (0.01e-6, 2.0e-6),  # shell mass: 0.01-2.0 ug
    "M_hs":      (1e-9,   0.1e-6),   # hotspot mass: 0.001-100 ng
    "delta":     (0.0,    0.15),      # mode-1 asymmetry fraction: 0-15%
    "mode":      (2.0,    40.0),      # RT mode number
    "roughness": (0.0,    3.0e-6),   # surface roughness: 0-3 um
}

def _clamp(name, value):
    lo, hi = PARAM_BOUNDS[name]
    clamped = max(lo, min(hi, value))
    if clamped != value:
        return clamped, f"  WARNING: {name}={value:.4e} clamped to [{lo:.4e}, {hi:.4e}] -> {clamped:.4e}\n"
    return clamped, ""

@mcp.tool()
def run_icf_implosion(
    R0: float, v0: float, T0: float, M_sh: float, M_hs: float,
    delta: float, mode: float, roughness: float
) -> str:
    """
    Runs the ICF implosion surrogate and returns the physics metrics.
    All inputs must be in strict SI units (meters, kg, m/s).

    Parameter bounds enforced by the server:
      R0:        [30e-6, 500e-6] m         (stagnation radius)
      v0:        [-800e3, -100e3] m/s       (implosion velocity, must be negative)
      T0:        [0.1, 10.0] keV            (initial hotspot temperature)
      M_sh:      [0.01e-6, 2.0e-6] kg      (shell mass)
      M_hs:      [1e-9, 0.1e-6] kg         (initial hotspot mass)
      delta:     [0.0, 0.15]               (mode-1 asymmetry fraction)
      mode:      [2, 40]                   (RT perturbation mode number)
      roughness: [0.0, 3.0e-6] m           (surface roughness)
    """
    warnings = ""
    R0, w = _clamp("R0", R0); warnings += w
    v0, w = _clamp("v0", v0); warnings += w
    T0, w = _clamp("T0", T0); warnings += w
    M_sh, w = _clamp("M_sh", M_sh); warnings += w
    M_hs, w = _clamp("M_hs", M_hs); warnings += w
    delta, w = _clamp("delta", abs(delta)); warnings += w
    mode, w = _clamp("mode", mode); warnings += w
    roughness, w = _clamp("roughness", roughness); warnings += w

    icf_core = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icf_core.py")
    command = [
        sys.executable, icf_core,
        f"--R0={R0}", f"--v0={v0}", f"--T0={T0}",
        f"--M_sh={M_sh}", f"--M_hs={M_hs}",
        f"--delta={delta}", f"--mode={mode}", f"--roughness={roughness}"
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return warnings + result.stdout
    except subprocess.CalledProcessError as e:
        return warnings + f"Simulation failed to integrate (Stiff physics boundaries). Error: {e.stderr}"

if __name__ == "__main__":
    mcp.run()
