"""Media generation for the ICF surrogate: implosion movies and diagnostic plots.

Uses the programmatic API in icf_core (solve_implosion / plot_time_histories) to render
an animation of the density field and the time-history diagnostic traces. Designed to run
headless (Agg backend) and to produce a fixed-size MP4 via ffmpeg.
"""
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

import icf_core

# Point matplotlib at whatever ffmpeg is on PATH (e.g. ~/bin/ffmpeg).
_ffmpeg = shutil.which("ffmpeg")
if _ffmpeg:
    plt.rcParams["animation.ffmpeg_path"] = _ffmpeg


def render_implosion_movie(params, outdir, fps=12):
    """Render an MP4 animation of the density field through stagnation.

    params: dict with keys R0, v0, T0, M_sh, M_hs, delta, mode, roughness (SI; T0 in keV).
    Returns the path to the written MP4.
    """
    os.makedirs(outdir, exist_ok=True)
    sol, M1, M2 = icf_core.solve_implosion(
        params["R0"], params["v0"], params["T0"], params["M_sh"], params["M_hs"],
        params["delta"], params["mode"], params["roughness"],
    )

    mode = params["mode"]
    roughness = params["roughness"]

    # Animate from start through ~20 steps past stagnation
    R_eff_array = (sol.y[0] + sol.y[2]) / 2.0
    stag_idx = int(np.argmin(R_eff_array))
    end_idx = min(stag_idx + 20, len(sol.t))
    frame_indices = list(range(0, end_idx, 2))
    if not frame_indices:
        frame_indices = [0]

    # Fixed canvas so every frame is identical in size (required for a valid MP4).
    grid_size = 400
    L = 200e-6
    x = np.linspace(-L, L, grid_size)
    y = np.linspace(-L, L, grid_size)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    angles = np.arctan2(Y, X)
    shell_thickness = 15e-6

    fig, ax = plt.subplots(figsize=(6, 6))

    def _density_map(R1, R2, rt_exponent):
        R_theta = np.where(X < 0, R1, R2)
        rt_amp = roughness * np.exp(rt_exponent)
        R_perturbed = R_theta + rt_amp * np.cos(mode * angles)
        dmap = np.zeros_like(X)
        shell_mask = (distance > R_perturbed) & (distance < R_perturbed + shell_thickness)
        hotspot_mask = distance <= R_perturbed
        dmap[shell_mask] = 100.0
        dmap[hotspot_mask] = 10.0
        return dmap

    dmap0 = _density_map(sol.y[0][0], sol.y[2][0], sol.y[6][0])
    quad = ax.pcolormesh(X * 1e6, Y * 1e6, dmap0, cmap="magma",
                         shading="nearest", vmin=0, vmax=100)
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    title = ax.set_title("")

    def _update(frame_no):
        i = frame_indices[frame_no]
        R1 = sol.y[0][i]
        R2 = sol.y[2][i]
        dmap = _density_map(R1, R2, sol.y[6][i])
        quad.set_array(dmap.ravel())
        delta = abs(R1 - R2) / (R1 + R2) if (R1 + R2) > 0 else 0.0
        title.set_text(f"t = {sol.t[i]*1e9:.3f} ns  (Delta={delta:.2f}, l={mode:.0f})")
        return quad, title

    anim = FuncAnimation(fig, _update, frames=len(frame_indices), blit=False)
    mp4_path = os.path.join(outdir, "implosion.mp4")
    writer = FFMpegWriter(fps=fps, metadata={"title": "ICF implosion"}, bitrate=2400)
    anim.save(mp4_path, writer=writer, dpi=120)
    plt.close(fig)
    return mp4_path


def render_diagnostics(params, outdir):
    """Render the time-history diagnostic traces (radius, areal density, temperature).

    Returns the path to the written PNG.
    """
    os.makedirs(outdir, exist_ok=True)
    sol, M1, M2 = icf_core.solve_implosion(
        params["R0"], params["v0"], params["T0"], params["M_sh"], params["M_hs"],
        params["delta"], params["mode"], params["roughness"],
    )
    icf_core.plot_time_histories(sol, M1, M2, output_dir=outdir)
    return os.path.join(outdir, "time_histories.png")
