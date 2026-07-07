import sys
import argparse
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import k, m_p, e, Stefan_Boltzmann
import matplotlib.pyplot as plt
import os

# --- Physical Constants ---
E_ALPHA = 3.5e6 * e  
M_DT = 2.5 * m_p     
SIGMA_SB = Stefan_Boltzmann

def nrl_dt_reactivity(T_keV):
    """Unconditionally stable Gamow fit from the NRL Plasma Formulary."""
    if T_keV < 0.5: return 0.0
    T = min(T_keV, 100.0) 
    # NRL formula yields cm^3/s, multiply by 1e-6 for m^3/s
    sigmav_cm3 = 3.68e-12 * (T**(-2.0/3.0)) * np.exp(-19.94 * (T**(-1.0/3.0)))
    return sigmav_cm3 * 1e-6

def asymmetric_odes(t, y, M1, M2, mode_number):
    R1, v1, R2, v2, E_hs, yield_J, rt_int, M_fuel, M_hs = y
    
    # Hard clamps to protect the solver from unphysical undershoots
    R1 = max(R1, 1e-9)
    R2 = max(R2, 1e-9)
    E_hs = max(E_hs, 1e-9)
    M_hs = max(M_hs, 1e-12) 
    M_fuel = max(M_fuel, 0.0) 

    R_eff = (R1 + R2) / 2.0
    V = (4.0 / 3.0) * np.pi * R_eff**3
    P_hs = (2.0 / 3.0) * E_hs / V 
    
    rho_hs = M_hs / V
    n_ion = rho_hs / M_DT
    T_K = P_hs / (2.0 * n_ion * k) if n_ion > 0 else 0.0
    T_keV = T_K / 1.16045e7
    T_keV = max(T_keV, 1e-5)
    
    # Flux-Limited Spitzer Ablation
    T_keV_abl = min(T_keV, 15.0)
    C_abl = 50000.0 
    # Prevent ablation shock at early, cold times
    dM_hs_dt = C_abl * R_eff * (T_keV_abl**2.5) if T_keV > 0.5 else 0.0
    
    sigmav = nrl_dt_reactivity(T_keV)
    
    fuel_fraction = M_fuel / M_hs if M_hs > 0 else 0.0
    n_fuel_ion = n_ion * fuel_fraction
    
    rate_density = (n_fuel_ion / 2.0)**2 * sigmav
    R_rxn = rate_density * V  
    
    W_alpha = R_rxn * E_ALPHA
    
    # Blackbody Radiative Cap
    W_rad_volumetric = 5.34e-37 * (n_ion**2) * np.sqrt(T_keV) * V
    W_blackbody = SIGMA_SB * (T_K**4) * (4.0 * np.pi * R_eff**2)
    W_rad = min(W_rad_volumetric, W_blackbody)

    # Force the yield derivative to be strictly positive
    dW_yield_dt = max(0.0, 5.0 * W_alpha)

    # Replenish fuel with ablated shell mass
    dM_fuel_dt = dM_hs_dt - R_rxn * (2.0 * M_DT)

    A_eff = 4.0 * np.pi * R_eff**2
    dv1_dt = (A_eff * P_hs) / M1
    dv2_dt = (A_eff * P_hs) / M2
    
    dV_dt = A_eff * ((v1 + v2) / 2.0)
    dE_dt = -P_hs * dV_dt + W_alpha - W_rad 
    
    g_eff = (dv1_dt + dv2_dt) / 2.0
    dRT_dt = 0.0
    if g_eff > 0:
        wave_number = mode_number / R_eff 
        dRT_dt = np.sqrt(wave_number * g_eff)
    
    return [v1, dv1_dt, v2, dv2_dt, dE_dt, dW_yield_dt, dRT_dt, dM_fuel_dt, dM_hs_dt]

def plot_synthetic_density(R1, R2, rt_exponent, time_val, step, mode_number, initial_roughness, output_dir="implosion_frames"):
    os.makedirs(output_dir, exist_ok=True)
    grid_size = 600
    L = 200e-6  
    x = np.linspace(-L, L, grid_size)
    y = np.linspace(-L, L, grid_size)
    X, Y = np.meshgrid(x, y)
    
    distance = np.sqrt(X**2 + Y**2)
    angles = np.arctan2(Y, X)
    
    R_theta = np.where(X < 0, R1, R2) 
    rt_amp = initial_roughness * np.exp(rt_exponent)
    R_perturbed = R_theta + rt_amp * np.cos(mode_number * angles)
    
    density_map = np.zeros_like(X)
    shell_thickness = 15e-6
    
    shell_mask = (distance > R_perturbed) & (distance < R_perturbed + shell_thickness)
    hotspot_mask = distance <= R_perturbed
    
    density_map[shell_mask] = 100.0  
    density_map[hotspot_mask] = 10.0 
    
    plt.figure(figsize=(6,6))
    plt.pcolormesh(X*1e6, Y*1e6, density_map, cmap='magma', shading='nearest')
    plt.title(f"Synthetic RT + Mode-1 Model: t = {time_val*1e9:.3f} ns\n(Delta={abs(R1-R2)/(R1+R2):.2f}, l={mode_number})")
    plt.xlabel("X (um)")
    plt.ylabel("Y (um)")
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    
    plt.savefig(f"{output_dir}/frame_{step:03d}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_histories(sol, M1, M2, output_dir="implosion_frames"):
    os.makedirs(output_dir, exist_ok=True)
    
    t_ns = sol.t * 1e9
    R1 = sol.y[0]
    R2 = sol.y[2]
    R_eff = (R1 + R2) / 2.0
    V = (4.0 / 3.0) * np.pi * R_eff**3
    E_hs = sol.y[4]
    M_hs_total = sol.y[8]
    
    rho_hs = M_hs_total / V
    rhoR_hs_gcm2 = (rho_hs * R_eff) * 0.1
    rhoR_shell_gcm2 = ((M1 + M2) / (4.0 * np.pi * R_eff**2)) * 0.1 
    
    P_hs = (2.0 / 3.0) * E_hs / V
    n_ion = rho_hs / M_DT
    T_keV = np.zeros_like(P_hs)
    valid = n_ion > 0
    T_keV[valid] = (P_hs[valid] / (2.0 * n_ion[valid] * k)) / 1.16045e7
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
    axs[0].plot(t_ns, R1*1e6, label='R1 (Left Piston)', color='blue')
    axs[0].plot(t_ns, R2*1e6, label='R2 (Right Piston)', color='red', linestyle='--')
    axs[0].set_ylabel('Radius (um)')
    axs[0].set_title('Implosion Trajectory')
    axs[0].legend()
    axs[0].grid(True, alpha=0.5)
    
    axs[1].plot(t_ns, rhoR_shell_gcm2, label='Shell $\\rho R$', color='green')
    axs[1].plot(t_ns, rhoR_hs_gcm2, label='Hot Spot $\\rho R$', color='orange')
    axs[1].set_ylabel('Areal Density (g/cm$^2$)')
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].grid(True, alpha=0.5)
    
    axs[2].plot(t_ns, T_keV, label='$T_{ion} \\approx T_{elec}$', color='purple')
    axs[2].set_ylabel('Temperature (keV)')
    axs[2].set_xlabel('Time (ns)')
    axs[2].legend()
    axs[2].grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_histories.png", dpi=300, bbox_inches='tight')
    plt.close()

def solve_implosion(R0, v0, T0, M_sh, M_hs, delta, mode, roughness):
    """Integrate the deceleration-phase ODE system for a given capsule design.

    Returns (sol, M1, M2): the solve_ivp solution object and the two piston masses.
    All inputs are in SI units except T0 which is in keV.
    """
    M1 = (M_sh / 2.0) * (1.0 + delta)
    M2 = (M_sh / 2.0) * (1.0 - delta)

    V0 = (4.0 / 3.0) * np.pi * R0**3
    P_hs0 = 2.0 * (M_hs / V0 / M_DT) * k * (T0 * 1.16045e7)
    E_hs0 = 1.5 * P_hs0 * V0

    y0 = [R0, v0, R0, v0, E_hs0, 0.0, 0.0, M_hs, M_hs]

    t_span = (0, 0.6e-9)
    t_eval = np.linspace(0, 0.6e-9, 200)

    # Safely back on Radau. No custom tolerance arrays to trip it up.
    sol = solve_ivp(asymmetric_odes, t_span, y0, t_eval=t_eval, method='Radau',
                    args=(M1, M2, mode), rtol=1e-6, atol=1e-8, max_step=1e-12)

    return sol, M1, M2


def compute_metrics(sol, M1, M2, M_hs, mode):
    """Compute stagnation metrics from an integrated solution. Returns a dict."""
    R_eff_array = (sol.y[0] + sol.y[2]) / 2.0
    stag_idx = np.argmin(R_eff_array)

    R1_stag = sol.y[0][stag_idx]
    R2_stag = sol.y[2][stag_idx]
    v1_stag = sol.y[1][stag_idx]
    v2_stag = sol.y[3][stag_idx]

    RKE = 0.5 * M1 * v1_stag**2 + 0.5 * M2 * v2_stag**2

    E_hs_stag = sol.y[4][stag_idx]
    V_stag = (4.0 / 3.0) * np.pi * ((R1_stag + R2_stag) / 2.0)**3
    P_stag = (2.0 / 3.0) * E_hs_stag / V_stag

    rt_exponent = sol.y[6][stag_idx]
    rt_amplification = np.exp(rt_exponent)

    # Force diagnostic yield to non-negative just in case
    final_yield_MJ = max(0.0, sol.y[5][-1] / 1e6)

    initial_fuel_ug = M_hs * 1e9
    final_fuel_ug = max(0.0, sol.y[7][-1] * 1e9)
    total_ablated_mass_ug = max(0.0, (sol.y[8][-1] - M_hs) * 1e9)

    total_pool = initial_fuel_ug + total_ablated_mass_ug
    burned_fraction = ((total_pool - final_fuel_ug) / total_pool * 100.0) if total_pool > 0 else 0.0

    return {
        "stag_idx": stag_idx,
        "R1_stag": R1_stag,
        "R2_stag": R2_stag,
        "P_stag_Gbar": P_stag / 1e14,
        "yield_MJ": final_yield_MJ,
        "ablated_ug": total_ablated_mass_ug,
        "burn_fraction": burned_fraction,
        "total_pool_ug": total_pool,
        "RKE_J": RKE,
        "rt_amplification": rt_amplification,
        "mode": mode,
    }


def main():
    parser = argparse.ArgumentParser(description="2D Asymmetric Piston Surrogate")
    parser.add_argument('--R0', type=float, default=150e-6)
    parser.add_argument('--v0', type=float, default=-450e3)
    parser.add_argument('--T0', type=float, default=1.0)
    parser.add_argument('--M_sh', type=float, default=0.25e-6)
    parser.add_argument('--M_hs', type=float, default=0.005e-6)
    parser.add_argument('--delta', type=float, default=0.0)
    parser.add_argument('--mode', type=float, default=10.0)
    parser.add_argument('--roughness', type=float, default=0.1e-6)
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()

    print(f"Simulating implosion (Delta = {args.delta}, Mode = {args.mode})...")

    sol, M1, M2 = solve_implosion(args.R0, args.v0, args.T0, args.M_sh, args.M_hs,
                                  args.delta, args.mode, args.roughness)

    t_eval_len = 200
    if len(sol.t) < t_eval_len:
        print(f"Warning: ODE solver stopped early at step {len(sol.t)}.")

    m = compute_metrics(sol, M1, M2, args.M_hs, args.mode)
    stag_idx = m["stag_idx"]

    if args.plot:
        print("Generating high-resolution PNGs and time-history traces...")
        output_dir = "implosion_frames"
        plot_time_histories(sol, M1, M2, output_dir=output_dir)

        end_idx = min(stag_idx + 20, len(sol.t))
        for i in range(0, end_idx, 2):
            plot_synthetic_density(
                sol.y[0][i], sol.y[2][i], sol.y[6][i], sol.t[i], i,
                args.mode, args.roughness, output_dir=output_dir
            )
        print(f"Saved visualization files to the '{output_dir}/' directory.")

    print(f"\n--- Asymmetric Mode-1 Metrics (Delta = {args.delta}) ---")
    print(f"Stagnation Radius (Left):  {m['R1_stag']*1e6:.2f} um")
    print(f"Stagnation Radius (Right): {m['R2_stag']*1e6:.2f} um")
    print(f"Stagnation Pressure:       {m['P_stag_Gbar']:.2f} Gbar")
    print(f"Total Fusion Yield:        {m['yield_MJ']:.3f} MJ")
    print(f"Ablated Mass Injected:     {m['ablated_ug']:.2f} ug")
    print(f"Fuel Burn Fraction:        {m['burn_fraction']:.2f}% (Total Pool: {m['total_pool_ug']:.2f} ug)")
    print(f"Residual Kinetic Energy:   {m['RKE_J']:.2f} Joules")
    print(f"RT Amplification (l={args.mode}): {m['rt_amplification']:.2f}x initial perturbation")


if __name__ == "__main__":
    main()
