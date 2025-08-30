import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import os

amu = 1.66053906660e-24
k_B = 1.380649e-16
m_H = 1.6735575e-24

n_H_default = 1e4
a = 1e-5
n_gr_factor = 1e-12
site_density = 1e15
sites_per_grain = site_density * 4 * np.pi * a**2

T_cold = 10.0
T_warm = 100.0
t0 = 1e2
t_warm_start = 1e11
t_warm_end = 1e13
t_final = 1e14

nu = 1e12
E_des = {
    'CO': 1000.0,
    'H': 400.0,
    'HCO': 1500.0,
    'H2CO': 2000.0,
    'CH3OH': 5000.0,
    'NH2CHO': 6000.0,
    'CPX': 8000.0
}

k_CO_H = 1e-10
k_HCO_H = 1e-10
k_H2CO_H = 1e-10

k_ph_base = 1e-17

k_radrec = 1e-16

k_conv_to_cpx = 1e-10

SOLVER_RTR = 1e-8
SOLVER_ATOL = 1e-20

out_dir = "outputs_model"
os.makedirs(out_dir, exist_ok=True)


def n_gr_from_nH(n_H):
    return n_gr_factor * n_H

def thermal_speed(mass_amu, T):
    m = mass_amu * amu
    return np.sqrt(8 * k_B * T / (np.pi * m))

def adsorption_rate_coeff(n_H, mass_amu, T, S=1.0):
    n_gr = n_gr_from_nH(n_H)
    v = thermal_speed(mass_amu, T)
    area = np.pi * a**2
    k_coll = S * v * area * n_gr
    return k_coll

def desorption_rate(nu_local, E_des_K, T):
    return nu_local * np.exp(-E_des_K / max(T, 1e-6))

def surface_encounter_rate(n_H, T):
    n_gr = n_gr_from_nH(n_H)
    E_diff_char = 1000.0
    D_char = nu * np.exp(-E_diff_char / max(T, 1.0))
    return (D_char / sites_per_grain) * n_gr

species_gas = ['CO','H','HCO','H2CO','CH3OH','NH2CHO','CPX']
species_surf = [s + '_s' for s in species_gas]
species_all = species_gas + species_surf
idx = {sp:i for i,sp in enumerate(species_all)}

def initial_abundances(n_H):
    CO = 1e-4 * n_H
    H = 1e-3 * n_H
    HCO = 0.0
    H2CO = 0.0
    CH3OH = 1e-9 * n_H
    NH2CHO = 1e-12 * n_H
    CPX = 0.0
    CO_s = 1e-8 * n_H
    H_s = 1e-8 * n_H
    HCO_s = 0.0
    H2CO_s = 0.0
    CH3OH_s = 0.0
    NH2CHO_s = 0.0
    CPX_s = 0.0
    y0 = np.array([CO,H,HCO,H2CO,CH3OH,NH2CHO,CPX,
                   CO_s,H_s,HCO_s,H2CO_s,CH3OH_s,NH2CHO_s,CPX_s], dtype=float)
    return y0

def odes(t, y, n_H, k_ph_factor, T_func):
    CO, H, HCO, H2CO, CH3OH, NH2CHO, CPX, CO_s, H_s, HCO_s, H2CO_s, CH3OH_s, NH2CHO_s, CPX_s = y
    T = T_func(t)
    k_ads = {}
    k_des = {}
    mass_guess = {'CO':28,'H':1,'HCO':29,'H2CO':30,'CH3OH':32,'NH2CHO':45,'CPX':100}
    for sp in species_gas:
        k_ads[sp] = adsorption_rate_coeff(n_H, mass_guess.get(sp,30), T, S=1.0)
        k_des[sp] = desorption_rate(nu, E_des.get(sp,2000.0), T)

    k_enc = surface_encounter_rate(n_H, T)
    k_ph = k_ph_base * k_ph_factor

    r1 = k_CO_H * CO * H
    r2 = k_HCO_H * HCO * H
    r3 = k_H2CO_H * H2CO * H

    ads = {sp: k_ads[sp] * y[idx[sp]] for sp in species_gas}

    r_s1 = k_enc * CO_s * H_s
    r_s2 = k_enc * HCO_s * H_s
    r_s3 = k_enc * H2CO_s * H_s

    rad_prod_CO = k_ph * CO_s
    rad_prod_NH2CHO = k_ph * NH2CHO_s
    rad_proxy = rad_prod_CO + rad_prod_NH2CHO + k_ph * (CH3OH_s + H2CO_s)

    r_radrec = k_radrec * rad_proxy

    k_formamide_gas = 1e-12
    r_formamide_gas = k_formamide_gas * H2CO * H
    k_formamide_surf = 1e-16
    r_formamide_surf = k_formamide_surf * H2CO_s * NH2CHO_s

    r_conv_to_cpx = k_conv_to_cpx * NH2CHO_s

    des = {sp: k_des[sp] * y[idx[sp + '_s']] for sp in species_gas}

    ph_gas = {sp: k_ph * y[idx[sp]] for sp in ['CO','CH3OH','NH2CHO']}

    dCO = -r1 - ads['CO'] + des['CO'] - ph_gas['CO']
    dH = -r1 - r2 - r3 - ads['H'] + des['H'] + 0.5 * ph_gas['CO']
    dHCO = +r1 - r2 - ads['HCO'] + des['HCO']
    dH2CO = +r2 - r3 - ads['H2CO'] + des['H2CO'] - r_formamide_gas
    dCH3OH = +r3 - ads['CH3OH'] + des['CH3OH'] - ph_gas['CH3OH']
    dNH2CHO = +r_formamide_gas - ads['NH2CHO'] + des['NH2CHO'] - ph_gas['NH2CHO']
    dCPX = -ads['CPX'] + des['CPX'] + r_radrec

    dCO_s = +ads['CO'] - r_s1 - des['CO'] - rad_prod_CO
    dH_s = +ads['H'] - r_s1 - r_s2 - r_s3 - des['H'] + 0.1 * ph_gas['CO']
    dHCO_s = +r_s1 - r_s2 - des['HCO']
    dH2CO_s = +r_s2 - r_s3 - des['H2CO'] - r_formamide_surf
    dCH3OH_s = +r_s3 - des['H2CO'] - k_ph * CH3OH_s
    dNH2CHO_s = +ads['NH2CHO'] + r_formamide_surf - des['NH2CHO'] - r_conv_to_cpx - k_ph * NH2CHO_s
    dCPX_s = +r_radrec + r_conv_to_cpx - des['CPX'] - 1e-12 * CPX_s

    dydt = np.array([dCO,dH,dHCO,dH2CO,dCH3OH,dNH2CHO,dCPX,
                     dCO_s,dH_s,dHCO_s,dH2CO_s,dCH3OH_s,dNH2CHO_s,dCPX_s], dtype=float)

    return dydt

def T_ramp(t):
    if t <= t_warm_start:
        return T_cold
    elif t >= t_warm_end:
        return T_warm
    else:
        frac = (t - t_warm_start) / (t_warm_end - t_warm_start)
        smooth = 0.5 * (1 - np.cos(np.pi * frac))
        return T_cold + smooth * (T_warm - T_cold)

def run_simulation(n_H=n_H_default, k_ph_factor=1.0):
    y0 = initial_abundances(n_H)
    t_span1 = (t0, t_warm_start)
    t_eval1 = np.logspace(np.log10(t0), np.log10(t_warm_start), 200)
    sol1 = solve_ivp(lambda t,y: odes(t,y,n_H,k_ph_factor,T_func=T_ramp),
                     t_span1, y0, method='BDF', t_eval=t_eval1,
                     atol=SOLVER_ATOL, rtol=SOLVER_RTR, max_step=(t_warm_start-t0)/100)
    y1_end = sol1.y[:, -1].copy()
    y1_end[y1_end < 0] = 0.0

    t_span2 = (t_warm_start, t_warm_end)
    t_eval2 = np.logspace(np.log10(t_warm_start), np.log10(t_warm_end), 1000)
    sol2 = solve_ivp(lambda t,y: odes(t,y,n_H,k_ph_factor,T_func=T_ramp),
                     t_span2, y1_end, method='BDF', t_eval=t_eval2,
                     atol=SOLVER_ATOL, rtol=SOLVER_RTR, max_step=(t_warm_end-t_warm_start)/1000)

    y2_end = sol2.y[:, -1].copy()
    y2_end[y2_end < 0] = 0.0

    t_span3 = (t_warm_end, t_final)
    t_eval3 = np.logspace(np.log10(t_warm_end), np.log10(t_final), 200)
    sol3 = solve_ivp(lambda t,y: odes(t,y,n_H,k_ph_factor,T_func=T_ramp),
                     t_span3, y2_end, method='BDF', t_eval=t_eval3,
                     atol=SOLVER_ATOL, rtol=SOLVER_RTR, max_step=(t_final-t_warm_end)/100)
    
    if not sol1.success or not sol2.success or not sol3.success:
        print("Warning: One or more simulation phases failed. Skipping this parameter combination.")
        return np.array([]), np.array([])
    
    t_all = np.concatenate([sol1.t, sol2.t, sol3.t])
    y_all = np.concatenate([sol1.y, sol2.y, sol3.y], axis=1)
    unique_t, unique_indices = np.unique(t_all, return_index=True)
    y_unique = y_all[:, unique_indices]
    y_unique[y_unique < 0] = 0.0
    return unique_t, y_unique

if __name__ == "__main__":
    n_H = n_H_default
    t_all, y_all = run_simulation(n_H=n_H, k_ph_factor=1.0)
    
    if t_all.size == 0:
        print("Initial reference simulation failed, cannot generate plots.")
    else:
        df = pd.DataFrame(y_all.T, columns=species_all)
        rolling_window = 100
        df_smoothed = df.rolling(window=rolling_window, min_periods=1).mean()

        plt.figure(figsize=(9,5))
        plot_species = ['CO','H2CO','CH3OH','NH2CHO','CPX']
        for sp in plot_species:
            plt.plot(t_all, df_smoothed[sp], label=f"{sp} (gas)")
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Time (s)'); plt.ylabel('Number density (cm$^{-3}$)')
        plt.title(f'Gas-phase abundances vs time')
        plt.legend()
        plt.grid(True, which='both', ls=':', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'gas_abundances_vs_time.png'), dpi=300)
        plt.show()

        plt.figure(figsize=(9,5))
        plot_species_s = ['CO_s','H2CO_s','CH3OH_s','NH2CHO_s','CPX_s']
        for sp in plot_species_s:
            plt.plot(t_all, df_smoothed[sp], label=f"{sp} (surface)")
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Time (s)'); plt.ylabel('Surface abundance (cm$^{-3}$)')
        plt.title(f'Surface abundances vs time')
        plt.legend()
        plt.grid(True, which='both', ls=':', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'surface_abundances_vs_time.png'), dpi=300)
        plt.show()

    n_H_values = np.logspace(3,6,12)
    kph_factors = np.logspace(-1,2,12)
    final_CPX = np.full((len(n_H_values), len(kph_factors)), np.nan)
    print("Starting parameter sweep.")
    for i, nH_val in enumerate(n_H_values):
        for j, kf in enumerate(kph_factors):
            t_tmp, y_tmp = run_simulation(n_H=nH_val, k_ph_factor=kf)
            if y_tmp.size > 0:
                final_CPX[i,j] = y_tmp[idx['CPX_s'], -1]
            else:
                final_CPX[i,j] = np.nan

    pd.DataFrame(final_CPX, index=n_H_values, columns=kph_factors).to_csv(os.path.join(out_dir, 'final_cpx_heatmap.csv'))

    plt.figure(figsize=(6,5))
    plt.pcolormesh(kph_factors, n_H_values, np.log10(final_CPX + 1e-30), shading='auto')
    plt.xscale('log'); plt.yscale('log')
    plt.colorbar(label='log10(final CPX_s abundance cm$^{-3}$)')
    plt.xlabel('Photoreaction factor (× baseline)')
    plt.ylabel('n_H (cm$^{-3}$)')
    plt.title('Final surface CPX abundance after model run')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'final_cpx_heatmap.png'), dpi=300)
    plt.show()

    print(f"All outputs saved in: {out_dir}")
