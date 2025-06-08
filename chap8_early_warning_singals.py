
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import Normalize
from matplotlib.cm import viridis
import os
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Any, Optional

# --- Configuration ---
NUM_COLLAPSE_RUNS_PER_SCENARIO = 50 # Reduced for quicker testing, set back to 50 for full runs
NUM_STABLE_RUNS = 50               # Reduced for quicker testing
EPSILON = 1e-9
SHOW_PLOTS = False # <<< PLOTS ARE OFF BY DEFAULT
RESULTS_DIR = "results"

# Analysis parameters
DEFAULT_WARNING_HORIZON_FRACTION = 0.25
DEFAULT_AGGREGATION_PERCENTILE_RISING = 95
DEFAULT_AGGREGATION_PERCENTILE_FALLING = 5
DEFAULT_K_FOLDS = 5
DEFAULT_SHUFFLE_K_FOLD = True
DEFAULT_MASTER_SIM_SEED = 12345
DEFAULT_K_FOLD_SEED = 42

DEFAULT_SPEED_EXP_PERCENTILE = 75
DEFAULT_COUPLE_EXP_NEG_PERCENTILE = 25 # Lower percentile means "more negative" or "less coupled"
DEFAULT_UHB_INDICATOR_THRESHOLD_PERCENTILE = 90

# Ensemble Configuration
ENSEMBLE_MEMBERS = ['Speed', 'FACR', 'RMA_norm', 'H_UHB']
ENSEMBLE_EWS_NAME = 'Ensemble_AvgScore_Top4'


# --- Core TD Functions ---
def calculate_tolerance_sheet(g: float, beta: float, F_crit: float,
                              w1: float, w2: float, w3: float, C: float = 1.0) -> float:
    g_calc = max(g, EPSILON)
    beta_calc = max(beta, EPSILON)
    F_crit_calc = max(F_crit, EPSILON)
    if F_crit <= 0 and w3 > 0: return 0.0 # If F_crit is gone and it has weight, tolerance is zero
    return C * (g_calc**w1) * (beta_calc**w2) * (F_crit_calc**w3)

# --- System Dynamics Simulation Function ---
def simulate_system_run(params: Dict[str, Any],
                        run_id: Any = 0,
                        rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng()

    dt = params.get('dt', 0.1)
    max_steps = params.get('max_steps', 1000)
    w = params.get('w', (1/3, 1/3, 1/3)); w1, w2, w3 = w
    C_tolerance = params.get('C_tolerance', 1.0)
    g_initial = params.get('g_initial', 10.0)
    beta_initial = params.get('beta_initial', 2.0)
    F_crit_initial = params.get('F_crit_initial', 20.0)
    sigma_g_flow = params.get('sigma_g_flow', 0.1)
    sigma_beta_flow = params.get('sigma_beta_flow', 0.1)
    sigma_F_crit_flow = params.get('sigma_F_crit_flow', 0.1)
    sigma_Y_obs_base = params.get('sigma_Y_obs_base', 1.0)
    sigma_Y_obs_state_dep_additive_scale = params.get('sigma_Y_obs_state_dep_additive_scale', 0.0)
    sigma_strain = params.get('sigma_strain', 0.05)
    initial_strain = params.get('initial_strain', 3.0)
    strain_increase_rate_mean = params.get('strain_increase_rate_mean', 0.0)
    strain_increase_rate_sigma = params.get('strain_increase_rate_sigma', 0.0)
    g_min = params.get('g_min', 0.5); g_max = params.get('g_max', 25.0)
    k_g_strain_response = params.get('k_g_strain_response', 0.05)
    g_target_stress_ratio = params.get('g_target_stress_ratio', 0.8)
    g_decay_rate_lever = params.get('g_decay_rate_lever', 0.01)
    g_cost_F_crit_scale_denom = params.get('g_cost_F_crit_scale_denom', 0.5)
    F_crit_target_beta = params.get('F_crit_target_beta', 12.0)
    k_beta_increase_stress = params.get('k_beta_increase_stress', 0.1)
    beta_decay_rate = params.get('beta_decay_rate', 0.005)
    beta_min = params.get('beta_min', 0.1); beta_max = params.get('beta_max', 10.0)
    beta_max_critical_level = params.get('beta_max_critical_level', 9.5)
    beta_max_critical_duration_steps = params.get('beta_max_critical_duration_steps', 50)
    F_crit_replenishment_base_mean = params.get('F_crit_replenishment_base_mean', 0.3)
    F_crit_replenishment_sigma = params.get('F_crit_replenishment_sigma', 0.0)
    F_crit_low_threshold_factor = params.get('F_crit_low_threshold_factor', 0.15)
    k_F_crit_low_bonus_factor = params.get('k_F_crit_low_bonus_factor', 3.0)
    k_F_beta_cost = params.get('k_F_beta_cost', 0.02)
    k_F_g_cost = params.get('k_F_g_cost', 0.02)
    F_crit_baseline_cost = params.get('F_crit_baseline_cost', 0.05)
    F_crit_safe_floor = params.get('F_crit_safe_floor', 0.01)
    couple_window = params.get('couple_window', 20)
    trad_ews_window = params.get('trad_ews_window', 30) # Used for Var_Y, AR1_Y, Skew_Y, Kurt_Y, Var_Theta_T, AR1_Theta_T
    velocity_smoothing_window = params.get('velocity_smoothing_window', 15)
    sigma_dot_theta_window = params.get('sigma_dot_theta_window', 30)
    Y_obs_base_level = params.get('Y_obs_base_level', 5.0)
    k_Y_g_lin = params.get('k_Y_g_lin', 0.002); k_Y_beta_lin = params.get('k_Y_beta_lin', -0.005); k_Y_F_crit_lin = params.get('k_Y_F_crit_lin', 0.01)
    k_Y_g_nl_amp = params.get('k_Y_g_nl_amp', 0.05); k_Y_beta_nl_amp = params.get('k_Y_beta_nl_amp', 0.1); k_Y_F_crit_nl_amp = params.get('k_Y_F_crit_nl_amp', 0.2)
    k_Y_gb_interact = params.get('k_Y_gb_interact', 0.0001); k_Y_gF_interact = params.get('k_Y_gF_interact', 0.00005); k_Y_betaF_interact = params.get('k_Y_betaF_interact', -0.0002)
    ext_factor_amp_Y = params.get('ext_factor_amp_Y', 0.5)
    ext_factor_period_Y_frac = params.get('ext_factor_period_Y_frac', 0.2)
    Y_obs_lag_g = params.get('Y_obs_lag_g', 0); Y_obs_lag_beta = params.get('Y_obs_lag_beta', 0); Y_obs_lag_F_crit = params.get('Y_obs_lag_F_crit', 0)
    phi_1_lci = params.get('phi_1_lci', params.get('phi_1', 0.75))
    phi_beta_lci = params.get('phi_beta_lci', params.get('phi_beta', 1.0))
    k_g_lci = params.get('k_g_lci', 0.1); k_beta_lci = params.get('k_beta_lci', 0.1)
    lci_F_crit_saturation_offset = params.get('lci_F_crit_saturation_offset', 0.5)
    proj_time_denom_offset = params.get('proj_time_denom_offset', 1.0)
    uhb_epsilon_sim = params.get('uhb_epsilon', EPSILON) # Renamed to avoid clash in perform_ensemble_analysis

    t_series = np.arange(0, max_steps * dt, dt)
    num_actual_steps = len(t_series)
    ext_factor_period_Y = max_steps * dt * ext_factor_period_Y_frac

    g = np.zeros(num_actual_steps); g[0] = g_initial
    beta = np.zeros(num_actual_steps); beta[0] = beta_initial
    F_crit = np.zeros(num_actual_steps); F_crit[0] = F_crit_initial
    g_hist = [g_initial] * max(1, Y_obs_lag_g)
    beta_hist = [beta_initial] * max(1, Y_obs_lag_beta)
    F_crit_hist = [F_crit_initial] * max(1, Y_obs_lag_F_crit)
    raw_dot_g = np.zeros(num_actual_steps); raw_dot_beta = np.zeros(num_actual_steps); raw_dot_F_crit = np.zeros(num_actual_steps)
    dot_g_smooth = np.full(num_actual_steps, np.nan); dot_beta_smooth = np.full(num_actual_steps, np.nan); dot_F_crit_smooth = np.full(num_actual_steps, np.nan)
    Theta_T = np.zeros(num_actual_steps); Theta_T[0] = calculate_tolerance_sheet(g[0], beta[0], F_crit[0], w1, w2, w3, C_tolerance)
    strain = np.zeros(num_actual_steps); strain[0] = initial_strain
    raw_dot_strain = np.zeros(num_actual_steps)
    dot_strain_smooth = np.full(num_actual_steps, np.nan)

    Y_obs = np.zeros(num_actual_steps)
    speed_idx = np.full(num_actual_steps, np.nan)
    couple_idx = np.full(num_actual_steps, np.nan)
    variance_Y = np.full(num_actual_steps, np.nan)
    ar1_Y = np.full(num_actual_steps, np.nan)
    skewness_Y = np.full(num_actual_steps, np.nan)
    kurtosis_Y = np.full(num_actual_steps, np.nan)
    dot_Theta_T_norm_vals = np.full(num_actual_steps, np.nan)
    H_UHB_vals = np.full(num_actual_steps, np.nan)
    S_3D_weighted_vals = np.full(num_actual_steps, np.nan)
    dot_G_vals = np.full(num_actual_steps, np.nan)
    LCI_vals = np.full(num_actual_steps, np.nan)
    Sigma_dot_Theta_T_norm_vals = np.full(num_actual_steps, np.nan)
    dot_LCI_S_vals = np.full(num_actual_steps, np.nan)
    Fcrit_DepVelN_vals = np.full(num_actual_steps, np.nan)
    PTTF_LCI_vals = np.full(num_actual_steps, np.nan)
    RelCostThetaT_vals = np.full(num_actual_steps, np.nan)
    Fcrit_AccDef_vals = np.full(num_actual_steps, np.nan)

    TMCRN_vals = np.full(num_actual_steps, np.nan)
    Variance_Theta_T_vals = np.full(num_actual_steps, np.nan)
    AR1_Theta_T_vals = np.full(num_actual_steps, np.nan)
    RMA_norm_vals = np.full(num_actual_steps, np.nan)
    FACR_vals = np.full(num_actual_steps, np.nan)
    CCD_vals = np.full(num_actual_steps, np.nan)

    accumulated_F_crit_deficit_tracker = 0.0
    raw_dot_LCI_series = np.zeros(num_actual_steps)
    raw_dot_Theta_T_Margin_series = np.zeros(num_actual_steps)

    beta_above_crit_counter = 0
    breached = False; breach_time = -1.0; breach_step_idx = -1; breach_type = "None"
    ews_calc_df = pd.DataFrame(index=np.arange(num_actual_steps))
    F_crit_low_threshold_abs = F_crit_initial * F_crit_low_threshold_factor

    for i in range(num_actual_steps - 1):
        t = t_series[i]

        if params.get('run_type') == 'stable' and params.get('stable_apply_sub_shocks', False):
            if rng.random() < params.get('stable_shock_prob', 0):
                F_crit_sub_shock = (rng.random() - 0.5) * 2 * params['F_crit_initial'] * params['stable_shock_F_crit_max_delta_frac']
                F_crit[i] = max(F_crit_safe_floor, F_crit[i] + F_crit_sub_shock)
            if rng.random() < params.get('stable_shock_prob', 0):
                strain_sub_shock = (rng.random() - 0.5) * 2 * params['stable_shock_strain_max_delta']
                strain[i] = max(0.0, strain[i] + strain_sub_shock)

        if params.get('apply_shock', False) and abs(t - params.get('shock_time', -1)) < dt / 2:
            if 'shock_F_crit_delta' in params and params['shock_F_crit_delta'] != 0.0:
                 F_crit[i] = max(0.0, F_crit[i] + params['shock_F_crit_delta'])
            if 'shock_strain_delta' in params and params['shock_strain_delta'] != 0.0:
                 strain[i] = max(0.0, strain[i] + params['shock_strain_delta'])
            if 'shock_g_delta' in params and params['shock_g_delta'] != 0.0:
                 g[i] = max(g_min, g[i] + params['shock_g_delta'])

        current_g = g[i]; current_beta = beta[i]; current_F_crit = F_crit[i]
        current_strain = strain[i]
        Theta_T[i] = calculate_tolerance_sheet(current_g, current_beta, current_F_crit, w1, w2, w3, C_tolerance)
        ews_calc_df.loc[i, 'Theta_T_series'] = Theta_T[i]

        g_noise = rng.normal(0, sigma_g_flow) * np.sqrt(dt) if sigma_g_flow > 0 else 0
        stress_metric_for_g = current_strain / (Theta_T[i] + EPSILON)
        g_increase_drive = k_g_strain_response * max(0, stress_metric_for_g - g_target_stress_ratio)
        g_resource_modulator = min(1.0, max(0.0, current_F_crit / (F_crit_initial * g_cost_F_crit_scale_denom + EPSILON)))
        raw_dot_g[i] = (g_increase_drive - g_decay_rate_lever * (current_g - g_initial * 0.8)) * g_resource_modulator + g_noise

        resource_stress_for_beta = max(0, (F_crit_target_beta - current_F_crit) / (F_crit_target_beta + EPSILON))
        beta_noise = rng.normal(0, sigma_beta_flow) * np.sqrt(dt) if sigma_beta_flow > 0 else 0
        raw_dot_beta[i] = k_beta_increase_stress * resource_stress_for_beta - beta_decay_rate * current_beta + beta_noise

        current_F_crit_replenishment = F_crit_replenishment_base_mean + \
            (rng.normal(0, F_crit_replenishment_sigma) if F_crit_replenishment_sigma > 0 else 0)
        effective_F_crit_replenishment = current_F_crit_replenishment
        if current_F_crit < F_crit_low_threshold_abs:
            effective_F_crit_replenishment = current_F_crit_replenishment * \
                (1 + k_F_crit_low_bonus_factor * (1 - current_F_crit / (F_crit_low_threshold_abs + EPSILON)))
        F_crit_costs = (k_F_beta_cost * current_beta + k_F_g_cost * current_g + F_crit_baseline_cost)
        F_crit_noise = rng.normal(0, sigma_F_crit_flow) * np.sqrt(dt) if sigma_F_crit_flow > 0 else 0
        raw_dot_F_crit[i] = effective_F_crit_replenishment - F_crit_costs + F_crit_noise

        g[i+1] = max(g_min, min(g_max, current_g + raw_dot_g[i] * dt))
        beta[i+1] = max(beta_min, min(beta_max, current_beta + raw_dot_beta[i] * dt))
        F_crit[i+1] = max(F_crit_safe_floor, current_F_crit + raw_dot_F_crit[i] * dt)

        current_strain_increase_rate = strain_increase_rate_mean + \
            (rng.normal(0, strain_increase_rate_sigma) if strain_increase_rate_sigma > 0 else 0)
        strain_noise = rng.normal(0, sigma_strain) * np.sqrt(dt) if sigma_strain > 0 else 0
        raw_dot_strain[i] = current_strain_increase_rate + (strain_noise / dt if dt > 0 else 0)
        strain[i+1] = max(0, current_strain + raw_dot_strain[i] * dt)

        g_hist.append(current_g); beta_hist.append(current_beta); F_crit_hist.append(current_F_crit)
        g_obs = g_hist.pop(0) if Y_obs_lag_g > 0 else current_g
        beta_obs = beta_hist.pop(0) if Y_obs_lag_beta > 0 else current_beta
        F_crit_obs = F_crit_hist.pop(0) if Y_obs_lag_F_crit > 0 else current_F_crit

        external_confounder_val = ext_factor_amp_Y * np.sin(2 * np.pi * t / (ext_factor_period_Y + EPSILON))
        y_obs_signal = Y_obs_base_level + \
                       k_Y_g_lin * g_obs + k_Y_beta_lin * beta_obs + k_Y_F_crit_lin * F_crit_obs + \
                       k_Y_g_nl_amp * np.sin(g_obs * 0.2) * max(0, g_obs - g_initial * 0.5) + \
                       k_Y_F_crit_nl_amp * np.tanh((F_crit_obs - F_crit_initial * 0.3) / (F_crit_initial * 0.1 + EPSILON)) + \
                       k_Y_beta_nl_amp * (max(0, beta_obs)**0.8) * np.cos(beta_obs * 0.3) + \
                       k_Y_gb_interact * g_obs * beta_obs * 0.1 * np.cos(t * 0.02) + \
                       k_Y_gF_interact * g_obs * np.sqrt(max(0, F_crit_obs)) * 0.1 * np.sin(t * 0.03) + \
                       k_Y_betaF_interact * beta_obs * F_crit_obs * 0.01 * np.cos(t * 0.04) + \
                       external_confounder_val
        fcrit_denom_for_noise = max(current_F_crit, F_crit_initial * 0.01) + F_crit_initial * 0.05 + EPSILON
        fcrit_effect_on_noise = (F_crit_initial / fcrit_denom_for_noise)**0.5
        current_sigma_Y_obs = sigma_Y_obs_base + sigma_Y_obs_state_dep_additive_scale * fcrit_effect_on_noise
        Y_obs_noise_val = rng.normal(0, current_sigma_Y_obs) if current_sigma_Y_obs > 0 else 0
        Y_obs[i] = y_obs_signal + Y_obs_noise_val

        ews_calc_df.loc[i, 'Y_obs'] = Y_obs[i]
        ews_calc_df.loc[i, 'raw_dot_g'] = raw_dot_g[i]; ews_calc_df.loc[i, 'raw_dot_beta'] = raw_dot_beta[i]; ews_calc_df.loc[i, 'raw_dot_F_crit'] = raw_dot_F_crit[i]
        ews_calc_df.loc[i, 'raw_dot_strain'] = raw_dot_strain[i]

        start_idx_smooth = max(0, i - velocity_smoothing_window + 1)
        dot_g_smooth[i] = ews_calc_df['raw_dot_g'].iloc[start_idx_smooth : i + 1].mean()
        dot_beta_smooth[i] = ews_calc_df['raw_dot_beta'].iloc[start_idx_smooth : i + 1].mean()
        dot_F_crit_smooth[i] = ews_calc_df['raw_dot_F_crit'].iloc[start_idx_smooth : i + 1].mean()
        dot_strain_smooth[i] = ews_calc_df['raw_dot_strain'].iloc[start_idx_smooth : i + 1].mean()
        ews_calc_df.loc[i, 'smooth_dot_g'] = dot_g_smooth[i]; ews_calc_df.loc[i, 'smooth_dot_beta'] = dot_beta_smooth[i]; ews_calc_df.loc[i, 'smooth_dot_F_crit'] = dot_F_crit_smooth[i]

        speed_idx[i] = np.sqrt(dot_beta_smooth[i]**2 + dot_F_crit_smooth[i]**2)
        if i >= couple_window - 1:
            window_data_couple = ews_calc_df.iloc[i - couple_window + 1 : i + 1]
            if not window_data_couple.empty and 'smooth_dot_beta' in window_data_couple and 'smooth_dot_F_crit' in window_data_couple and \
               len(window_data_couple['smooth_dot_beta'].dropna()) > 1 and len(window_data_couple['smooth_dot_F_crit'].dropna()) > 1 and \
               window_data_couple['smooth_dot_beta'].std(ddof=0) > EPSILON and window_data_couple['smooth_dot_F_crit'].std(ddof=0) > EPSILON:
                couple_idx[i] = window_data_couple['smooth_dot_beta'].corr(window_data_couple['smooth_dot_F_crit'])
            elif len(window_data_couple['smooth_dot_beta'].dropna()) <=1 or len(window_data_couple['smooth_dot_F_crit'].dropna()) <=1:
                 couple_idx[i] = np.nan
            else:
                 couple_idx[i] = 0.0 if window_data_couple['smooth_dot_beta'].nunique()==1 or window_data_couple['smooth_dot_F_crit'].nunique()==1 else np.nan

        if i >= trad_ews_window - 1:
            current_Y_pd = ews_calc_df['Y_obs'].iloc[i - trad_ews_window + 1 : i + 1]
            current_ThetaT_pd = ews_calc_df['Theta_T_series'].iloc[i - trad_ews_window + 1 : i + 1]
            if len(current_Y_pd.dropna()) > 1 :
                variance_Y[i] = current_Y_pd.var(ddof=0)
                skewness_Y[i] = current_Y_pd.skew()
                kurtosis_Y[i] = current_Y_pd.kurtosis()
                if current_Y_pd.std(ddof=0) > EPSILON:
                    ar1_Y[i] = current_Y_pd.autocorr(lag=1)
                else:
                    ar1_Y[i] = 1.0 if current_Y_pd.nunique() == 1 else np.nan
            else:
                variance_Y[i] = np.nan; ar1_Y[i] = np.nan; skewness_Y[i] = np.nan; kurtosis_Y[i] = np.nan

            if len(current_ThetaT_pd.dropna()) > 1:
                Variance_Theta_T_vals[i] = current_ThetaT_pd.var(ddof=0)
                if current_ThetaT_pd.std(ddof=0) > EPSILON:
                    AR1_Theta_T_vals[i] = current_ThetaT_pd.autocorr(lag=1)
                else:
                    AR1_Theta_T_vals[i] = 1.0 if current_ThetaT_pd.nunique() == 1 else np.nan
            else:
                Variance_Theta_T_vals[i] = np.nan; AR1_Theta_T_vals[i] = np.nan

        term_g_dttn, term_beta_dttn, term_F_crit_dttn = 0.0, 0.0, 0.0
        if current_g > EPSILON: term_g_dttn = w1 * dot_g_smooth[i] / current_g
        if current_beta > EPSILON: term_beta_dttn = w2 * dot_beta_smooth[i] / current_beta
        if current_F_crit > EPSILON:
            term_F_crit_dttn = w3 * dot_F_crit_smooth[i] / current_F_crit
        elif dot_F_crit_smooth[i] < -EPSILON and current_F_crit <= EPSILON and w3 > 0:
            term_F_crit_dttn = -1e3
        dot_Theta_T_norm_vals[i] = term_g_dttn + term_beta_dttn + term_F_crit_dttn
        ews_calc_df.loc[i, 'dot_Theta_T_norm'] = dot_Theta_T_norm_vals[i]

        if not np.isnan(speed_idx[i]) and not np.isnan(couple_idx[i]):
            H_UHB_vals[i] = (speed_idx[i]**2) / (1.0 - couple_idx[i] + uhb_epsilon_sim)

        term_g_s3d_sq, term_beta_s3d_sq, term_F_crit_s3d_sq = 0.0, 0.0, 0.0
        if current_g > EPSILON: term_g_s3d_sq = (w1 * dot_g_smooth[i] / current_g)**2
        elif abs(dot_g_smooth[i]) > EPSILON : term_g_s3d_sq = (w1 * dot_g_smooth[i] / EPSILON)**2
        if current_beta > EPSILON: term_beta_s3d_sq = (w2 * dot_beta_smooth[i] / current_beta)**2
        elif abs(dot_beta_smooth[i]) > EPSILON : term_beta_s3d_sq = (w2 * dot_beta_smooth[i] / EPSILON)**2
        if current_F_crit > EPSILON: term_F_crit_s3d_sq = (w3 * dot_F_crit_smooth[i] / current_F_crit)**2
        elif abs(dot_F_crit_smooth[i]) > EPSILON : term_F_crit_s3d_sq = (w3 * dot_F_crit_smooth[i] / EPSILON)**2
        S_3D_weighted_vals[i] = np.sqrt(term_g_s3d_sq + term_beta_s3d_sq + term_F_crit_s3d_sq)

        if not np.isnan(dot_Theta_T_norm_vals[i]):
            if Theta_T[i] > EPSILON: dot_G_vals[i] = dot_Theta_T_norm_vals[i] * Theta_T[i]
            elif dot_Theta_T_norm_vals[i] < -EPSILON and Theta_T[i] <= EPSILON: dot_G_vals[i] = -1e3
            else: dot_G_vals[i] = 0.0
        else: dot_G_vals[i] = 0.0

        lci_num = k_g_lci * (max(current_g, EPSILON)**phi_1_lci) + k_beta_lci * (max(current_beta, EPSILON)**phi_beta_lci)
        lci_denom = current_F_crit + lci_F_crit_saturation_offset
        LCI_vals[i] = lci_num / max(lci_denom, EPSILON)
        ews_calc_df.loc[i, 'LCI'] = LCI_vals[i]

        if i >= sigma_dot_theta_window - 1:
            current_dttn_pd = ews_calc_df['dot_Theta_T_norm'].iloc[i - sigma_dot_theta_window + 1 : i + 1]
            if len(current_dttn_pd.dropna()) > 1:
                Sigma_dot_Theta_T_norm_vals[i] = current_dttn_pd.std(ddof=0)
            else:
                Sigma_dot_Theta_T_norm_vals[i] = np.nan

        if i > 0: raw_dot_LCI_series[i] = (LCI_vals[i] - LCI_vals[i-1]) / dt if dt > EPSILON else 0.0
        else: raw_dot_LCI_series[i] = 0.0
        ews_calc_df.loc[i, 'raw_dot_LCI'] = raw_dot_LCI_series[i]
        start_idx_dot_LCI_smooth = max(0, i - velocity_smoothing_window + 1)
        dot_LCI_S_vals[i] = ews_calc_df['raw_dot_LCI'].iloc[start_idx_dot_LCI_smooth : i + 1].mean()

        if not np.isnan(dot_F_crit_smooth[i]):
            if dot_F_crit_smooth[i] < 0: Fcrit_DepVelN_vals[i] = (-1 * dot_F_crit_smooth[i]) / (F_crit_initial + EPSILON)
            else: Fcrit_DepVelN_vals[i] = 0.0
        else: Fcrit_DepVelN_vals[i] = np.nan

        if not np.isnan(dot_F_crit_smooth[i]) and not np.isnan(LCI_vals[i]):
            if dot_F_crit_smooth[i] < -EPSILON:
                time_to_floor_val = (current_F_crit - F_crit_safe_floor) / (-dot_F_crit_smooth[i] + EPSILON)
                time_to_floor_val = max(0, time_to_floor_val)
                PTTF_LCI_vals[i] = LCI_vals[i] / (time_to_floor_val + proj_time_denom_offset)
            else: PTTF_LCI_vals[i] = 0.0
        else: PTTF_LCI_vals[i] = np.nan

        if not np.isnan(Theta_T[i]): RelCostThetaT_vals[i] = lci_num / (Theta_T[i] + EPSILON)
        else: RelCostThetaT_vals[i] = np.nan

        net_flow_for_deficit = raw_dot_F_crit[i]
        deficit_increment = max(0, -1 * net_flow_for_deficit) * dt
        accumulated_F_crit_deficit_tracker += deficit_increment
        Fcrit_AccDef_vals[i] = accumulated_F_crit_deficit_tracker

        Theta_T_Margin_current = Theta_T[i] - strain[i]
        if i > 0: raw_dot_Theta_T_Margin_series[i] = (Theta_T_Margin_current - (Theta_T[i-1] - strain[i-1])) / dt if dt > EPSILON else 0.0
        else: raw_dot_Theta_T_Margin_series[i] = 0.0
        ews_calc_df.loc[i, 'raw_dot_Theta_T_Margin'] = raw_dot_Theta_T_Margin_series[i]
        dot_Theta_T_Margin_smooth = ews_calc_df['raw_dot_Theta_T_Margin'].iloc[max(0, i - velocity_smoothing_window + 1) : i+1].mean()
        if (Theta_T_Margin_current + EPSILON) != 0:
             TMCRN_vals[i] = dot_Theta_T_Margin_smooth / (Theta_T_Margin_current + EPSILON)
        else: TMCRN_vals[i] = np.nan if dot_Theta_T_Margin_smooth == 0 else -np.inf * np.sign(dot_Theta_T_Margin_smooth)

        F_crit_Cost_of_Levers = k_F_g_cost * current_g + k_F_beta_cost * current_beta
        RMA_norm_vals[i] = (current_F_crit - F_crit_Cost_of_Levers) / (F_crit_initial + EPSILON)

        Adaptation_F_crit_Drain_Rate = k_F_g_cost * current_g + k_F_beta_cost * current_beta
        FACR_vals[i] = Adaptation_F_crit_Drain_Rate / (max(current_F_crit_replenishment, EPSILON) if params.get('FACR_denom_replenish', True) else (current_F_crit + EPSILON))

        contrib_g = w1 * dot_g_smooth[i] / (current_g + EPSILON) if not np.isnan(dot_g_smooth[i]) else np.nan
        contrib_beta = w2 * dot_beta_smooth[i] / (current_beta + EPSILON) if not np.isnan(dot_beta_smooth[i]) else np.nan
        contrib_F_crit_val = w3 * dot_F_crit_smooth[i] / (current_F_crit + EPSILON) if not np.isnan(dot_F_crit_smooth[i]) else np.nan
        contribs = [c for c in [contrib_g, contrib_beta, contrib_F_crit_val] if not np.isnan(c)]
        if len(contribs) > 1: CCD_vals[i] = np.std(contribs, ddof=0)
        else: CCD_vals[i] = 0.0

        if current_F_crit <= (F_crit_safe_floor + EPSILON):
            breached = True; breach_time = t; breach_step_idx = i; breach_type = "F_crit_depletion"
        elif strain[i+1] > Theta_T[i] and Theta_T[i] > EPSILON:
            breached = True; breach_time = t; breach_step_idx = i; breach_type = "Strain_exceeds_Theta_T"
        if current_beta >= beta_max_critical_level:
            beta_above_crit_counter += 1
            if beta_above_crit_counter >= beta_max_critical_duration_steps:
                breached = True; breach_time = t; breach_step_idx = i; breach_type = "Beta_over_critical"
        else:
            beta_above_crit_counter = 0

        if breached:
            final_idx = i + 1
            arrays_to_truncate = [
                't_series', 'g', 'beta', 'F_crit', 'strain', 'raw_dot_strain', 'dot_strain_smooth',
                'raw_dot_g', 'raw_dot_beta', 'raw_dot_F_crit',
                'dot_g_smooth', 'dot_beta_smooth', 'dot_F_crit_smooth',
                'Theta_T', 'Y_obs', 'speed_idx', 'couple_idx', 'variance_Y', 'ar1_Y', 'skewness_Y', 'kurtosis_Y',
                'dot_Theta_T_norm_vals', 'H_UHB_vals', 'S_3D_weighted_vals',
                'dot_G_vals', 'LCI_vals', 'Sigma_dot_Theta_T_norm_vals',
                'dot_LCI_S_vals', 'Fcrit_DepVelN_vals', 'PTTF_LCI_vals',
                'RelCostThetaT_vals', 'Fcrit_AccDef_vals',
                'TMCRN_vals', 'Variance_Theta_T_vals', 'AR1_Theta_T_vals',
                'RMA_norm_vals', 'FACR_vals', 'CCD_vals',
                'raw_dot_LCI_series', 'raw_dot_Theta_T_Margin_series'
            ]
            for arr_name_str in arrays_to_truncate:
                locals()[arr_name_str] = locals()[arr_name_str][:final_idx]
            ews_calc_df = ews_calc_df.iloc[:final_idx].copy()
            break

    last_calc_idx = len(t_series) - 1
    if last_calc_idx >= 0:
        i = last_calc_idx
        current_g = g[i]; current_beta = beta[i]; current_F_crit = F_crit[i]; current_strain = strain[i]
        Theta_T[i] = calculate_tolerance_sheet(current_g, current_beta, current_F_crit, w1, w2, w3, C_tolerance)
        ews_calc_df.loc[i, 'Theta_T_series'] = Theta_T[i]

        if i > 0:
            raw_dot_g[i] = raw_dot_g[i-1]; raw_dot_beta[i] = raw_dot_beta[i-1]; raw_dot_F_crit[i] = raw_dot_F_crit[i-1]; raw_dot_strain[i] = raw_dot_strain[i-1]
        else:
            raw_dot_g[i]=0; raw_dot_beta[i]=0; raw_dot_F_crit[i]=0; raw_dot_strain[i]=0

        ews_calc_df.loc[i, 'raw_dot_g'] = raw_dot_g[i]; ews_calc_df.loc[i, 'raw_dot_beta'] = raw_dot_beta[i]
        ews_calc_df.loc[i, 'raw_dot_F_crit'] = raw_dot_F_crit[i]; ews_calc_df.loc[i, 'raw_dot_strain'] = raw_dot_strain[i]

        start_idx_smooth = max(0, i - velocity_smoothing_window + 1)
        dot_g_smooth[i] = ews_calc_df['raw_dot_g'].iloc[start_idx_smooth : i + 1].mean()
        dot_beta_smooth[i] = ews_calc_df['raw_dot_beta'].iloc[start_idx_smooth : i + 1].mean()
        dot_F_crit_smooth[i] = ews_calc_df['raw_dot_F_crit'].iloc[start_idx_smooth : i + 1].mean()
        dot_strain_smooth[i] = ews_calc_df['raw_dot_strain'].iloc[start_idx_smooth : i + 1].mean()
        ews_calc_df.loc[i, 'smooth_dot_g'] = dot_g_smooth[i]; ews_calc_df.loc[i, 'smooth_dot_beta'] = dot_beta_smooth[i]; ews_calc_df.loc[i, 'smooth_dot_F_crit'] = dot_F_crit_smooth[i]

        g_obs = g_hist[-1] if Y_obs_lag_g > 0 and g_hist else current_g
        beta_obs = beta_hist[-1] if Y_obs_lag_beta > 0 and beta_hist else current_beta
        F_crit_obs = F_crit_hist[-1] if Y_obs_lag_F_crit > 0 and F_crit_hist else current_F_crit
        external_confounder_val = ext_factor_amp_Y * np.sin(2 * np.pi * t_series[i] / (ext_factor_period_Y + EPSILON))
        y_obs_signal = Y_obs_base_level + k_Y_g_lin * g_obs + k_Y_beta_lin * beta_obs + k_Y_F_crit_lin * F_crit_obs + \
                       k_Y_g_nl_amp * np.sin(g_obs * 0.2) * max(0,g_obs-g_initial*0.5) + \
                       k_Y_F_crit_nl_amp * np.tanh((F_crit_obs - F_crit_initial*0.3)/(F_crit_initial*0.1+EPSILON)) + \
                       k_Y_beta_nl_amp * (max(0,beta_obs)**0.8) * np.cos(beta_obs * 0.3) + \
                       k_Y_gb_interact * g_obs * beta_obs * 0.1 * np.cos(t_series[i]*0.02) + \
                       k_Y_gF_interact * g_obs * np.sqrt(max(0,F_crit_obs)) * 0.1 * np.sin(t_series[i]*0.03) + \
                       k_Y_betaF_interact * beta_obs * F_crit_obs * 0.01 * np.cos(t_series[i]*0.04) + \
                       external_confounder_val
        fcrit_denom_for_noise = max(current_F_crit, F_crit_initial * 0.01) + F_crit_initial * 0.05 + EPSILON
        fcrit_effect_on_noise = (F_crit_initial / fcrit_denom_for_noise)**0.5
        current_sigma_Y_obs = sigma_Y_obs_base + sigma_Y_obs_state_dep_additive_scale * fcrit_effect_on_noise
        Y_obs_noise_val = rng.normal(0, current_sigma_Y_obs) if current_sigma_Y_obs > 0 else 0
        Y_obs[i] = y_obs_signal + Y_obs_noise_val
        ews_calc_df.loc[i, 'Y_obs'] = Y_obs[i]

        speed_idx[i] = np.sqrt(dot_beta_smooth[i]**2 + dot_F_crit_smooth[i]**2)
        if i >= couple_window - 1:
            window_data_couple = ews_calc_df.iloc[i - couple_window + 1 : i + 1]
            if not window_data_couple.empty and 'smooth_dot_beta' in window_data_couple and 'smooth_dot_F_crit' in window_data_couple and \
               len(window_data_couple['smooth_dot_beta'].dropna()) > 1 and len(window_data_couple['smooth_dot_F_crit'].dropna()) > 1 and \
               window_data_couple['smooth_dot_beta'].std(ddof=0) > EPSILON and window_data_couple['smooth_dot_F_crit'].std(ddof=0) > EPSILON:
                couple_idx[i] = window_data_couple['smooth_dot_beta'].corr(window_data_couple['smooth_dot_F_crit'])
            elif len(window_data_couple['smooth_dot_beta'].dropna()) <=1 or len(window_data_couple['smooth_dot_F_crit'].dropna()) <=1 : couple_idx[i] = np.nan
            else: couple_idx[i] = 0.0 if window_data_couple['smooth_dot_beta'].nunique()==1 or window_data_couple['smooth_dot_F_crit'].nunique()==1 else np.nan

        if i >= trad_ews_window - 1:
            current_Y_pd = ews_calc_df['Y_obs'].iloc[i - trad_ews_window + 1 : i + 1]
            current_ThetaT_pd = ews_calc_df['Theta_T_series'].iloc[i - trad_ews_window + 1 : i + 1]
            if len(current_Y_pd.dropna()) > 1 :
                variance_Y[i] = current_Y_pd.var(ddof=0); skewness_Y[i] = current_Y_pd.skew(); kurtosis_Y[i] = current_Y_pd.kurtosis()
                if current_Y_pd.std(ddof=0) > EPSILON: ar1_Y[i] = current_Y_pd.autocorr(lag=1)
                else: ar1_Y[i] = 1.0 if current_Y_pd.nunique() == 1 else np.nan
            else: variance_Y[i] = np.nan; ar1_Y[i] = np.nan; skewness_Y[i] = np.nan; kurtosis_Y[i] = np.nan
            if len(current_ThetaT_pd.dropna()) > 1:
                Variance_Theta_T_vals[i] = current_ThetaT_pd.var(ddof=0)
                if current_ThetaT_pd.std(ddof=0) > EPSILON: AR1_Theta_T_vals[i] = current_ThetaT_pd.autocorr(lag=1)
                else: AR1_Theta_T_vals[i] = 1.0 if current_ThetaT_pd.nunique() == 1 else np.nan
            else: Variance_Theta_T_vals[i] = np.nan; AR1_Theta_T_vals[i] = np.nan

        term_g_dttn, term_beta_dttn, term_F_crit_dttn = 0.0, 0.0, 0.0
        if current_g > EPSILON: term_g_dttn = w1 * dot_g_smooth[i] / current_g
        if current_beta > EPSILON: term_beta_dttn = w2 * dot_beta_smooth[i] / current_beta
        if current_F_crit > EPSILON: term_F_crit_dttn = w3 * dot_F_crit_smooth[i] / current_F_crit
        elif dot_F_crit_smooth[i] < -EPSILON and current_F_crit <= EPSILON and w3 > 0: term_F_crit_dttn = -1e3
        dot_Theta_T_norm_vals[i] = term_g_dttn + term_beta_dttn + term_F_crit_dttn
        ews_calc_df.loc[i, 'dot_Theta_T_norm'] = dot_Theta_T_norm_vals[i]

        if not np.isnan(speed_idx[i]) and not np.isnan(couple_idx[i]): H_UHB_vals[i] = (speed_idx[i]**2) / (1.0 - couple_idx[i] + uhb_epsilon_sim)

        term_g_s3d_sq, term_beta_s3d_sq, term_F_crit_s3d_sq = 0.0, 0.0, 0.0
        if current_g > EPSILON: term_g_s3d_sq = (w1 * dot_g_smooth[i] / current_g)**2
        elif abs(dot_g_smooth[i]) > EPSILON : term_g_s3d_sq = (w1 * dot_g_smooth[i] / EPSILON)**2
        if current_beta > EPSILON: term_beta_s3d_sq = (w2 * dot_beta_smooth[i] / current_beta)**2
        elif abs(dot_beta_smooth[i]) > EPSILON : term_beta_s3d_sq = (w2 * dot_beta_smooth[i] / EPSILON)**2
        if current_F_crit > EPSILON: term_F_crit_s3d_sq = (w3 * dot_F_crit_smooth[i] / current_F_crit)**2
        elif abs(dot_F_crit_smooth[i]) > EPSILON : term_F_crit_s3d_sq = (w3 * dot_F_crit_smooth[i] / EPSILON)**2
        S_3D_weighted_vals[i] = np.sqrt(term_g_s3d_sq + term_beta_s3d_sq + term_F_crit_s3d_sq)

        lci_num = k_g_lci * (max(current_g, EPSILON)**phi_1_lci) + k_beta_lci * (max(current_beta, EPSILON)**phi_beta_lci)
        lci_denom = current_F_crit + lci_F_crit_saturation_offset
        LCI_vals[i] = lci_num / max(lci_denom, EPSILON)
        ews_calc_df.loc[i, 'LCI'] = LCI_vals[i]

        if i > 0: raw_dot_LCI_series[i] = (LCI_vals[i] - LCI_vals[i-1]) / dt if dt > EPSILON else 0.0
        else: raw_dot_LCI_series[i] = 0.0
        ews_calc_df.loc[i, 'raw_dot_LCI'] = raw_dot_LCI_series[i]
        start_idx_dot_LCI_smooth = max(0, i - velocity_smoothing_window + 1)
        dot_LCI_S_vals[i] = ews_calc_df['raw_dot_LCI'].iloc[start_idx_dot_LCI_smooth : i + 1].mean()

        if not np.isnan(dot_F_crit_smooth[i]):
            if dot_F_crit_smooth[i] < 0: Fcrit_DepVelN_vals[i] = (-1 * dot_F_crit_smooth[i]) / (F_crit_initial + EPSILON)
            else: Fcrit_DepVelN_vals[i] = 0.0
        else: Fcrit_DepVelN_vals[i] = np.nan
        if not np.isnan(dot_F_crit_smooth[i]) and not np.isnan(LCI_vals[i]):
            if dot_F_crit_smooth[i] < -EPSILON:
                time_to_floor_val = (current_F_crit - F_crit_safe_floor) / (-dot_F_crit_smooth[i] + EPSILON)
                time_to_floor_val = max(0, time_to_floor_val)
                PTTF_LCI_vals[i] = LCI_vals[i] / (time_to_floor_val + proj_time_denom_offset)
            else: PTTF_LCI_vals[i] = 0.0
        else: PTTF_LCI_vals[i] = np.nan
        if not np.isnan(Theta_T[i]): RelCostThetaT_vals[i] = lci_num / (Theta_T[i] + EPSILON)
        else: RelCostThetaT_vals[i] = np.nan
        
        net_flow_for_deficit = raw_dot_F_crit[i]
        deficit_increment = max(0, -1 * net_flow_for_deficit) * dt
        prev_deficit = Fcrit_AccDef_vals[i-1] if i > 0 and not np.isnan(Fcrit_AccDef_vals[i-1]) else 0.0
        Fcrit_AccDef_vals[i] = prev_deficit + deficit_increment

        Theta_T_Margin_current = Theta_T[i] - strain[i]
        if i > 0: raw_dot_Theta_T_Margin_series[i] = (Theta_T_Margin_current - (Theta_T[i-1] - strain[i-1])) / dt if dt > EPSILON else 0.0
        else: raw_dot_Theta_T_Margin_series[i] = 0.0
        ews_calc_df.loc[i, 'raw_dot_Theta_T_Margin'] = raw_dot_Theta_T_Margin_series[i]
        dot_Theta_T_Margin_smooth = ews_calc_df['raw_dot_Theta_T_Margin'].iloc[max(0, i - velocity_smoothing_window + 1) : i+1].mean()
        if (Theta_T_Margin_current + EPSILON) != 0: TMCRN_vals[i] = dot_Theta_T_Margin_smooth / (Theta_T_Margin_current + EPSILON)
        else: TMCRN_vals[i] = np.nan if dot_Theta_T_Margin_smooth == 0 else -np.inf * np.sign(dot_Theta_T_Margin_smooth)

        F_crit_Cost_of_Levers = k_F_g_cost * current_g + k_F_beta_cost * current_beta
        RMA_norm_vals[i] = (current_F_crit - F_crit_Cost_of_Levers) / (F_crit_initial + EPSILON)
        Adaptation_F_crit_Drain_Rate = k_F_g_cost * current_g + k_F_beta_cost * current_beta
        FACR_vals[i] = Adaptation_F_crit_Drain_Rate / (max(F_crit_replenishment_base_mean, EPSILON) if params.get('FACR_denom_replenish', True) else (current_F_crit + EPSILON))

        contrib_g = w1 * dot_g_smooth[i] / (current_g + EPSILON) if not np.isnan(dot_g_smooth[i]) else np.nan
        contrib_beta = w2 * dot_beta_smooth[i] / (current_beta + EPSILON) if not np.isnan(dot_beta_smooth[i]) else np.nan
        contrib_F_crit_val = w3 * dot_F_crit_smooth[i] / (current_F_crit + EPSILON) if not np.isnan(dot_F_crit_smooth[i]) else np.nan
        contribs = [c for c in [contrib_g, contrib_beta, contrib_F_crit_val] if not np.isnan(c)]
        if len(contribs) > 1: CCD_vals[i] = np.std(contribs, ddof=0)
        else: CCD_vals[i] = 0.0

    return {
        'run_id': run_id, 't': t_series, 'g': g, 'beta': beta, 'F_crit': F_crit, 'Strain': strain,
        'raw_dot_g': raw_dot_g, 'raw_dot_beta': raw_dot_beta, 'raw_dot_F_crit': raw_dot_F_crit,
        'dot_g': dot_g_smooth, 'dot_beta': dot_beta_smooth, 'dot_F_crit': dot_F_crit_smooth,
        'Theta_T': Theta_T, 'Y_obs': Y_obs,
        'Speed': speed_idx, 'Couple': couple_idx,
        'Variance_Y': variance_Y, 'AR1_Y': ar1_Y, 'Skewness_Y': skewness_Y, 'Kurtosis_Y': kurtosis_Y,
        'dot_Theta_T_norm': dot_Theta_T_norm_vals, 'H_UHB': H_UHB_vals,
        'S_3D_weighted': S_3D_weighted_vals,
        'LCI': LCI_vals,
        'dot_LCI_S': dot_LCI_S_vals,
        'Fcrit_DepVelN': Fcrit_DepVelN_vals,
        'PTTF_LCI_Ctx': PTTF_LCI_vals,
        'Rel_Cost_ThetaT': RelCostThetaT_vals,
        'Fcrit_Acc_Deficit': Fcrit_AccDef_vals,
        'TMCRN': TMCRN_vals,
        'Variance_Theta_T': Variance_Theta_T_vals,
        'AR1_Theta_T': AR1_Theta_T_vals,
        'RMA_norm': RMA_norm_vals,
        'FACR': FACR_vals,
        'CCD': CCD_vals,
        'breached': breached, 'breach_time': breach_time, 'breach_step_idx': breach_step_idx,
        'breach_type': breach_type, 'params': params,
        'original_label': params.get('original_label', -1)
    }


# --- Parameter Configurations ---
base_config = {
    'max_steps': 3000, 'dt': 0.1, 'w': (1/3, 1/3, 1/3), 'C_tolerance': 1.0,
    'g_initial': 10.0, 'beta_initial': 2.5, 'F_crit_initial': 20.0,
    'g_min': 0.2, 'g_max': 35.0,
    'k_g_strain_response': 0.06, 'g_target_stress_ratio': 0.75,
    'g_decay_rate_lever': 0.015, 'g_cost_F_crit_scale_denom': 0.25,
    'sigma_g_flow': 0.3, 'sigma_beta_flow': 0.3, 'sigma_F_crit_flow': 0.4,
    'sigma_Y_obs_base': 3.5,
    'sigma_Y_obs_state_dep_additive_scale': 0.25,
    'sigma_strain': 0.07,
    'couple_window': 25, 'trad_ews_window': 35, 'velocity_smoothing_window': 18,
    'sigma_dot_theta_window': 35,
    'Y_obs_base_level': 2.0, 'k_Y_g_lin': 0.001, 'k_Y_beta_lin': -0.001, 'k_Y_F_crit_lin': 0.002,
    'k_Y_g_nl_amp': 0.03, 'k_Y_beta_nl_amp': 0.06, 'k_Y_F_crit_nl_amp': 0.15,
    'k_Y_gb_interact': 0.00001,'k_Y_gF_interact': 0.000005,'k_Y_betaF_interact': -0.00001,
    'ext_factor_amp_Y': 2.0, 'ext_factor_period_Y_frac': 0.15,
    'Y_obs_lag_g': 5, 'Y_obs_lag_beta': 3, 'Y_obs_lag_F_crit': 8,
    'F_crit_target_beta': 8.0, 'k_beta_increase_stress': 0.06, 'beta_decay_rate': 0.015,
    'beta_min': 0.15, 'beta_max': 18.0, 'beta_max_critical_level': 17.0, 'beta_max_critical_duration_steps': 30,
    'F_crit_replenishment_base_mean': 0.20, 'F_crit_replenishment_sigma': 0.05,
    'F_crit_low_threshold_factor': 0.1, 'k_F_crit_low_bonus_factor': 5.0,
    'F_crit_safe_floor': 0.05,
    'k_F_g_cost': 0.03, 'k_F_beta_cost': 0.03, 'F_crit_baseline_cost': 0.02,
    'phi_1_lci': 0.65,
    'phi_beta_lci': 0.85,
    'k_g_lci': 0.03, 'k_beta_lci': 0.03,
    'uhb_epsilon': EPSILON, # used for sim
    'lci_F_crit_saturation_offset': 0.75,
    'proj_time_denom_offset': 1.0,
    'FACR_denom_replenish': True,
}
params_stable_config = {**base_config, 'original_label': 0, 'run_type': 'stable', 'initial_strain': 1.0, 'strain_increase_rate_mean': 0.0, 'strain_increase_rate_sigma': 0.0005,'F_crit_initial': 30.0, 'F_crit_replenishment_base_mean': 0.6, 'F_crit_replenishment_sigma': 0.1,'k_beta_increase_stress': 0.005, 'k_F_beta_cost': 0.005, 'F_crit_baseline_cost': 0.005,'k_F_g_cost': 0.002, 'k_g_strain_response': 0.005,'sigma_g_flow': base_config['sigma_g_flow'] * 0.7,'sigma_beta_flow': base_config['sigma_beta_flow'] * 0.7,'sigma_F_crit_flow': base_config['sigma_F_crit_flow'] * 0.7,'ext_factor_amp_Y': 0.5,'stable_apply_sub_shocks': True,'stable_shock_prob': 0.04, 'stable_shock_F_crit_max_delta_frac': 0.15, 'stable_shock_strain_max_delta': 0.25, }
params_collapse_original = {**base_config, 'original_label': 1, 'run_type': 'original_collapse', 'F_crit_initial': 18.0,'initial_strain': 2.0, 'strain_increase_rate_mean': 0.001, 'strain_increase_rate_sigma': 0.0005,'F_crit_replenishment_base_mean': 0.005, 'F_crit_replenishment_sigma': 0.002,}
params_collapse_shock = {**base_config, 'original_label': 1, 'run_type': 'shock_collapse', 'F_crit_initial': 25.0, 'initial_strain': 1.8, 'strain_increase_rate_mean': 0.00001, 'F_crit_replenishment_base_mean': 0.1,'apply_shock': True, 'shock_time': base_config['max_steps'] * base_config['dt'] * 0.7,'shock_F_crit_delta': -22.0, 'shock_strain_delta': 2.5, }
params_collapse_resource = {**base_config, 'original_label': 1, 'run_type': 'resource_collapse', 'F_crit_initial': 12.0, 'initial_strain': 2.0, 'strain_increase_rate_mean': 0.0002,'F_crit_replenishment_base_mean': 0.02, 'F_crit_replenishment_sigma': 0.015,'k_F_g_cost': 0.06, 'k_F_beta_cost': 0.06, 'F_crit_baseline_cost': 0.12, }
params_collapse_adaptation = {**base_config, 'original_label': 1, 'run_type':
                              'adaptation_collapse', 'F_crit_initial': 28.0,'initial_strain': 1.5,
                              'strain_increase_rate_mean': 0.0010, 'F_crit_replenishment_base_mean': 0.04,
                              'k_g_strain_response': 0.0003, 'g_decay_rate_lever': 0.15, 'sigma_g_flow': 0.5,
                              'k_beta_increase_stress': 0.0002, 'beta_decay_rate': 0.10, 'sigma_beta_flow': 0.5,}
params_collapse_lever_instability = {
    **base_config, 'original_label': 1, 'run_type': 'lever_instability_collapse',
    'F_crit_initial': 18.0, 'F_crit_replenishment_base_mean': 0.15,
    'sigma_g_flow': 0.9, 'sigma_beta_flow': 0.9,
    'F_crit_baseline_cost' : 0.05,
    'k_g_strain_response': 0.04, 'k_beta_increase_stress': 0.12,
    'g_decay_rate_lever': 0.020, 'beta_decay_rate': 0.020,
    'initial_strain': 3.0, 'strain_increase_rate_mean': 0.001,
    'k_F_g_cost': 0.06, 'k_F_beta_cost': 0.06, 'w': (0.5, 0.4, 0.1),
}
params_collapse_strain_shock = {
    **base_config, 'original_label': 1, 'run_type': 'massive_strain_shock_collapse',
    'F_crit_initial': 30.0, 'F_crit_replenishment_base_mean': 0.3,
    'initial_strain': 1.0, 'strain_increase_rate_mean': 0.00001,
    'apply_shock': True, 'shock_time': base_config['max_steps'] * base_config['dt'] * 0.4,
    'shock_strain_delta': +15.0, 'shock_F_crit_delta': 0.0,
}
params_collapse_g_catastrophe = {
    **base_config, 'original_label': 1, 'run_type': 'g_catastrophe_collapse',
    'F_crit_initial': 50.0,
    'F_crit_replenishment_base_mean': 1.0,
    'k_F_g_cost': 0.001, 'k_F_beta_cost': 0.001, 'F_crit_baseline_cost': 0.001,
    'initial_strain': 3.0,
    'strain_increase_rate_mean': 0.0,
    'w': (0.6, 0.2, 0.2),
    'apply_shock': True,
    'shock_time': base_config['max_steps'] * base_config['dt'] * 0.5,
    'shock_g_delta': - (base_config['g_initial'] - base_config['g_min'] - EPSILON*10),
    'shock_strain_delta': 0.0,
    'shock_F_crit_delta': 0.0,
}
params_collapse_beta_destruct = {
    **base_config, 'original_label': 1, 'run_type': 'beta_destruct_collapse',
    'F_crit_initial': 50.0,
    'F_crit_replenishment_base_mean': 2.0,
    'k_F_beta_cost': 0.0001,
    'k_F_g_cost': 0.0001,
    'F_crit_baseline_cost': 0.0001,
    'beta_initial': 0.2,
    'beta_min': 0.1,
    'beta_max': 60.0,
    'F_crit_target_beta': 1000.0,
    'k_beta_increase_stress': 1.5,
    'beta_decay_rate': 0.000001,
    'beta_max_critical_level': 55.0,
    'beta_max_critical_duration_steps': 2,
    'initial_strain': 0.1,
    'strain_increase_rate_mean': 0.0,
    'w': (0.4, 0.01, 0.59),
    'g_initial': 10.0,
    'k_g_strain_response': 0.001,
    'sigma_beta_flow': 0.01,
}

# --- HELPER FUNCTIONS ---
def get_ews_aggregates_from_horizon(run_data: Dict[str, Any], ews_name_func: str,
                                    is_actually_breached_func: bool, breach_step_idx_func: int,
                                    warning_horizon_fraction_func: float,
                                    agg_perc_rising_func: float, agg_perc_falling_func: float,
                                    min_window_fields_func: List[str],
                                    params_func: Dict[str, Any]
                                    ) -> Dict[str, float]:
    signal_array = run_data.get(ews_name_func)
    if signal_array is None: return {'rising': np.nan, 'falling': np.nan}

    full_signal_func = np.array(signal_array)
    effective_end_idx_for_signal = breach_step_idx_func if is_actually_breached_func and breach_step_idx_func >= 0 else len(full_signal_func)
    full_signal_func = full_signal_func[:effective_end_idx_for_signal]

    if len(full_signal_func) == 0: return {'rising': np.nan, 'falling': np.nan}

    current_min_windows_list_func = [params_func.get(field, 1) for field in min_window_fields_func if field in params_func]
    min_total_calc_window_func = max(current_min_windows_list_func) if current_min_windows_list_func else 1

    # Adjust min_total_calc_window based on EWS type
    if ews_name_func in ['dot_LCI_S', 'ddot_Theta_T_norm', 'TMCRN']:
        min_total_calc_window_func = max(min_total_calc_window_func, params_func.get('velocity_smoothing_window', 1))
    elif ews_name_func in ['Couple']:
         min_total_calc_window_func = max(min_total_calc_window_func, params_func.get('couple_window',1))
    elif ews_name_func in ['Sigma_dot_Theta_T_norm']:
         min_total_calc_window_func = max(min_total_calc_window_func, params_func.get('sigma_dot_theta_window',1))
    elif ews_name_func in ['Variance_Y', 'AR1_Y', 'Skewness_Y', 'Kurtosis_Y', 'Variance_Theta_T', 'AR1_Theta_T']:
        min_total_calc_window_func = max(min_total_calc_window_func, params_func.get('trad_ews_window', 1))

    if len(full_signal_func) < min_total_calc_window_func: return {'rising': np.nan, 'falling': np.nan}
    valid_ews_start_idx_func = min_total_calc_window_func - 1
    if valid_ews_start_idx_func >= len(full_signal_func): return {'rising': np.nan, 'falling': np.nan}

    ews_signal_part_func = full_signal_func[valid_ews_start_idx_func:]
    if len(ews_signal_part_func) == 0: return {'rising': np.nan, 'falling': np.nan}

    horizon_len_steps_func = int(len(ews_signal_part_func) * warning_horizon_fraction_func)
    if horizon_len_steps_func == 0 and len(ews_signal_part_func) > 0: horizon_len_steps_func = 1

    signal_in_horizon_func = ews_signal_part_func[-horizon_len_steps_func:] if horizon_len_steps_func > 0 else ews_signal_part_func
    finite_signal_in_horizon_func = signal_in_horizon_func[np.isfinite(signal_in_horizon_func)]

    if len(finite_signal_in_horizon_func) == 0: return {'rising': np.nan, 'falling': np.nan}
    rising_agg = np.percentile(finite_signal_in_horizon_func, agg_perc_rising_func)
    falling_agg = np.percentile(finite_signal_in_horizon_func, agg_perc_falling_func)
    return {'rising': rising_agg, 'falling': falling_agg}

def scale_instant_value(value: float, p5: float, p95: float, is_rising_interpretation: bool) -> float:
    """Scales an instantaneous EWS value to [0, 1] where 1 means stronger warning."""
    if np.isnan(value) or abs(p95 - p5) < EPSILON:
        return np.nan
    if is_rising_interpretation: # Higher raw value means warning
        scaled = (value - p5) / (p95 - p5)
    else: # Lower raw value means warning (falling signal)
        scaled = (p95 - value) / (p95 - p5) # Invert so higher scaled value means warning
    return np.clip(scaled, 0, 1)

# --- Ensemble Analysis & Main Execution ---
def perform_ensemble_analysis(
    all_collapse_runs_input: List[Dict[str, Any]],
    all_stable_runs_input: List[Dict[str, Any]],
    title_suffix: str = "",
    warning_horizon_fraction: float = DEFAULT_WARNING_HORIZON_FRACTION,
    agg_perc_rising: float = DEFAULT_AGGREGATION_PERCENTILE_RISING,
    agg_perc_falling: float = DEFAULT_AGGREGATION_PERCENTILE_FALLING,
    k_folds: int = DEFAULT_K_FOLDS,
    k_fold_shuffle: bool = DEFAULT_SHUFFLE_K_FOLD,
    k_fold_seed: int = DEFAULT_K_FOLD_SEED,
    speed_exp_percentile: float = DEFAULT_SPEED_EXP_PERCENTILE,
    couple_exp_neg_percentile: float = DEFAULT_COUPLE_EXP_NEG_PERCENTILE,
    uhb_indicator_threshold_percentile: float = DEFAULT_UHB_INDICATOR_THRESHOLD_PERCENTILE,
    results_dir: str = RESULTS_DIR
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    print(f"\n--- Performing K-Fold CV Ensemble Analysis {title_suffix} ({k_folds} folds) ---")
    print(f"Horizon: {warning_horizon_fraction*100}%, Aggregation Rising: {agg_perc_rising}th, Falling: {agg_perc_falling}th percentile")
    uhb_epsilon_analysis = EPSILON # Used in H_UHB and derived continuous

    ews_signals_info = {
        'Speed': {}, 'Couple': {}, 'Variance_Y': {}, 'AR1_Y': {}, 'Skewness_Y': {}, 'Kurtosis_Y': {},
        'dot_Theta_T_norm': {}, 'H_UHB': {}, 'S_3D_weighted': {}, 'LCI': {},
        'TD_Experimental_Point': {'is_point_metric': True},
        'TD_Speed_Minus_Couple_Continuous': {'natural_type': 'rising'},
        'UHB_Indicator_Continuous': {'natural_type': 'rising'},
        'UHB_Point_Indicator': {'is_point_metric': True},
        'dot_LCI_S': {}, 'Fcrit_DepVelN': {}, 'PTTF_LCI_Ctx': {}, 'Rel_Cost_ThetaT': {}, 'Fcrit_Acc_Deficit': {},
        'TMCRN': {}, 'Variance_Theta_T': {}, 'AR1_Theta_T': {}, 'RMA_norm': {}, 'FACR': {}, 'CCD': {},
        ENSEMBLE_EWS_NAME: {'natural_type': 'rising'}, # Ensemble is designed to be rising
    }

    final_roc_results = {ews: {'auc_mean': np.nan, 'auc_std': np.nan,
                               'fpr_pts_fold0': np.array([0.,1.]),
                               'tpr_pts_fold0': np.array([0.,1.]),
                               'chosen_interpretation_fold0': 'rising'}
                         for ews in ews_signals_info if not ews_signals_info[ews].get('is_point_metric')}
    lead_time_summary: Dict[str, Dict[str, Any]] = {}
    for ews_name_setup, info_setup in ews_signals_info.items():
        if info_setup.get('is_point_metric', False):
            final_roc_results[ews_name_setup] = {
                'tpr_mean': np.nan, 'fpr_mean': np.nan,
                'tpr_std': np.nan, 'fpr_std': np.nan,
                'fpr_pts_fold0': np.array([0.]), 'tpr_pts_fold0': np.array([0.])
            }

    all_runs = np.array(all_collapse_runs_input + all_stable_runs_input, dtype=object)
    y_all = np.array([run['original_label'] for run in all_runs])

    min_samples_in_class = min(np.sum(y_all == 0), np.sum(y_all == 1))
    if min_samples_in_class < k_folds and min_samples_in_class > 1:
        k_folds = min_samples_in_class
    elif min_samples_in_class <=1: return final_roc_results, lead_time_summary
    if len(all_runs) < k_folds: k_folds = max(2,len(all_runs))
    if k_folds <=1 : return final_roc_results, lead_time_summary

    skf = StratifiedKFold(n_splits=k_folds, shuffle=k_fold_shuffle, random_state=k_fold_seed)
    fold_auc_scores = {ews_name: [] for ews_name, info in ews_signals_info.items() if not info.get('is_point_metric')}
    fold_point_tpr_fpr = {ews_name: [] for ews_name, info in ews_signals_info.items() if info.get('is_point_metric')}
    min_window_fields = ['couple_window', 'trad_ews_window', 'velocity_smoothing_window', 'sigma_dot_theta_window']

    for fold_idx, (train_indices, test_indices) in enumerate(skf.split(all_runs, y_all)):
        print(f"  Processing Fold {fold_idx+1}/{k_folds}...")
        train_runs_fold = all_runs[train_indices]
        test_runs_fold = all_runs[test_indices]

        train_speed_vals_exp, train_couple_vals_exp_neg = [], []
        train_uhb_indicator_vals = []
        train_collapse_runs_for_thresh = [r for r in train_runs_fold if r['original_label'] == 1]

        for run_data_train in train_runs_fold: # For point EWS thresholds
            speed_aggs_pt = get_ews_aggregates_from_horizon(run_data_train, 'Speed', run_data_train['breached'], run_data_train['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_train['params'])
            couple_aggs_pt = get_ews_aggregates_from_horizon(run_data_train, 'Couple', run_data_train['breached'], run_data_train['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_train['params'])
            if not np.isnan(speed_aggs_pt['rising']): train_speed_vals_exp.append(speed_aggs_pt['rising'])
            if not np.isnan(couple_aggs_pt['falling']): train_couple_vals_exp_neg.append(couple_aggs_pt['falling'])

        s_thresh_exp_fold = np.percentile(train_speed_vals_exp, speed_exp_percentile) if train_speed_vals_exp else 0.5
        c_thresh_exp_neg_fold = np.percentile(train_couple_vals_exp_neg, couple_exp_neg_percentile) if train_couple_vals_exp_neg else -0.5

        for run_data_train_collapse in train_collapse_runs_for_thresh: # For UHB point EWS
            speed_aggs_uhb = get_ews_aggregates_from_horizon(run_data_train_collapse, 'Speed', run_data_train_collapse['breached'], run_data_train_collapse['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_train_collapse['params'])
            couple_aggs_uhb = get_ews_aggregates_from_horizon(run_data_train_collapse, 'Couple', run_data_train_collapse['breached'], run_data_train_collapse['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_train_collapse['params'])
            speed_val_uhb, couple_val_uhb = speed_aggs_uhb['rising'], couple_aggs_uhb['rising']
            if not np.isnan(speed_val_uhb) and not np.isnan(couple_val_uhb):
                uhb_score = (speed_val_uhb**2) / (1.0 - couple_val_uhb + uhb_epsilon_analysis)
                train_uhb_indicator_vals.append(uhb_score)
        uhb_thresh_fold = np.percentile(train_uhb_indicator_vals, uhb_indicator_threshold_percentile) if train_uhb_indicator_vals else np.inf

        # --- Ensemble: Calculate scalers for members based on this fold's training data ---
        fold_member_aggregate_scalers = {}
        if ENSEMBLE_EWS_NAME in ews_signals_info:
            for member_name in ENSEMBLE_MEMBERS:
                member_train_agg_scores_rising = []
                member_train_agg_scores_falling = []
                member_train_y_true = []
                for run_train_fold in train_runs_fold:
                    # Avoid using runs that breached if they were supposed to be stable for training scalers
                    if run_train_fold['original_label'] == 0 and run_train_fold['breached']: continue
                    agg = get_ews_aggregates_from_horizon(run_train_fold, member_name, run_train_fold['breached'], run_train_fold['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_train_fold['params'])
                    if not np.isnan(agg['rising']): member_train_agg_scores_rising.append(agg['rising'])
                    else: member_train_agg_scores_rising.append(np.nan) # Keep length consistent
                    if not np.isnan(agg['falling']): member_train_agg_scores_falling.append(agg['falling'])
                    else: member_train_agg_scores_falling.append(np.nan)
                    member_train_y_true.append(run_train_fold['original_label'])
                
                member_train_y_true = np.array(member_train_y_true)
                auc_as_rising_agg, auc_as_falling_agg_signal = 0.5, 0.5
                
                valid_rising_agg = [(yt, sc) for yt, sc in zip(member_train_y_true, member_train_agg_scores_rising) if not np.isnan(sc)]
                if len(valid_rising_agg) > 1 and len(np.unique([yt for yt,_ in valid_rising_agg])) == 2:
                    yt_r, sc_r = zip(*valid_rising_agg); auc_as_rising_agg = roc_auc_score(yt_r, sc_r)

                valid_falling_agg = [(yt, sc) for yt, sc in zip(member_train_y_true, member_train_agg_scores_falling) if not np.isnan(sc)]
                if len(valid_falling_agg) > 1 and len(np.unique([yt for yt,_ in valid_falling_agg])) == 2:
                    yt_f, sc_f = zip(*valid_falling_agg); auc_as_falling_agg_signal = roc_auc_score(yt_f, [-s for s in sc_f])

                chosen_agg_scores_for_scaling = []
                interp_type = 'agg_rising'
                if auc_as_rising_agg >= auc_as_falling_agg_signal:
                    chosen_agg_scores_for_scaling = [s for s in member_train_agg_scores_rising if not np.isnan(s)]
                else:
                    interp_type = 'agg_falling'
                    chosen_agg_scores_for_scaling = [s for s in member_train_agg_scores_falling if not np.isnan(s)]
                
                p5_agg, p95_agg = np.nan, np.nan
                if chosen_agg_scores_for_scaling:
                    p5_agg = np.percentile(chosen_agg_scores_for_scaling, 5)
                    p95_agg = np.percentile(chosen_agg_scores_for_scaling, 95)
                
                fold_member_aggregate_scalers[member_name] = {
                    'interpretation_type': interp_type, # 'agg_rising' or 'agg_falling'
                    'p5_agg': p5_agg, 'p95_agg': p95_agg
                }
        # --- End Ensemble Scaler Calculation for Fold ---

        for ews_name, info in ews_signals_info.items():
            y_true_test_fold = []
            if info.get('is_point_metric', False):
                tp_point, fp_point, actual_pos_point, actual_neg_point = 0,0,0,0
                for run_data_test in test_runs_fold:
                    original_label_test = run_data_test['original_label']
                    prediction_point = 0
                    speed_aggs_test = get_ews_aggregates_from_horizon(run_data_test, 'Speed', run_data_test['breached'], run_data_test['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_test['params'])
                    couple_aggs_test = get_ews_aggregates_from_horizon(run_data_test, 'Couple', run_data_test['breached'], run_data_test['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_test['params'])
                    speed_score_test, couple_score_test_raw = speed_aggs_test['rising'], couple_aggs_test['falling']

                    if ews_name == 'TD_Experimental_Point':
                        if not (np.isnan(speed_score_test) or np.isnan(couple_score_test_raw)):
                            if speed_score_test > s_thresh_exp_fold and couple_score_test_raw < c_thresh_exp_neg_fold:
                                prediction_point = 1
                    elif ews_name == 'UHB_Point_Indicator':
                        couple_score_test_uhb = couple_aggs_test['rising']
                        if not (np.isnan(speed_score_test) or np.isnan(couple_score_test_uhb)):
                            uhb_cont_score_test = (speed_score_test**2) / (1.0 - couple_score_test_uhb + uhb_epsilon_analysis)
                            if uhb_cont_score_test > uhb_thresh_fold: prediction_point = 1
                    if original_label_test == 1: actual_pos_point += 1
                    else: actual_neg_point += 1
                    if prediction_point == 1:
                        if original_label_test == 1: tp_point += 1
                        else: fp_point += 1
                tpr_fold = tp_point / max(1, actual_pos_point); fpr_fold = fp_point / max(1, actual_neg_point)
                fold_point_tpr_fpr[ews_name].append((tpr_fold, fpr_fold))
                if fold_idx == 0:
                    final_roc_results[ews_name]['fpr_pts_fold0'] = np.array([fpr_fold])
                    final_roc_results[ews_name]['tpr_pts_fold0'] = np.array([tpr_fold])
            else: # Continuous EWS (including ensemble)
                test_scores_rising_agg, test_scores_falling_agg = [], []
                for run_data_test in test_runs_fold:
                    y_true_test_fold.append(run_data_test['original_label'])

                    if ews_name == ENSEMBLE_EWS_NAME:
                        scaled_member_aggregates_for_ensemble_run = []
                        for member_name_ens in ENSEMBLE_MEMBERS:
                            raw_aggregates_member = get_ews_aggregates_from_horizon(run_data_test, member_name_ens, run_data_test['breached'], run_data_test['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_test['params'])
                            
                            scalers = fold_member_aggregate_scalers.get(member_name_ens)
                            if not scalers or np.isnan(scalers['p5_agg']) or np.isnan(scalers['p95_agg']):
                                scaled_member_aggregates_for_ensemble_run.append(np.nan)
                                continue

                            score_to_scale_member = raw_aggregates_member[scalers['interpretation_type'].split('_')[1]] # 'rising' or 'falling'
                            
                            p5_mem = scalers['p5_agg']
                            p95_mem = scalers['p95_agg']
                            scaled_agg_member = np.nan

                            if abs(p95_mem - p5_mem) < EPSILON:
                                scaled_agg_member = 0.5 if not np.isnan(score_to_scale_member) else np.nan
                            elif not np.isnan(score_to_scale_member):
                                if scalers['interpretation_type'] == 'agg_rising': # Higher raw aggregate is warning
                                    scaled_agg_member = (score_to_scale_member - p5_mem) / (p95_mem - p5_mem)
                                else: # 'agg_falling', so lower raw aggregate is warning
                                    scaled_agg_member = (p95_mem - score_to_scale_member) / (p95_mem - p5_mem)
                                scaled_agg_member = np.clip(scaled_agg_member, 0, 1)
                            scaled_member_aggregates_for_ensemble_run.append(scaled_agg_member)
                        
                        ensemble_agg_score_for_this_run = np.nanmean(scaled_member_aggregates_for_ensemble_run)
                        test_scores_rising_agg.append(ensemble_agg_score_for_this_run) # Ensemble is always rising
                        test_scores_falling_agg.append(np.nan) # Not applicable for ensemble

                    elif 'natural_type' in info: # Derived continuous EWS
                        score_test = np.nan
                        speed_aggs_derived = get_ews_aggregates_from_horizon(run_data_test, 'Speed', run_data_test['breached'], run_data_test['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_test['params'])
                        couple_aggs_derived = get_ews_aggregates_from_horizon(run_data_test, 'Couple', run_data_test['breached'], run_data_test['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_test['params'])
                        speed_sc, couple_sc = speed_aggs_derived['rising'], couple_aggs_derived['rising']
                        if ews_name == 'TD_Speed_Minus_Couple_Continuous':
                            if not (np.isnan(speed_sc) or np.isnan(couple_sc)): score_test = speed_sc - couple_sc
                        elif ews_name == 'UHB_Indicator_Continuous':
                            if not (np.isnan(speed_sc) or np.isnan(couple_sc)): score_test = (speed_sc**2) / (1.0 - couple_sc + uhb_epsilon_analysis)
                        test_scores_rising_agg.append(score_test); test_scores_falling_agg.append(np.nan)
                    else: # Basic continuous EWS
                        aggregates = get_ews_aggregates_from_horizon(run_data_test, ews_name, run_data_test['breached'], run_data_test['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_test['params'])
                        test_scores_rising_agg.append(aggregates['rising']); test_scores_falling_agg.append(aggregates['falling'])

                if len(np.unique(y_true_test_fold)) < 2:
                    fold_auc_scores[ews_name].append(np.nan)
                    if fold_idx == 0: final_roc_results[ews_name]['fpr_pts_fold0'], final_roc_results[ews_name]['tpr_pts_fold0'] = np.array([0.,1.]), np.array([0.,1.])
                    continue

                auc_as_rising, auc_as_falling_signal = 0.5, 0.5
                valid_rising_scores = [(yt, sc) for yt, sc in zip(y_true_test_fold, test_scores_rising_agg) if not np.isnan(sc)]
                if len(valid_rising_scores) > 1 and len(np.unique([yt for yt,_ in valid_rising_scores])) == 2:
                    yt_r, sc_r = zip(*valid_rising_scores); auc_as_rising = roc_auc_score(yt_r, sc_r)

                chosen_interpretation_this_fold = 'rising'
                if 'natural_type' not in info: # e.g. not ensemble or derived
                    valid_falling_scores = [(yt, sc) for yt, sc in zip(y_true_test_fold, test_scores_falling_agg) if not np.isnan(sc)]
                    if len(valid_falling_scores) > 1 and len(np.unique([yt for yt,_ in valid_falling_scores])) == 2:
                        yt_f, sc_f = zip(*valid_falling_scores); auc_as_falling_signal = roc_auc_score(yt_f, [-s for s in sc_f])
                    best_auc_this_fold = max(auc_as_rising, auc_as_falling_signal)
                    if auc_as_falling_signal > auc_as_rising: chosen_interpretation_this_fold = 'falling'
                else: best_auc_this_fold = auc_as_rising # Ensemble & derived are naturally rising

                fold_auc_scores[ews_name].append(best_auc_this_fold)
                if fold_idx == 0:
                    final_roc_results[ews_name]['chosen_interpretation_fold0'] = chosen_interpretation_this_fold
                    if chosen_interpretation_this_fold == 'rising' and len(valid_rising_scores) > 1:
                        yt_r_plot, sc_r_plot = zip(*valid_rising_scores); fpr_pts_f0, tpr_pts_f0, _ = roc_curve(yt_r_plot, sc_r_plot)
                    elif chosen_interpretation_this_fold == 'falling' and 'natural_type' not in info and len(valid_falling_scores) > 1:
                        yt_f_plot, sc_f_plot = zip(*valid_falling_scores); fpr_pts_f0, tpr_pts_f0, _ = roc_curve(yt_f_plot, [-s for s in sc_f_plot])
                    else: fpr_pts_f0, tpr_pts_f0 = np.array([0.,1.]), np.array([0.,1.])
                    final_roc_results[ews_name]['fpr_pts_fold0'], final_roc_results[ews_name]['tpr_pts_fold0'] = fpr_pts_f0, tpr_pts_f0

    for ews_name_summary in fold_auc_scores:
        valid_aucs = [auc for auc in fold_auc_scores[ews_name_summary] if not np.isnan(auc)]
        if valid_aucs:
            final_roc_results[ews_name_summary]['auc_mean'] = np.mean(valid_aucs)
            final_roc_results[ews_name_summary]['auc_std'] = np.std(valid_aucs)
            print(f"  EWS {ews_name_summary}: Mean AUC = {final_roc_results[ews_name_summary]['auc_mean']:.3f} +/- {final_roc_results[ews_name_summary]['auc_std']:.3f}")
        else: print(f"  EWS {ews_name_summary}: No valid AUCs across folds.")
    for ews_name_summary in fold_point_tpr_fpr:
        if fold_point_tpr_fpr[ews_name_summary]:
            tprs_point, fprs_point = zip(*fold_point_tpr_fpr[ews_name_summary])
            final_roc_results[ews_name_summary]['tpr_mean'] = np.mean(tprs_point); final_roc_results[ews_name_summary]['fpr_mean'] = np.mean(fprs_point)
            final_roc_results[ews_name_summary]['tpr_std'] = np.std(tprs_point); final_roc_results[ews_name_summary]['fpr_std'] = np.std(fprs_point)
            print(f"  Point EWS {ews_name_summary}: Mean TPR={np.mean(tprs_point):.3f} (+/- {np.std(tprs_point):.3f}), Mean FPR={np.mean(fprs_point):.3f} (+/- {np.std(fprs_point):.3f})")
        else: print(f"  Point EWS {ews_name_summary}: No valid TPR/FPR data across folds.")

    print("\nLead Time Analysis (Thresholds from dynamically determined best interpretation from ALL training data, evaluated on all collapse runs):")
    temp_skf_for_lt_indices = StratifiedKFold(n_splits=k_folds, shuffle=k_fold_shuffle, random_state=k_fold_seed)
    all_train_indices_lt = set()
    for train_idx_set, _ in temp_skf_for_lt_indices.split(all_runs, y_all): all_train_indices_lt.update(train_idx_set)
    all_train_runs_for_lead_time_thresh = all_runs[list(all_train_indices_lt)]

    ews_best_interpretation_lt = {} # For individual EWS
    all_train_labels_lt_lead = []
    all_train_scores_basic_lt_lead = {ews_n: {'rising_agg': [], 'falling_agg': []} for ews_n in ews_signals_info if not ews_signals_info[ews_n].get('is_point_metric') and 'natural_type' not in ews_signals_info[ews_n] and ews_n != ENSEMBLE_EWS_NAME}
    all_train_scores_derived_lt_lead = {ews_n: [] for ews_n in ews_signals_info if ews_signals_info[ews_n].get('natural_type') and ews_n != ENSEMBLE_EWS_NAME}

    for run_data_train_lt in all_train_runs_for_lead_time_thresh:
        if run_data_train_lt['original_label'] == 0 and run_data_train_lt['breached']: continue
        all_train_labels_lt_lead.append(run_data_train_lt['original_label'])
        for ews_n_lt in all_train_scores_basic_lt_lead.keys():
            aggregates = get_ews_aggregates_from_horizon(run_data_train_lt, ews_n_lt, run_data_train_lt['breached'], run_data_train_lt['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_train_lt['params'])
            all_train_scores_basic_lt_lead[ews_n_lt]['rising_agg'].append(aggregates['rising']); all_train_scores_basic_lt_lead[ews_n_lt]['falling_agg'].append(aggregates['falling'])
        for ews_n_lt in all_train_scores_derived_lt_lead.keys():
            score_lt_derived = np.nan
            speed_aggs_d = get_ews_aggregates_from_horizon(run_data_train_lt, 'Speed', run_data_train_lt['breached'], run_data_train_lt['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_train_lt['params'])
            couple_aggs_d = get_ews_aggregates_from_horizon(run_data_train_lt, 'Couple', run_data_train_lt['breached'], run_data_train_lt['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_train_lt['params'])
            speed_sc_d, couple_sc_d = speed_aggs_d['rising'], couple_aggs_d['rising']
            if ews_n_lt == 'TD_Speed_Minus_Couple_Continuous':
                if not (np.isnan(speed_sc_d) or np.isnan(couple_sc_d)): score_lt_derived = speed_sc_d - couple_sc_d
            elif ews_n_lt == 'UHB_Indicator_Continuous':
                if not (np.isnan(speed_sc_d) or np.isnan(couple_sc_d)): score_lt_derived = (speed_sc_d**2) / (1.0 - couple_sc_d + uhb_epsilon_analysis)
            all_train_scores_derived_lt_lead[ews_n_lt].append(score_lt_derived)

    for ews_name_lt_interp in all_train_scores_basic_lt_lead.keys():
        valid_indices = [i for i, (r_val, f_val) in enumerate(zip(all_train_scores_basic_lt_lead[ews_name_lt_interp]['rising_agg'], all_train_scores_basic_lt_lead[ews_name_lt_interp]['falling_agg'])) if not np.isnan(r_val) or not np.isnan(f_val)]
        if not valid_indices or len(np.unique([all_train_labels_lt_lead[i] for i in valid_indices])) < 2:
            ews_best_interpretation_lt[ews_name_lt_interp] = 'rising'; continue
        current_y_true = [all_train_labels_lt_lead[i] for i in valid_indices]
        
        auc_lt_as_rising_interp = 0.5
        current_scores_rising_valid = [all_train_scores_basic_lt_lead[ews_name_lt_interp]['rising_agg'][i] for i in valid_indices if not np.isnan(all_train_scores_basic_lt_lead[ews_name_lt_interp]['rising_agg'][i])]
        current_y_true_rising = [all_train_labels_lt_lead[i] for i in valid_indices if not np.isnan(all_train_scores_basic_lt_lead[ews_name_lt_interp]['rising_agg'][i])]
        if len(current_scores_rising_valid) > 1 and len(np.unique(current_y_true_rising)) == 2:
            auc_lt_as_rising_interp = roc_auc_score(current_y_true_rising, current_scores_rising_valid)

        auc_lt_as_falling_signal_interp = 0.5
        current_scores_falling_valid = [all_train_scores_basic_lt_lead[ews_name_lt_interp]['falling_agg'][i] for i in valid_indices if not np.isnan(all_train_scores_basic_lt_lead[ews_name_lt_interp]['falling_agg'][i])]
        current_y_true_falling = [all_train_labels_lt_lead[i] for i in valid_indices if not np.isnan(all_train_scores_basic_lt_lead[ews_name_lt_interp]['falling_agg'][i])]
        if len(current_scores_falling_valid) > 1 and len(np.unique(current_y_true_falling)) == 2:
            auc_lt_as_falling_signal_interp = roc_auc_score(current_y_true_falling, [-s for s in current_scores_falling_valid])
        
        ews_best_interpretation_lt[ews_name_lt_interp] = 'rising' if auc_lt_as_rising_interp >= auc_lt_as_falling_signal_interp else 'falling'
    for ews_name_lt_interp in all_train_scores_derived_lt_lead.keys(): ews_best_interpretation_lt[ews_name_lt_interp] = ews_signals_info[ews_name_lt_interp].get('natural_type', 'rising')
    if ENSEMBLE_EWS_NAME in ews_signals_info: ews_best_interpretation_lt[ENSEMBLE_EWS_NAME] = 'rising'


    # --- Ensemble: Global scalers for lead time analysis ---
    global_member_raw_ts_percentiles = {}
    global_member_aggregate_scalers = {}
    if ENSEMBLE_EWS_NAME in ews_signals_info:
        for member_name_global in ENSEMBLE_MEMBERS:
            # Raw time-series percentiles
            all_raw_ts_values_member = []
            for run_train_lt_global in all_train_runs_for_lead_time_thresh:
                if run_train_lt_global['original_label'] == 0 and run_train_lt_global['breached']: continue
                if member_name_global in run_train_lt_global and run_train_lt_global[member_name_global] is not None:
                    valid_ts_vals = np.array(run_train_lt_global[member_name_global])
                    all_raw_ts_values_member.extend(valid_ts_vals[np.isfinite(valid_ts_vals)])
            
            p5_raw_ts, p95_raw_ts = np.nan, np.nan
            if all_raw_ts_values_member:
                p5_raw_ts = np.percentile(all_raw_ts_values_member, 5)
                p95_raw_ts = np.percentile(all_raw_ts_values_member, 95)
            global_member_raw_ts_percentiles[member_name_global] = {'p5_raw_ts': p5_raw_ts, 'p95_raw_ts': p95_raw_ts}

            # Aggregated score percentiles (similar to fold logic but on all_train_runs_for_lead_time_thresh)
            member_train_agg_scores_rising_global = []
            member_train_agg_scores_falling_global = []
            # y_true for these aggregates is all_train_labels_lt_lead (already filtered for bad stable runs)
            
            # Re-fetch aggregates for global scaling (could optimize by storing earlier)
            temp_agg_rising = []
            temp_agg_falling = []
            for run_train_lt_global in all_train_runs_for_lead_time_thresh:
                 if run_train_lt_global['original_label'] == 0 and run_train_lt_global['breached']: continue
                 agg = get_ews_aggregates_from_horizon(run_train_lt_global, member_name_global, run_train_lt_global['breached'], run_train_lt_global['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_train_lt_global['params'])
                 temp_agg_rising.append(agg['rising'])
                 temp_agg_falling.append(agg['falling'])

            # Use ews_best_interpretation_lt to decide which aggregates to use for p5/p95_agg_global
            interp_agg_global = ews_best_interpretation_lt.get(member_name_global, 'rising')
            chosen_agg_scores_for_scaling_global = []
            if interp_agg_global == 'rising':
                chosen_agg_scores_for_scaling_global = [s for s in temp_agg_rising if not np.isnan(s)]
                agg_interp_type_global = 'agg_rising'
            else: # falling
                chosen_agg_scores_for_scaling_global = [s for s in temp_agg_falling if not np.isnan(s)]
                agg_interp_type_global = 'agg_falling'

            p5_agg_glob, p95_agg_glob = np.nan, np.nan
            if chosen_agg_scores_for_scaling_global:
                p5_agg_glob = np.percentile(chosen_agg_scores_for_scaling_global, 5)
                p95_agg_glob = np.percentile(chosen_agg_scores_for_scaling_global, 95)
            global_member_aggregate_scalers[member_name_global] = {
                'interpretation_type': agg_interp_type_global, # 'agg_rising' or 'agg_falling'
                'p5_agg': p5_agg_glob, 'p95_agg': p95_agg_glob
            }
    # --- End Ensemble Global Scaler Calculation ---


    for ews_name_lt, info_lt_main in ews_signals_info.items():
        if info_lt_main.get('is_point_metric', False):
            if ews_name_lt in final_roc_results and 'tpr_mean' in final_roc_results[ews_name_lt] :
                 if np.isnan(final_roc_results[ews_name_lt]['tpr_mean']):
                      print(f"  {ews_name_lt}: No valid scores from training collapse runs for lead time thresholding (Point EWS).")
            else: print(f"  {ews_name_lt}: Point EWS - lead time logic not fully integrated for this EWS type.")
            continue

        best_type_for_lt = ews_best_interpretation_lt.get(ews_name_lt)
        if not best_type_for_lt: continue # Should always be set for continuous

        scores_from_collapse_train_for_thresh = []
        
        if ews_name_lt == ENSEMBLE_EWS_NAME:
            for run_data_train_lt_ens in all_train_runs_for_lead_time_thresh:
                if run_data_train_lt_ens['original_label'] == 1: # Only collapse runs for threshold
                    if run_data_train_lt_ens['breached'] == False : continue # Ensure it actually collapsed
                    
                    scaled_member_aggregates_for_ens_thresh = []
                    for member_name_ens_lt in ENSEMBLE_MEMBERS:
                        raw_aggregates_member_lt = get_ews_aggregates_from_horizon(run_data_train_lt_ens, member_name_ens_lt, run_data_train_lt_ens['breached'], run_data_train_lt_ens['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_train_lt_ens['params'])
                        scalers_lt = global_member_aggregate_scalers.get(member_name_ens_lt)
                        if not scalers_lt or np.isnan(scalers_lt['p5_agg']) or np.isnan(scalers_lt['p95_agg']):
                            scaled_member_aggregates_for_ens_thresh.append(np.nan)
                            continue
                        
                        score_to_scale_member_lt = raw_aggregates_member_lt[scalers_lt['interpretation_type'].split('_')[1]]
                        p5_lt = scalers_lt['p5_agg']; p95_lt = scalers_lt['p95_agg']
                        scaled_agg_member_lt = np.nan

                        if abs(p95_lt - p5_lt) < EPSILON: scaled_agg_member_lt = 0.5 if not np.isnan(score_to_scale_member_lt) else np.nan
                        elif not np.isnan(score_to_scale_member_lt):
                            if scalers_lt['interpretation_type'] == 'agg_rising':
                                scaled_agg_member_lt = (score_to_scale_member_lt - p5_lt) / (p95_lt - p5_lt)
                            else: # agg_falling
                                scaled_agg_member_lt = (p95_lt - score_to_scale_member_lt) / (p95_lt - p5_lt)
                            scaled_agg_member_lt = np.clip(scaled_agg_member_lt, 0, 1)
                        scaled_member_aggregates_for_ens_thresh.append(scaled_agg_member_lt)
                    
                    ensemble_agg_score_for_thresh_run = np.nanmean(scaled_member_aggregates_for_ens_thresh)
                    if not np.isnan(ensemble_agg_score_for_thresh_run):
                        scores_from_collapse_train_for_thresh.append(ensemble_agg_score_for_thresh_run)
        else: # Individual EWS threshold calculation
            for run_data_train_lt in all_train_runs_for_lead_time_thresh:
                if run_data_train_lt['original_label'] == 1:
                    if run_data_train_lt['breached'] == False : continue
                    aggregates_thresh = get_ews_aggregates_from_horizon(run_data_train_lt, ews_name_lt, run_data_train_lt['breached'], run_data_train_lt['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_train_lt['params'])
                    score_to_add = aggregates_thresh.get(best_type_for_lt)
                    if score_to_add is None and 'natural_type' in info_lt_main:
                        score_to_add = aggregates_thresh.get(info_lt_main['natural_type'])

                    if 'natural_type' in info_lt_main: # Derived EWS score calculation for threshold
                        score_lt_derived_thresh = np.nan
                        speed_aggs_d_thresh = get_ews_aggregates_from_horizon(run_data_train_lt, 'Speed', run_data_train_lt['breached'], run_data_train_lt['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_train_lt['params'])
                        couple_aggs_d_thresh = get_ews_aggregates_from_horizon(run_data_train_lt, 'Couple', run_data_train_lt['breached'], run_data_train_lt['breach_step_idx'], warning_horizon_fraction, agg_perc_rising, agg_perc_falling, min_window_fields, run_data_train_lt['params'])
                        speed_sc_d_thresh, couple_sc_d_thresh = speed_aggs_d_thresh['rising'], couple_aggs_d_thresh['rising']
                        if ews_name_lt == 'TD_Speed_Minus_Couple_Continuous':
                            if not (np.isnan(speed_sc_d_thresh) or np.isnan(couple_sc_d_thresh)): score_lt_derived_thresh = speed_sc_d_thresh - couple_sc_d_thresh
                        elif ews_name_lt == 'UHB_Indicator_Continuous':
                            if not (np.isnan(speed_sc_d_thresh) or np.isnan(couple_sc_d_thresh)): score_lt_derived_thresh = (speed_sc_d_thresh**2) / (1.0 - couple_sc_d_thresh + uhb_epsilon_analysis)
                        score_to_add = score_lt_derived_thresh
                    if not np.isnan(score_to_add): scores_from_collapse_train_for_thresh.append(score_to_add)
        
        if not scores_from_collapse_train_for_thresh:
            print(f"  {ews_name_lt}: No valid scores from training collapse runs for lead time thresholding ({best_type_for_lt} interpretation).")
            continue

        # For rising signals (ensemble is always rising), a lower percentile of high scores is a more sensitive threshold.
        # For falling signals, a higher percentile of low scores is more sensitive.
        lt_percentile_for_thresh = 25 if best_type_for_lt == 'rising' else 75
        lead_time_threshold = np.percentile(scores_from_collapse_train_for_thresh, lt_percentile_for_thresh)

        current_lead_times, detections_at_op, num_valid_collapse_for_lead = [], 0, 0
        stable_detections, num_valid_stable_for_fp = 0, 0
        for run_data_lt_eval in all_collapse_runs_input:
            if not run_data_lt_eval['breached'] or run_data_lt_eval['breach_step_idx'] < 0 : continue
            
            # For ensemble, generate its instantaneous time series first
            if ews_name_lt == ENSEMBLE_EWS_NAME:
                ensemble_instant_score_timeseries = np.full(run_data_lt_eval['breach_step_idx'], np.nan)
                for t_idx in range(run_data_lt_eval['breach_step_idx']):
                    member_instant_scaled_scores_eval = []
                    for member_name_eval in ENSEMBLE_MEMBERS:
                        raw_val_member = run_data_lt_eval[member_name_eval][t_idx]
                        scalers_raw_ts = global_member_raw_ts_percentiles.get(member_name_eval)
                        interp_member_raw = ews_best_interpretation_lt.get(member_name_eval) == 'rising'

                        if scalers_raw_ts and not np.isnan(scalers_raw_ts['p5_raw_ts']) and not np.isnan(scalers_raw_ts['p95_raw_ts']):
                            scaled_val = scale_instant_value(raw_val_member, scalers_raw_ts['p5_raw_ts'], scalers_raw_ts['p95_raw_ts'], interp_member_raw)
                            member_instant_scaled_scores_eval.append(scaled_val)
                        else:
                            member_instant_scaled_scores_eval.append(np.nan)
                    ensemble_instant_score_timeseries[t_idx] = np.nanmean(member_instant_scaled_scores_eval)
                full_signal_pre_breach_lt = ensemble_instant_score_timeseries
            elif ews_name_lt not in run_data_lt_eval or run_data_lt_eval[ews_name_lt] is None:
                continue # Skip if EWS data is missing for individual EWS
            else: # Individual EWS
                full_signal_pre_breach_lt = np.array(run_data_lt_eval[ews_name_lt][:run_data_lt_eval['breach_step_idx']]).flatten()

            num_valid_collapse_for_lead +=1
            params_eval = run_data_lt_eval['params']
            
            # Min window for ensemble is implicitly handled by its members having data.
            # For individual EWS, apply their specific min window.
            min_total_calc_window_lt = 1
            if ews_name_lt != ENSEMBLE_EWS_NAME:
                current_min_windows_list_lt = [params_eval.get(field, 1) for field in min_window_fields if field in params_eval]
                min_total_calc_window_lt = max(current_min_windows_list_lt) if current_min_windows_list_lt else 1
                if ews_name_lt in ['dot_LCI_S', 'TMCRN']: min_total_calc_window_lt = max(min_total_calc_window_lt, params_eval.get('velocity_smoothing_window',1))
                elif ews_name_lt == 'Couple': min_total_calc_window_lt = max(min_total_calc_window_lt, params_eval.get('couple_window',1))
                elif ews_name_lt in ['Variance_Y', 'AR1_Y', 'Skewness_Y', 'Kurtosis_Y', 'Variance_Theta_T', 'AR1_Theta_T']: min_total_calc_window_lt = max(min_total_calc_window_lt, params_eval.get('trad_ews_window', 1))
            
            if len(full_signal_pre_breach_lt) < min_total_calc_window_lt: continue
            valid_ews_start_idx_lt = min_total_calc_window_lt - 1
            if valid_ews_start_idx_lt >= len(full_signal_pre_breach_lt) : continue
            
            ews_signal_part_lt = full_signal_pre_breach_lt[valid_ews_start_idx_lt:]
            original_indices_in_full_signal_lt = np.arange(valid_ews_start_idx_lt, valid_ews_start_idx_lt + len(ews_signal_part_lt))
            if len(ews_signal_part_lt) == 0: continue
            
            warn_indices_logic = np.zeros_like(ews_signal_part_lt, dtype=bool); finite_mask = np.isfinite(ews_signal_part_lt)
            # best_type_for_lt is 'rising' for ensemble
            if best_type_for_lt == 'rising': warn_indices_logic = (ews_signal_part_lt > lead_time_threshold) & finite_mask
            else: warn_indices_logic = (ews_signal_part_lt < lead_time_threshold) & finite_mask
            actual_warn_indices_in_ews_part = np.where(warn_indices_logic)[0]

            if actual_warn_indices_in_ews_part.size > 0:
                detections_at_op +=1
                first_warn_idx_relative = actual_warn_indices_in_ews_part[0]
                first_warn_idx_in_full = original_indices_in_full_signal_lt[first_warn_idx_relative]
                current_lead_times.append((run_data_lt_eval['breach_step_idx'] - first_warn_idx_in_full) * params_eval['dt'])

        for run_data_stable_eval in all_stable_runs_input:
            if run_data_stable_eval['breached']:
                continue
            if ews_name_lt == ENSEMBLE_EWS_NAME:
                ensemble_instant_score_timeseries_s = np.full(len(run_data_stable_eval['t']), np.nan)
                for t_idx in range(len(run_data_stable_eval['t'])):
                    member_vals_s = []
                    for member_name_eval in ENSEMBLE_MEMBERS:
                        raw_val_member = run_data_stable_eval[member_name_eval][t_idx]
                        scalers_raw_ts = global_member_raw_ts_percentiles.get(member_name_eval)
                        interp_member_raw = ews_best_interpretation_lt.get(member_name_eval) == 'rising'
                        if scalers_raw_ts and not np.isnan(scalers_raw_ts['p5_raw_ts']) and not np.isnan(scalers_raw_ts['p95_raw_ts']):
                            scaled_val = scale_instant_value(raw_val_member, scalers_raw_ts['p5_raw_ts'], scalers_raw_ts['p95_raw_ts'], interp_member_raw)
                            member_vals_s.append(scaled_val)
                        else:
                            member_vals_s.append(np.nan)
                    ensemble_instant_score_timeseries_s[t_idx] = np.nanmean(member_vals_s)
                full_signal_stable = ensemble_instant_score_timeseries_s
            elif ews_name_lt not in run_data_stable_eval or run_data_stable_eval[ews_name_lt] is None:
                continue
            else:
                full_signal_stable = np.array(run_data_stable_eval[ews_name_lt]).flatten()

            params_eval_s = run_data_stable_eval['params']
            min_total_calc_window_s = 1
            if ews_name_lt != ENSEMBLE_EWS_NAME:
                current_min_windows_list_s = [params_eval_s.get(field,1) for field in min_window_fields if field in params_eval_s]
                min_total_calc_window_s = max(current_min_windows_list_s) if current_min_windows_list_s else 1
                if ews_name_lt in ['dot_LCI_S', 'TMCRN']:
                    min_total_calc_window_s = max(min_total_calc_window_s, params_eval_s.get('velocity_smoothing_window',1))
                elif ews_name_lt == 'Couple':
                    min_total_calc_window_s = max(min_total_calc_window_s, params_eval_s.get('couple_window',1))
                elif ews_name_lt in ['Variance_Y','AR1_Y','Skewness_Y','Kurtosis_Y','Variance_Theta_T','AR1_Theta_T']:
                    min_total_calc_window_s = max(min_total_calc_window_s, params_eval_s.get('trad_ews_window',1))

            if len(full_signal_stable) < min_total_calc_window_s:
                continue
            valid_start_idx_s = min_total_calc_window_s - 1
            if valid_start_idx_s >= len(full_signal_stable):
                continue
            ews_signal_part_s = full_signal_stable[valid_start_idx_s:]
            if len(ews_signal_part_s) == 0:
                continue
            finite_mask_s = np.isfinite(ews_signal_part_s)
            if best_type_for_lt == 'rising':
                cross_bool = np.any((ews_signal_part_s > lead_time_threshold) & finite_mask_s)
            else:
                cross_bool = np.any((ews_signal_part_s < lead_time_threshold) & finite_mask_s)
            num_valid_stable_for_fp += 1
            if cross_bool:
                stable_detections += 1

        sd_lt = np.std(current_lead_times) if current_lead_times else np.nan
        stable_fp_rate = (stable_detections / num_valid_stable_for_fp) if num_valid_stable_for_fp else np.nan

        if current_lead_times:
            mean_lt = np.mean(current_lead_times)
            med_lt = np.median(current_lead_times)
            min_lt = np.min(current_lead_times)
            max_lt = np.max(current_lead_times)
            print(f"  {ews_name_lt} (Thresh={lead_time_threshold:.3f} from {lt_percentile_for_thresh}th perc of train collapses ({best_type_for_lt}), Dets: {detections_at_op}/{num_valid_collapse_for_lead}): "
                  f"MeanL: {mean_lt:.2f}, MedL: {med_lt:.2f}, MinL: {min_lt:.2f}, MaxL: {max_lt:.2f}")
            lead_time_summary[ews_name_lt] = {
                'threshold': float(lead_time_threshold),
                'interpretation': best_type_for_lt,
                'detections': int(detections_at_op),
                'num_valid_runs': int(num_valid_collapse_for_lead),
                'mean': float(mean_lt),
                'median': float(med_lt),
                'min': float(min_lt),
                'max': float(max_lt),
                'sd': float(sd_lt),
                'stable_fp_rate': float(stable_fp_rate)
            }
        else:
            print(f"  {ews_name_lt} (Thresh={lead_time_threshold:.3f}, interp: {best_type_for_lt}): No detections out of {num_valid_collapse_for_lead} valid collapse runs.")
            lead_time_summary[ews_name_lt] = {
                'threshold': float(lead_time_threshold),
                'interpretation': best_type_for_lt,
                'detections': 0,
                'num_valid_runs': int(num_valid_collapse_for_lead),
                'mean': np.nan,
                'median': np.nan,
                'min': np.nan,
                'max': np.nan,
                'sd': np.nan,
                'stable_fp_rate': float(stable_fp_rate)
            }

    if SHOW_PLOTS: # Plotting logic remains largely the same
        plt.figure(figsize=(16, 12))
        plot_order = list(ews_signals_info.keys())
        for ews_name_plot in plot_order:
            if ews_name_plot not in final_roc_results: continue
            data_plot = final_roc_results[ews_name_plot]
            if ews_signals_info[ews_name_plot].get('is_point_metric', False):
                if not np.isnan(data_plot.get('tpr_mean', np.nan)):
                     plt.plot(data_plot['fpr_mean'], data_plot['tpr_mean'], marker='*', markersize=15, label=f"{ews_name_plot} (Mean Pt: TPR={data_plot['tpr_mean']:.2f}, FPR={data_plot['fpr_mean']:.2f})", linestyle='None', zorder=10)
                elif len(data_plot['fpr_pts_fold0']) > 0 and len(data_plot['tpr_pts_fold0']) > 0 and not (np.isnan(data_plot['fpr_pts_fold0'][0]) or np.isnan(data_plot['tpr_pts_fold0'][0])):
                     plt.plot(data_plot['fpr_pts_fold0'][0], data_plot['tpr_pts_fold0'][0], marker='x', markersize=10, label=f"{ews_name_plot} (Fold0 Pt: TPR={data_plot['tpr_pts_fold0'][0]:.2f}, FPR={data_plot['fpr_pts_fold0'][0]:.2f})", linestyle='None', zorder=9)
            else:
                fpr_pts, tpr_pts = data_plot['fpr_pts_fold0'], data_plot['tpr_pts_fold0']
                auc_mean_plot, auc_std_plot = data_plot.get('auc_mean', np.nan), data_plot.get('auc_std', np.nan)
                interp_f0_plot = data_plot.get('chosen_interpretation_fold0', 'rising')
                label_plot = f"{ews_name_plot} (AUC={auc_mean_plot:.2f}\u00B1{auc_std_plot:.2f}, F0:{interp_f0_plot[0]})" if not np.isnan(auc_mean_plot) else f"{ews_name_plot} (AUC=NaN)"
                plt.plot(fpr_pts, tpr_pts, marker='.', label=label_plot, lw=1.5, ms=5)
        plt.plot([0,1],[0,1], linestyle='--', color='gray', label='Random (AUC=0.50)')
        plt.xlabel('False Positive Rate (FPR)'); plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC Curves (Fold 0 data, best interp.) & Mean Performance over {k_folds} Folds{title_suffix}')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small'); plt.grid(True, alpha=0.4)
        plt.xlim([-0.02,1.02]); plt.ylim([-0.02,1.02]); plt.tight_layout()
        fig_name = f"roc_curves{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(os.path.join(results_dir, fig_name))
        plt.show()
    return final_roc_results, lead_time_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TD early warning signal study")
    parser.add_argument("--quick", action="store_true", help="Disable plotting for faster execution")
    parser.add_argument("--debug", action="store_true", help="Also write per-scenario CSV outputs")
    args = parser.parse_args()
    if args.quick:
        SHOW_PLOTS = False

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_scenario_params = {
        "Original Slow Collapse": params_collapse_original,
        "Shock-Induced Collapse": params_collapse_shock,
        "Resource Exhaustion Collapse": params_collapse_resource,
        "Adaptation Failure Collapse": params_collapse_adaptation,
        "Lever Instability Collapse": params_collapse_lever_instability,
        "Massive Strain Shock Collapse": params_collapse_strain_shock,
        "G-Lever Catastrophe Collapse": params_collapse_g_catastrophe,
        "Beta Self-Destruct Collapse": params_collapse_beta_destruct,
    }

    master_seed_sequence = np.random.SeedSequence(DEFAULT_MASTER_SIM_SEED)
    num_total_scenarios = len(all_scenario_params)
    child_seeds = master_seed_sequence.spawn(NUM_STABLE_RUNS + NUM_COLLAPSE_RUNS_PER_SCENARIO * num_total_scenarios)
    rng_counter = 0

    print(f"--- Generating Stable Scenario Runs ({NUM_STABLE_RUNS} runs) ---")
    all_stable_results = []
    for i in range(NUM_STABLE_RUNS):
        if (i+1) % (max(1, NUM_STABLE_RUNS // 10)) == 0 : print(f"  Running Stable Scenario: {i+1}/{NUM_STABLE_RUNS}")
        sim_rng = np.random.Generator(np.random.PCG64(child_seeds[rng_counter])); rng_counter += 1
        all_stable_results.append(simulate_system_run(params_stable_config, run_id=f"stable_{i}", rng=sim_rng))

    num_stable_collapses = sum(1 for r in all_stable_results if r['breached'])
    print(f"\nStable Run Summary: {num_stable_collapses}/{NUM_STABLE_RUNS} inadvertently collapsed.")

    scenarios_to_plot_examples = [
        "Original Slow Collapse",
        "Lever Instability Collapse",
        "G-Lever Catastrophe Collapse"
    ]
    plotted_example_count = {name: False for name in scenarios_to_plot_examples}

    all_roc_results = {}
    all_lt_results = {}
    run_counts_list = []


    for scenario_name, current_collapse_params in all_scenario_params.items():
        print(f"\n--- Running {scenario_name} ({NUM_COLLAPSE_RUNS_PER_SCENARIO} runs) ---")
        all_current_collapse_results = []
        plot_this_scenario_type_example = scenario_name in scenarios_to_plot_examples and not plotted_example_count[scenario_name]

        for i in range(NUM_COLLAPSE_RUNS_PER_SCENARIO):
            if (i+1) % (max(1, NUM_COLLAPSE_RUNS_PER_SCENARIO // 10)) == 0 : print(f"  Running {scenario_name}: {i+1}/{NUM_COLLAPSE_RUNS_PER_SCENARIO}")
            sim_rng = np.random.Generator(np.random.PCG64(child_seeds[rng_counter])); rng_counter += 1
            run_result = simulate_system_run(current_collapse_params, run_id=f"{current_collapse_params.get('run_type', 'collapse')}_{i}", rng=sim_rng)
            all_current_collapse_results.append(run_result)

            if i == 0 and plot_this_scenario_type_example and SHOW_PLOTS: # Plotting logic is unchanged
                data = run_result
                print(f"    Example {scenario_name} Run {data['run_id']} details: Breached={data['breached']} at t={data['breach_time']:.2f} ({data['breach_type']})")
                fig_ens, axes_ens = plt.subplots(6, 1, figsize=(12, 21), sharex=True)
                axes_ens[0].plot(data['t'], data['Theta_T'], label=r'$\Theta_T$', lw=1.5, color='black'); axes_ens[0].plot(data['t'], data['Strain'], label='Strain', ls='--', lw=1.5, color='red')
                ax0_twin = axes_ens[0].twinx(); ax0_twin.plot(data['t'], data['g'], label=r'$g$',alpha=0.6, color='blue'); ax0_twin.plot(data['t'], data['beta'], label=r'$\beta$',alpha=0.6, color='purple'); ax0_twin.plot(data['t'], data['F_crit'], label=r'$F_{crit}$',alpha=0.6, color='brown')
                if data['breached']: axes_ens[0].axvline(data['breach_time'], color='k', ls='-.', alpha=0.7)
                lines1, labels1 = axes_ens[0].get_legend_handles_labels(); lines2, labels2 = ax0_twin.get_legend_handles_labels(); axes_ens[0].legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize='small'); axes_ens[0].set_title(f'State Variables - Example {scenario_name}')
                axes_ens[1].plot(data['t'], data['Speed'], label='Speed',lw=1.5, color='orangered'); ax1_twin = axes_ens[1].twinx(); ax1_twin.plot(data['t'], data['Couple'], label='Couple',lw=1.5, color='deepskyblue')
                if data['breached']: axes_ens[1].axvline(data['breach_time'], color='k', ls='-.', alpha=0.7)
                lines1, labels1 = axes_ens[1].get_legend_handles_labels(); lines2, labels2 = ax1_twin.get_legend_handles_labels(); axes_ens[1].legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize='small'); axes_ens[1].set_title('Original TD EWS (Speed, Couple)')
                axes_ens[2].plot(data['t'], data['Y_obs'], label='Y_obs (for Var/AR1)', lw=1.0, color='gray', alpha=0.3)
                ax2_var = axes_ens[2].twinx(); ax2_var.plot(data['t'], data['Variance_Y'], label='Var(Y)',lw=1.5, color='green');
                ax2_ar1 = axes_ens[2].twinx(); ax2_ar1.spines["right"].set_position(("outward", 60)); ax2_ar1.plot(data['t'], data['AR1_Y'], label='AR1(Y)',lw=1.5, color='limegreen')
                if data['breached']: axes_ens[2].axvline(data['breach_time'], color='k', ls='-.', alpha=0.7)
                lines1, labels1 = axes_ens[2].get_legend_handles_labels(); lines2, labels2 = ax2_var.get_legend_handles_labels(); lines3, labels3 = ax2_ar1.get_legend_handles_labels(); axes_ens[2].legend(lines1+lines2+lines3, labels1+labels2+labels3, loc='center right', fontsize='small'); axes_ens[2].set_title('Traditional EWS (Y_obs background)')
                axes_ens[3].plot(data['t'], data['LCI'], label='LCI', lw=1.5, color='gold', alpha=0.9)
                ax3_twin = axes_ens[3].twinx()
                uhb_indicator_plot = (np.array(data['Speed'])**2) / (1.0 - np.array(data['Couple']) + data['params'].get('uhb_epsilon', EPSILON)) if 'Speed' in data and 'Couple' in data else np.full_like(data['t'], np.nan)
                ax3_twin.plot(data['t'], uhb_indicator_plot, label=r'UHB Ind. Cont.', lw=1.2, color='darkviolet', alpha=0.9)
                if data['breached']: axes_ens[3].axvline(data['breach_time'], color='k', ls='-.', alpha=0.7)
                lines1, labels1 = axes_ens[3].get_legend_handles_labels(); lines2, labels2 = ax3_twin.get_legend_handles_labels(); axes_ens[3].legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize='small'); axes_ens[3].set_title('LCI & UHB-Indicator (Continuous)')
                min_uhb_val_plot = np.nanmin(np.abs(uhb_indicator_plot[np.isfinite(uhb_indicator_plot)])) if np.any(np.isfinite(uhb_indicator_plot)) else EPSILON
                ax3_twin.set_yscale('symlog', linthresh=max(min_uhb_val_plot*0.1, EPSILON))
                axes_ens[4].plot(data['t'], data['dot_Theta_T_norm'], label=r'$\dot{\Theta}_T/{\Theta}_T$', lw=1.2, color='teal', alpha=0.8)
                ax4_twin_a = axes_ens[4].twinx()
                ax4_twin_a.plot(data['t'], data['Fcrit_DepVelN'], label='Fcrit_DepVelN', lw=1.2, color='darkorange', alpha=0.6)
                if data['breached']: axes_ens[4].axvline(data['breach_time'], color='k', ls='-.', alpha=0.7)
                lines1, labels1 = axes_ens[4].get_legend_handles_labels(); lines2, labels2 = ax4_twin_a.get_legend_handles_labels()
                axes_ens[4].legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize='small'); axes_ens[4].set_title(r'$\dot{\Theta}_T/{\Theta}_T$ & Fcrit_DepVelN')
                axes_ens[5].plot(data['t'], data.get('TMCRN', np.full_like(data['t'], np.nan)), label='TMCRN', lw=1.2, color='magenta', alpha=0.8)
                ax5_twin_a = axes_ens[5].twinx()
                ax5_twin_a.plot(data['t'], data.get('AR1_Theta_T', np.full_like(data['t'], np.nan)), label=r'AR1($\Theta_T$)', lw=1.2, color='firebrick', alpha=0.7)
                ax5_twin_b = axes_ens[5].twinx()
                ax5_twin_b.spines["right"].set_position(("outward", 60))
                ax5_twin_b.plot(data['t'], data.get('RMA_norm', np.full_like(data['t'], np.nan)), label='RMA_norm', lw=1.2, color='dodgerblue', alpha=0.6)
                if data['breached']: axes_ens[5].axvline(data['breach_time'], color='k', ls='-.', alpha=0.7)
                lines1, labels1 = axes_ens[5].get_legend_handles_labels(); lines2, labels2 = ax5_twin_a.get_legend_handles_labels(); lines3, labels3 = ax5_twin_b.get_legend_handles_labels()
                axes_ens[5].legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='center right', fontsize='small'); axes_ens[5].set_title('Selected New TD EWS')
                axes_ens[5].set_xlabel('Time')
                plt.suptitle(f"Example Run: {scenario_name}", fontsize=14); plt.tight_layout(rect=[0,0,1,0.96])
                fig_name = f"example_{scenario_name.replace(' ', '_').replace('/', '_')}.png"
                fig_ens.savefig(os.path.join(RESULTS_DIR, fig_name))
                plt.show()
                plotted_example_count[scenario_name] = True

        num_actual_collapses = sum(1 for r in all_current_collapse_results if r['breached'])
        print(f"  {scenario_name} Summary: {num_actual_collapses}/{NUM_COLLAPSE_RUNS_PER_SCENARIO} actually collapsed.")

        valid_stable_runs = [r for r in all_stable_results if not r['breached']]
        valid_collapse_runs = [r for r in all_current_collapse_results if r['breached']]
        run_counts_list.append({
            'scenario': scenario_name,
            'num_valid_collapse_runs': len(valid_collapse_runs),
            'num_valid_stable_runs': len(valid_stable_runs)
        })

        if len(valid_collapse_runs) > 0 and len(valid_stable_runs) > 0 :
            if len(valid_collapse_runs) < DEFAULT_K_FOLDS or len(valid_stable_runs) < DEFAULT_K_FOLDS :
                 print(f"  Warning: Not enough valid runs for full {DEFAULT_K_FOLDS}-Fold CV for {scenario_name}. "
                       f"(Valid Collapses: {len(valid_collapse_runs)}, Valid Stable: {len(valid_stable_runs)}). Will attempt with fewer folds if possible.")

            roc_summary, lt_summary = perform_ensemble_analysis(
                valid_collapse_runs, valid_stable_runs,
                title_suffix=f" ({scenario_name})",
                results_dir=RESULTS_DIR)

            all_roc_results[scenario_name] = roc_summary
            all_lt_results[scenario_name] = lt_summary

            if args.debug:
                roc_df = pd.DataFrame({k: {kk: (vv if not isinstance(vv, np.ndarray) else np.nan)
                                     for kk, vv in v.items()} for k, v in roc_summary.items()}).T
                roc_df.to_csv(os.path.join(RESULTS_DIR,
                                 f"roc_summary_{scenario_name.replace(' ', '_').replace('/', '_')}.csv"))

                lt_df = pd.DataFrame(lt_summary).T
                lt_df.to_csv(os.path.join(RESULTS_DIR,
                                f"lead_times_{scenario_name.replace(' ', '_').replace('/', '_')}.csv"))
        else:
            print(f"  Skipping ROC for {scenario_name} due to insufficient valid runs for CV "
                  f"(Valid Collapses: {len(valid_collapse_runs)}, Valid Stable: {len(valid_stable_runs)}).")

    print("\n--- Multi-Scenario Ensemble Simulation Study Complete (Ensemble Implemented) ---")

    df_roc_list = []
    for scen, roc_res in all_roc_results.items():
        df_tmp = pd.DataFrame(roc_res).T
        df_tmp['scenario'] = scen
        df_tmp['ews'] = df_tmp.index
        df_roc_list.append(df_tmp)
    df_roc = pd.concat(df_roc_list, ignore_index=True)

    df_lead_list = []
    for scen, lt_res in all_lt_results.items():
        df_tmp = pd.DataFrame(lt_res).T
        df_tmp['scenario'] = scen
        df_tmp['ews'] = df_tmp.index
        df_lead_list.append(df_tmp)
    df_lead = pd.concat(df_lead_list, ignore_index=True)

    auc_wide = df_roc.pivot_table(index='ews', columns='scenario', values='auc_mean')
    lead_wide = df_lead.pivot_table(index='ews', columns='scenario', values='mean')
    auc_sd_wide = df_roc.pivot_table(index='ews', columns='scenario', values='auc_std')
    lead_sd_wide = df_lead.pivot_table(index='ews', columns='scenario', values='sd')
    lead_median_wide = df_lead.pivot_table(index='ews', columns='scenario', values='median')
    lead_det_wide = df_lead.pivot_table(index='ews', columns='scenario', values='detections')
    lead_valid_wide = df_lead.pivot_table(index='ews', columns='scenario', values='num_valid_runs')
    fp_rate_wide = df_lead.pivot_table(index='ews', columns='scenario', values='stable_fp_rate')
    lead_min_wide = df_lead.pivot_table(index='ews', columns='scenario', values='min')
    lead_max_wide = df_lead.pivot_table(index='ews', columns='scenario', values='max')
    thresh_wide = df_lead.pivot_table(index='ews', columns='scenario', values='threshold')
    interp_wide = df_lead.pivot_table(index='ews', columns='scenario', values='interpretation', aggfunc='first')
    auc_rank_wide = auc_wide.rank(ascending=False, method='min').astype(int)
    tpr_wide = df_roc.pivot_table(index='ews', columns='scenario', values='tpr_mean')
    fpr_wide = df_roc.pivot_table(index='ews', columns='scenario', values='fpr_mean')
    run_counts_df = pd.DataFrame(run_counts_list)
    overall_df = pd.concat([
        df_roc.groupby('ews')['auc_mean'].agg(['mean','std']).rename(columns={'mean':'auc_mean_overall','std':'auc_sd_overall'}),
        df_lead.groupby('ews')['mean'].agg(['mean','std']).rename(columns={'mean':'lead_mean_overall','std':'lead_sd_overall'})
    ], axis=1).reset_index()

    summary_path = os.path.join(RESULTS_DIR, "EWS_summary.xlsx")

    try:
        import openpyxl       # If this succeeds we can use the default engine
        excel_engine = "openpyxl"
    except ModuleNotFoundError:
        excel_engine = "xlsxwriter"

    with pd.ExcelWriter(summary_path, engine=excel_engine) as writer:
        df_roc.to_excel(writer,  sheet_name="ROC_long",        index=False)
        df_lead.to_excel(writer, sheet_name="LeadTimes_long",  index=False)
        auc_wide.to_excel(writer,  sheet_name="AUC_mean_wide")
        auc_sd_wide.to_excel(writer, sheet_name="AUC_sd_wide")
        lead_wide.to_excel(writer, sheet_name="Lead_mean_wide")
        lead_sd_wide.to_excel(writer, sheet_name="Lead_sd_wide")
        lead_median_wide.to_excel(writer, sheet_name="Lead_median_wide")
        lead_det_wide.to_excel(writer, sheet_name="Lead_nDet_wide")
        lead_valid_wide.to_excel(writer, sheet_name="Lead_nValid_wide")
        fp_rate_wide.to_excel(writer, sheet_name="Lead_FPRate_wide")
        lead_min_wide.to_excel(writer, sheet_name="Lead_min_wide")
        lead_max_wide.to_excel(writer, sheet_name="Lead_max_wide")
        thresh_wide.to_excel(writer, sheet_name="Lead_threshold_wide")
        interp_wide.to_excel(writer, sheet_name="Lead_interp_wide")
        auc_rank_wide.to_excel(writer, sheet_name="AUC_rank_wide")
        tpr_wide.to_excel(writer, sheet_name="TPR_point_wide")
        fpr_wide.to_excel(writer, sheet_name="FPR_point_wide")
        run_counts_df.to_excel(writer, sheet_name="RunCounts", index=False)
        overall_df.to_excel(writer, sheet_name="Overall_Summary", index=False)

    summary_txt_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_txt_path, "w") as txt:
        txt.write("AUC_mean_wide\n")
        txt.write(auc_wide.to_string())
        txt.write("\n\nAUC_sd_wide\n")
        txt.write(auc_sd_wide.to_string())
        txt.write("\n\nLead_mean_wide\n")
        txt.write(lead_wide.to_string())
        txt.write("\n\nLead_sd_wide\n")
        txt.write(lead_sd_wide.to_string())
        txt.write("\n\nLead_median_wide\n")
        txt.write(lead_median_wide.to_string())
        txt.write("\n\nLead_nDet_wide\n")
        txt.write(lead_det_wide.to_string())
        txt.write("\n\nLead_nValid_wide\n")
        txt.write(lead_valid_wide.to_string())
        txt.write("\n\nLead_FPRate_wide\n")
        txt.write(fp_rate_wide.to_string())
        txt.write("\n\nLead_min_wide\n")
        txt.write(lead_min_wide.to_string())
        txt.write("\n\nLead_max_wide\n")
        txt.write(lead_max_wide.to_string())
        txt.write("\n\nLead_threshold_wide\n")
        txt.write(thresh_wide.to_string())
        txt.write("\n\nLead_interp_wide\n")
        txt.write(interp_wide.to_string())
        txt.write("\n\nAUC_rank_wide\n")
        txt.write(auc_rank_wide.to_string())
        txt.write("\n\nTPR_point_wide\n")
        txt.write(tpr_wide.to_string())
        txt.write("\n\nFPR_point_wide\n")
        txt.write(fpr_wide.to_string())
        txt.write("\n\nRunCounts\n")
        txt.write(run_counts_df.to_string(index=False))
        txt.write("\n\nOverall_Summary\n")
        txt.write(overall_df.to_string(index=False))
    try:
        import seaborn as sns  # type: ignore
    except ImportError:
        print("Install seaborn to get the AUC heat-map.")
    else:
        auc_wide_numeric = auc_wide.apply(pd.to_numeric, errors="coerce")
        plt.figure(figsize=(1 + 0.5 * len(auc_wide.columns), 0.5 + 0.4 * len(auc_wide)))
        sns.heatmap(auc_wide_numeric, fmt='.2f', cmap='viridis')
        plt.title('Mean AUC by Scenario')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'auc_heatmap.png'))
        if SHOW_PLOTS:
            plt.show()

        # --- Lead-time heat-map ---------------------------------------------
        lead_wide_numeric = lead_wide.apply(pd.to_numeric, errors="coerce")

        plt.figure(figsize=(1 + 0.5 * len(lead_wide_numeric.columns),
                            0.5 + 0.4 * len(lead_wide_numeric)))

        sns.heatmap(
            lead_wide_numeric,
            fmt='.1f',
            cmap='viridis_r'       # long lead = bright
        )

        plt.title('Mean Lead Time by Scenario (seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'leadtime_heatmap.png'))
        if SHOW_PLOTS:
            plt.show()
