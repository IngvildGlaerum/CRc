# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 09:50:19 2025

@author: rejwanfs
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI; faster/safer in loops
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa
from tqdm.auto import tqdm
import seaborn as sns
from scipy.stats import mannwhitneyu
from q_learning_optimisation_and_plotting import sim_q_learning_agents_vectorized
from multiprocessing import Pool, cpu_count

import rl_functions as rlf
rlf.set_plotting_params()
plt.ioff()
plt.close('all')

DECIMALS = 3
def qtag(q: float) -> str:
    return f"{q:.{DECIMALS}f}"

# --- user-selectable AUL / plotting subranges (inclusive) ---
# set to None to use the full grid
area_eps_range  = None      # e.g., (0.2, 0.8)
area_beta_range = None      # e.g., (5, 15)


# ---- Pool globals + initializer to avoid copying large arrays to each job ----
_ALL_TRIAL_DURS = None
_ALL_TRIAL_ERRS = None
_ERR_ONLY = True
_DT = 1.0
_OPT_N_SKIP = 0

def _init_worker(all_trial_durs, all_trial_errs, err_only, dt, opt_n_skip):
    global _ALL_TRIAL_DURS, _ALL_TRIAL_ERRS, _ERR_ONLY, _DT, _OPT_N_SKIP
    _ALL_TRIAL_DURS = all_trial_durs
    _ALL_TRIAL_ERRS = all_trial_errs
    _ERR_ONLY = bool(err_only)
    _DT = float(dt)
    _OPT_N_SKIP = int(opt_n_skip)


# ---------- persistent cache for (eps, beta, q_noise) sim means ----------
def _cell_cache_path(cache_dir, i, j):
    return os.path.join(cache_dir, f"cell_{i}_{j}.npz")

def have_cached_cell(cache_dir, i, j):
    return os.path.exists(_cell_cache_path(cache_dir, i, j))

def save_cached_cell(cache_dir, i, j, mean_actions, mean_errors):
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(_cell_cache_path(cache_dir, i, j), ma=mean_actions, me=mean_errors)

def load_cached_cell(cache_dir, i, j):
    data = np.load(_cell_cache_path(cache_dir, i, j))
    return data["ma"], data["me"]




def plot_box_with_sig(df_vals, value_col, ylabel, fname_base, save_dir):
    def _sigstars(p):
        return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'ns'

    def _add_sig_bar(ax, x1, x2, y, p, line_h, text_off):
        ax.plot([x1, x1, x2, x2], [y, y + line_h, y + line_h, y], lw=1.5, c='k')
        ax.text((x1 + x2) / 2, y + line_h + text_off, _sigstars(p),
                ha='center', va='bottom', fontsize=16)

    order = ['control', 'experimental']
    colors = {'control': '#1f77b4', 'experimental': '#d62728'}  # blue/red

    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    sns.boxplot(
        data=df_vals, x='genotype', y=value_col, order=order,
        showcaps=True, showfliers=False, whis=[0, 100],
        boxprops={'facecolor': 'none', 'edgecolor': 'k', 'linewidth': 2},
        whiskerprops={'color': 'k', 'linewidth': 2},
        medianprops={'color': 'w', 'linewidth': 2},
        ax=ax
    )

    # manually recolor
    for patch, genotype in zip(ax.patches, order):
        patch.set_facecolor(colors[genotype])
        patch.set_alpha(0.9)
        patch.set_edgecolor("k")
        patch.set_linewidth(2)

    sns.stripplot(
        data=df_vals, x='genotype', y=value_col, order=order,
        dodge=False, alpha=0.9, jitter=0.1, size=8,
        color="#e6d3a3", edgecolor="k", linewidth=0.5, ax=ax
    )

    # significance
    ctrl = df_vals.loc[df_vals['genotype'] == 'control', value_col].values
    exp  = df_vals.loc[df_vals['genotype'] == 'experimental', value_col].values
    try:
        _, p = mannwhitneyu(ctrl, exp, alternative='two-sided')
    except ValueError:
        p = 1.0

    y_min, y_max = df_vals[value_col].min(), df_vals[value_col].max()
    span = (y_max - y_min) if (y_max > y_min) else 1.0
    y_bar = y_max + 0.1 * span
    _add_sig_bar(ax, 0, 1, y_bar, p, line_h=0.05 * span, text_off=0.01 * span)

    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticklabels(['Control', 'Experimental'], fontsize=12)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{fname_base}.png", dpi=300)
    plt.savefig(f"{save_dir}/{fname_base}.svg")
    plt.close()

def select_indices_in_range(arr, rng):
    """Return sorted indices of arr within inclusive rng=(lo, hi), or full if None."""
    if rng is None:
        return np.arange(len(arr), dtype=int)
    lo, hi = float(rng[0]), float(rng[1])
    return np.where((arr >= lo) & (arr <= hi))[0]

def range_tag(rng, name, decimals=DECIMALS):
    if rng is None:
        return f"{name}=full"
    lo, hi = rng
    if name == "beta":  # usually integers
        return f"{name}={int(lo)}-{int(hi)}"
    return f"{name}={lo:.{decimals}f}-{hi:.{decimals}f}"



def seed_for_grid(i, j, q_noise):
    base = 20240915
    qk = int(round(float(q_noise) * 10000))
    return base + i * 10007 + j * 101 + qk * 1000003

# -------- vectorized loss for ALL mice at once --------
def block_mean(arr, block_size=8):
    """
    Split array into blocks of size block_size, return mean per block.
    arr: shape (T,)
    Returns: shape (n_blocks,)
    """
    n_blocks = arr.shape[0] // block_size
    arr = arr[:n_blocks * block_size].reshape(n_blocks, block_size)
    return arr.mean(axis=1)

def vectorized_loss_from_means(mean_num_actions, mean_num_errors,
                               dt, all_trial_durs, all_trial_errs,
                               block_size=8, opt_n_skip=0, err_only=True):
    """
    Compare daily means instead of trial-level values.
    Returns losses per mouse.
    """
    # Convert sim to daily means
    sim_err_blocks = block_mean(mean_num_errors, block_size=block_size)
    sim_dur_blocks = block_mean(mean_num_actions * dt, block_size=block_size)

    n_blocks = min(sim_err_blocks.shape[0], all_trial_errs.shape[1] // block_size)
    sim_err_blocks = sim_err_blocks[:n_blocks][None, :]  # (1, n_blocks)
    sim_dur_blocks = sim_dur_blocks[:n_blocks][None, :]

    real_err_blocks = np.stack(
        [block_mean(all_trial_errs[m, :], block_size)[:n_blocks] for m in range(all_trial_errs.shape[0])],
        axis=0
    )  # (n_mice, n_blocks)
    real_dur_blocks = np.stack(
        [block_mean(all_trial_durs[m, :], block_size)[:n_blocks] for m in range(all_trial_durs.shape[0])],
        axis=0
    )

    if err_only:
        loss = np.mean(np.abs(sim_err_blocks - real_err_blocks), axis=1)
    else:
        loss_d = np.mean(np.abs(sim_dur_blocks - real_dur_blocks), axis=1)
        loss_e = np.mean(np.abs(sim_err_blocks - real_err_blocks), axis=1)
        loss = loss_d + loss_e
    return loss.astype(np.float32, copy=False)


def run_grid_cell(args):
    """
    Simulate one (eps, beta, q_noise) cell and return per-mouse losses.
    Reads large arrays from worker globals to avoid per-task pickle overhead.
    """
    (i, j, eps, beta, q_noise,
     Tdata, num_agents, num_trials,
     r_treat, gamma, r_all,
     penalize_return, opt_n_skip_unused, dt_unused,
     seed) = args

    rng = np.random.default_rng(seed)
    actions_agents, errors_agents, *_ = sim_q_learning_agents_vectorized(
        num_agents=num_agents,
        num_trials=num_trials,
        r_treat=r_treat,
        epsilon=float(eps),
        beta=float(beta),
        gamma=gamma,
        r_all=r_all,
        penalize_return=penalize_return,
        opt_n_skip=_OPT_N_SKIP,
        q_noise=float(q_noise),
        rng=rng
    )
    mean_actions = actions_agents.mean(axis=0).astype(np.float64, copy=False)[:Tdata]
    mean_errors  = errors_agents.mean(axis=0).astype(np.float64, copy=False)[:Tdata]

    # Compute losses for all mice immediately
    losses_vec = vectorized_loss_from_means(
        mean_actions, mean_errors,
        dt=_DT,
        all_trial_durs=_ALL_TRIAL_DURS,
        all_trial_errs=_ALL_TRIAL_ERRS,
        opt_n_skip=_OPT_N_SKIP,
        err_only=_ERR_ONLY
    )

    return i, j, losses_vec


















if __name__ == "__main__": 
    
    # --- switches ---
    err_only = True
    ndays = 8
    resume = True
    
    # what to plot
    make_aul_plots = True
    make_example_plot = False
    
    # example params (used only if make_example_plot is True)
    example_q_noise = 0.3
    example_eps     = 0.5
    example_beta    = 10


    # --- sim params ---
    num_agents = 500
    num_trials = 8 * ndays
    gamma = 0
    r_treat = 1.0
    r_all = -0.1
    penalize_return = True
    opt_n_skip = 0
    dt = 1.0
    
    # --- parallelism ---
    n_cpus = 6   # set to how many cores you want, or None for all available

    # --- grid ---
    eps_arr = np.arange(0.1, 1.1, 0.1, dtype=np.float64)   # (10,)
    beta_arr = np.arange(1, 21, 1, dtype=np.float64)       # (20,)
    # q_noise_values = (np.arange(0, 51, 1, dtype=np.float64) / 1000.0)
    q_noise_values = [0.   , 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 
                      0.008, 0.009, 0.01 , 0.011, 0.012, 0.013, 0.014, 0.015, 
                      0.016, 0.017, 0.018, 0.019, 0.02 , 0.021, 0.022, 0.023, 
                      0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03 , 0.031,
                      0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 
                      0.04 , 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 
                      0.048, 0.049, 0.05 , 0.06 , 0.07 , 0.08 , 0.09 , 0.1  , 
                      0.15 , 0.3  ]
    
    
    # indices for the subgrid used for AUL + plotting
    eps_idx  = select_indices_in_range(eps_arr, area_eps_range)
    beta_idx = select_indices_in_range(beta_arr, area_beta_range)
    
    # sub-arrays for plotting
    eps_sub  = eps_arr[eps_idx]
    beta_sub = beta_arr[beta_idx]
    
    # --- save dirs ---
    base_dir = "./../../Output/RL/theoretical_spatial_1hist/"
    save_dir = os.path.join(
        base_dir, f"grid_search_ERR_ONLY_{ndays}_days_gamma={gamma}_mean_DAY_BASED" +
        ("_penalizedReturn" if penalize_return else "")
    )
    os.makedirs(save_dir, exist_ok=True)
   
    # plot directory based on ranges
    eps_tag  = range_tag(area_eps_range, "eps")
    beta_tag = range_tag(area_beta_range, "beta")
    plot_dir = os.path.join(
        save_dir,
        f"plots_{eps_tag}__{beta_tag}"
    )
    os.makedirs(plot_dir, exist_ok=True)



    # --- load behaviour once and stack to arrays ---
    psych_dir = "./../../Output/RL/real_behaviour_spatial"
    df = pd.read_pickle(f"{psych_dir}/pop_basic_data.pkl")
    df = df[np.isin(df.day, np.arange(1, ndays + 1, 1))]

    # Keep ordering consistent and deterministic
    mouse_meta = []  # list of dicts with (mouse_id, genotype_str, grid_file)
    series_durs = []
    series_errs = []

    for geno_id, genotype in enumerate(['control', 'experimental']):
        for mouse_id in range(1, 5):
            mdf = df[(df['animal_id'] == mouse_id) & (df.genotype == geno_id + 1)]
            if mdf.empty:
                continue
            # Each mouse contributes a 1D array over trials
            trial_durs = mdf.t_trial.values.astype(np.float64, copy=False)
            trial_errs = mdf.num_errors.values.astype(np.float64, copy=False)
            series_durs.append(trial_durs)
            series_errs.append(trial_errs)

            mouse_meta.append({
                "mouse_id": int(mouse_id),
                "genotype": genotype,
                "grid_file_tpl": os.path.join(
                    save_dir, f"grid_search_data_{mouse_id}_{genotype}_qnoise={{qtag}}.pkl"
                )
            })

    if not series_durs:
        raise RuntimeError("No mouse data found for the given filters.")
        
    # Stack into (n_mice, Tdata). Assumes equal length; if not, pad/truncate consistently.
    lengths = [len(x) for x in series_durs]
    Tdata = min(min(lengths), num_trials)  # safe cap
    all_trial_durs = np.stack([x[:Tdata] for x in series_durs], axis=0)  # (n_mice, Tdata)
    all_trial_errs = np.stack([x[:Tdata] for x in series_errs], axis=0)  # (n_mice, Tdata)
    n_mice = all_trial_durs.shape[0]
    
    if make_aul_plots:
    
        # Prebuild full grid index once
        full_grid = [(i, j, float(eps), float(beta))
                     for i, eps in enumerate(eps_arr)
                     for j, beta in enumerate(beta_arr)]
        n_cells = len(full_grid)
    
        # --- simulation cache for means (shared across mice) ---
        sim_cache = {}  # (i,j,q_noise)->(mean_actions_T, mean_errors_T), both float64
    
        summary_records = []
    
        # For each noise level, maintain a loss surface PER mouse, but fill them using one vectorized pass per cell
        for q_noise in q_noise_values:
            qtag_nm = qtag(float(q_noise))
            
            # print(f"q noise: {qtag_nm}")
    
            # Load/resume per-mouse loss arrays (or init)
            per_mouse_loss = []
            per_mouse_grid_files = []
            for m in mouse_meta:
                grid_file = m["grid_file_tpl"].format(qtag=qtag_nm)
                per_mouse_grid_files.append(grid_file)
                if resume and os.path.exists(grid_file):
                    with open(grid_file, "rb") as f:
                        gd = pickle.load(f)
                    if (np.array_equal(gd.get('eps', []), eps_arr) and
                        np.array_equal(gd.get('beta', []), beta_arr)):
                        loss_mat = gd['loss'].astype(np.float64, copy=False)
                        # In case shapes changed, coerce
                        if loss_mat.shape != (len(eps_arr), len(beta_arr)):
                            loss_mat = np.full((len(eps_arr), len(beta_arr)), np.nan, dtype=np.float64)
                    else:
                        loss_mat = np.full((len(eps_arr), len(beta_arr)), np.nan, dtype=np.float64)
                else:
                    loss_mat = np.full((len(eps_arr), len(beta_arr)), np.nan, dtype=np.float64)
                per_mouse_loss.append(loss_mat)
    
            # Which cells are still missing for at least one mouse?
            # We’ll compute sims only for cells where ANY mouse has NaN.
            any_missing = np.zeros((len(eps_arr), len(beta_arr)), dtype=bool)
            for L in per_mouse_loss:
                any_missing |= np.isnan(L)
    
            todo = [(i, j, eps, beta) for (i, j, eps, beta) in full_grid if any_missing[i, j]]
            if not todo:
                # Still do plots and summary below
                pass
            
            
            # ---------- parallel compute of missing sim means with persistent cache ----------
            jobs = []
            for (i, j, eps, beta) in todo:
                seed = seed_for_grid(i, j, float(q_noise))
                jobs.append((
                    i, j, eps, beta, q_noise,
                    Tdata, num_agents, num_trials,
                    r_treat, gamma, r_all,
                    penalize_return, opt_n_skip, dt,
                    seed
                ))
            
            if jobs:
                with Pool(
                    processes=(n_cpus or cpu_count()),
                    initializer=_init_worker,
                    initargs=(all_trial_durs, all_trial_errs, err_only, dt, opt_n_skip)
                ) as pool:
                    for i, j, losses_vec in tqdm(
                        pool.imap_unordered(run_grid_cell, jobs, chunksize=1),
                        total=len(jobs),
                        desc=f"losses q={qtag_nm}",
                        leave=True
                    ):
                        for m_idx in range(n_mice):
                            if np.isnan(per_mouse_loss[m_idx][i, j]):
                                per_mouse_loss[m_idx][i, j] = losses_vec[m_idx]


            # ---- save all per-mouse losses for this q_noise to one file ----
            q_file = os.path.join(plot_dir, f"losses_qnoise={qtag_nm}.pkl")
            loss_data = {
                "eps": eps_arr,
                "beta": beta_arr,
                "q_noise": float(q_noise),
                "losses": np.stack(per_mouse_loss, axis=0),  # shape (n_mice, len(eps), len(beta))
                "mouse_meta": mouse_meta,
            }
            with open(q_file, "wb") as f:
                pickle.dump(loss_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved losses for q_noise={qtag_nm} → {q_file}")            


    
            # ---- plotting and AUL for each mouse x q_noise ----
            # spacing for subgrid (uniform by construction)
            d_eps_sub  = float(np.diff(eps_sub).mean()) if len(eps_sub)  > 1 else 1.0
            d_beta_sub = float(np.diff(beta_sub).mean()) if len(beta_sub) > 1 else 1.0
            
            # mesh for the cropped surface
            EPS_sub, BETA_sub = np.meshgrid(eps_sub, beta_sub, indexing='ij')
            
            for m_idx, L_full in enumerate(loss_data['losses']):
                # crop to selected ranges
                L = L_full[np.ix_(eps_idx, beta_idx)]
                
                
            
                # 3D surface **of the subgrid**
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(EPS_sub, BETA_sub, L, cmap=cm.viridis,
                                       edgecolor='k', alpha=0.85)
                ax.set_xlabel('Epsilon')
                ax.set_ylabel('Beta')
                ax.set_zlabel('Loss')
                ax.set_title(
                    rf'Loss Surface (cropped) | Mouse {mouse_meta[m_idx]["mouse_id"]} ({mouse_meta[m_idx]["genotype"]}) | '
                    rf'q_noise={qtag_nm}, $\gamma={gamma}$'
                    + f"\n[{eps_tag}, {beta_tag}]"
                )
                fig.colorbar(surf, shrink=0.55, aspect=10, label='Loss')
                ax.view_init(elev=25, azim=30)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        plot_dir,
                        f"3d_loss_grid_{mouse_meta[m_idx]['mouse_id']}_{mouse_meta[m_idx]['genotype']}"
                        f"_qnoise={qtag_nm}__{eps_tag}__{beta_tag}.png"
                    ),
                    dpi=150
                )
                plt.savefig(
                    os.path.join(
                        plot_dir,
                        f"3d_loss_grid_{mouse_meta[m_idx]['mouse_id']}_{mouse_meta[m_idx]['genotype']}"
                        f"_qnoise={qtag_nm}__{eps_tag}__{beta_tag}.svg"
                    ),
                    dpi=150
                )
                plt.close(fig)
            
                # AUL computed over the selected subgrid
                area = float(np.nansum(L) * d_eps_sub * d_beta_sub)
                summary_records.append({
                    "mouse_id": mouse_meta[m_idx]["mouse_id"],
                    "genotype": mouse_meta[m_idx]["genotype"],
                    "q_noise": float(q_noise),
                    "area": area
                })
    
    
        # ----------------- summary plots (use subgrid AUL) -----------------
        # summary_df_backup = pd.DataFrame(summary_records)
        # summary_df = summary_df[summary_df.q_noise <= 0.1]
        summary_df = pd.DataFrame(summary_records)
        plt.figure()
        for (geno, mid), sdf in summary_df.sort_values('q_noise').groupby(['genotype', 'mouse_id']):
            lbl = f"{geno} {int(mid)}"
            plt.plot(sdf['q_noise'], sdf['area'], marker='o', linewidth=1.5, label=lbl, markersize=1)
        plt.xlabel("q_noise")
        plt.ylabel("Area under loss surface")
        plt.title(f"AUL vs q_noise (individuals)\n[{eps_tag}, {beta_tag}]")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"AUL_vs_qnoise_individuals__{eps_tag}__{beta_tag}.png"), dpi=600)
        plt.savefig(os.path.join(plot_dir, f"AUL_vs_qnoise_individuals__{eps_tag}__{beta_tag}.svg"))
        plt.close()
        
        plt.figure()
        colors = {"experimental": "red", "control": "blue"}
        for (geno, mid), sdf in summary_df.sort_values('q_noise').groupby(['genotype', 'mouse_id']):
            plt.plot(sdf['q_noise'], sdf['area'], marker='o', linewidth=0.5, color=colors[geno], alpha=0.5, markersize=1)
        for geno, gdf in summary_df.groupby('genotype'):
            avg = gdf.groupby('q_noise', sort=True)['area'].mean()
            plt.plot(avg.index, avg.values, linewidth=2, color=colors[geno], label=f"{geno} avg")
        plt.xlabel("q_noise")
        plt.ylabel("Area under loss surface")
        plt.title(f"AUL vs q_noise (genotype averages + individuals)\n[{eps_tag}, {beta_tag}]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"AUL_vs_qnoise_genotype__{eps_tag}__{beta_tag}.png"), dpi=600)
        plt.savefig(os.path.join(plot_dir, f"AUL_vs_qnoise_genotype__{eps_tag}__{beta_tag}.svg"))
        plt.close()
        
        # ----------------- NEW: box plot of AUL for q_noise = 0 -----------------
        aul_q0_df = summary_df[summary_df['q_noise'] == 0.0].copy()
        if not aul_q0_df.empty:
            plot_box_with_sig(
                df_vals=aul_q0_df,
                value_col="area",
                ylabel="Area under loss surface (q_noise=0)",
                fname_base="AUL_q0_boxplot",
                save_dir=plot_dir
            )
        
        aul_noisy_df = summary_df[summary_df['q_noise'] == 0.027].copy()
        if not aul_noisy_df.empty:
            plot_box_with_sig(
                df_vals=aul_noisy_df,
                value_col="area",
                ylabel="Area under loss surface (q_noise=0.027)",
                fname_base="AUL_noisy_boxplot",
                save_dir=plot_dir
            )
            
            
        # ----------------- NEW: box plot of q_noise that minimizes AUL -----------------
        min_qnoise_records = []
        for (geno, mid), gdf in summary_df.groupby(['genotype', 'mouse_id']):
            # take q_noise value that gives minimum area
            best_row = gdf.loc[gdf['area'].idxmin()]
            min_qnoise_records.append({
                "mouse_id": mid,
                "genotype": geno,
                "best_q_noise": best_row['q_noise']
            })
        min_qnoise_df = pd.DataFrame(min_qnoise_records)
    
        if not min_qnoise_df.empty:
            plot_box_with_sig(
                df_vals=min_qnoise_df,
                value_col="best_q_noise",
                ylabel="q_noise minimizing AUL",
                fname_base="best_qnoise_boxplot",
                save_dir=plot_dir
            )

            
    
    # ----------------- example plot: sim vs real errors -----------------
    if make_example_plot:
        print(f"Generating example plot: q_noise={example_q_noise}, "
              f"eps={example_eps}, beta={example_beta}")

        # run sim once
        rng = np.random.default_rng(12345)
        actions_agents, errors_agents, *_ = sim_q_learning_agents_vectorized(
            num_agents=num_agents,
            num_trials=num_trials,
            r_treat=r_treat,
            epsilon=float(example_eps),
            beta=float(example_beta),
            gamma=gamma,
            r_all=r_all,
            penalize_return=penalize_return,
            opt_n_skip=opt_n_skip,
            q_noise=float(example_q_noise),
            rng=rng
        )
        mean_errors = errors_agents.mean(axis=0)[:Tdata]

        # compute per-mouse loss values
        losses_vec = vectorized_loss_from_means(
            mean_num_actions=np.zeros_like(mean_errors),  # dummy, since err_only=True
            mean_num_errors=mean_errors,
            dt=dt,
            all_trial_durs=all_trial_durs,
            all_trial_errs=all_trial_errs,
            opt_n_skip=opt_n_skip,
            err_only=True
        )

        # plot in a 2x4 grid
        fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)
        axes = axes.flatten()
        colors = {"experimental": "red", "control": "blue"}

        for m_idx, m in enumerate(mouse_meta):
            ax = axes[m_idx]
            trial_errs = all_trial_errs[m_idx, :Tdata]

            ax.plot(trial_errs, color=colors[m["genotype"]], alpha=0.7, label="real mouse")
            ax.plot(mean_errors, color="black", linewidth=1.5, label="RL agents")

            loss_val = losses_vec[m_idx]
            ax.set_title(f"Mouse {m['mouse_id']} ({m['genotype']})\nLoss={loss_val:.3f}")
            ax.grid(alpha=0.3)

        # hide empty axes if <8 mice
        for j in range(len(mouse_meta), len(axes)):
            axes[j].axis("off")

        # global labels
        fig.supxlabel("Trial")
        fig.supylabel("# errors")

        # overall title and legend
        fig.suptitle(f"Example errors | eps={example_eps}, beta={example_beta}, q_noise={example_q_noise}")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")

        plt.tight_layout(rect=[0, 0, 0.95, 0.93])
        plt.savefig(os.path.join(
            plot_dir,
            f"example_errors_grid_eps={example_eps}_beta={example_beta}_qnoise={qtag(example_q_noise)}.png"
        ), dpi=600)
        plt.close()




