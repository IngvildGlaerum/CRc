# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 15:10:50 2025

@author: rejwanfs

Currently, this takes the mean over 8 trials, then averages this over
the 500 agents.
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rl_functions as rlf
from scipy.optimize import differential_evolution
import hashlib
from numba import njit
from scipy.stats import sem, mannwhitneyu



########### set plotting parameters ############
rlf.set_plotting_params()
plt.ioff()
plt.close('all')

plt.rcParams['xtick.bottom'] = True
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = False
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
################################################

# Maze Constants
NUM_ARMS = 8
BINS_PER_ARM = 1
TOTAL_STATES = 1 + NUM_ARMS * BINS_PER_ARM  # 1 center + arms * bins
NUM_ACTIONS = 9  # 8 directions + 1 stay

# Maze relevant dimensions
arm_length = 35 #in cm

# genotype data
genotype_map = {2: 'cre+', 1: 'cre-'}

def _sigstars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'ns'

def plot_box_with_sig(df_vals, value_col, ylabel, fname_base, save_dir):
    def _sigstars(p):
        return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'ns'

    def _add_sig_bar(ax, x1, x2, y, p, line_h, text_off):
        ax.plot([x1, x1, x2, x2], [y, y + line_h, y + line_h, y], lw=1.5, c='k')
        ax.text((x1 + x2) / 2, y + line_h + text_off, _sigstars(p),
                ha='center', va='bottom', fontsize=16)

    order = ['control', 'experimental']
    colors = {'control': '#1f77b4', 'experimental': '#d62728'}  # blue, red

    plt.figure(figsize=(5, 5))
    ax = plt.gca()

    # Draw the boxplot
    sns.boxplot(
        data=df_vals, x='genotype', y=value_col, order=order,
        showcaps=True, showfliers=False,
        boxprops={'facecolor': 'none', 'edgecolor': 'k', 'linewidth': 2},
        whiskerprops={'color': 'k', 'linewidth': 2},
        medianprops={'color': 'w', 'linewidth': 2},
        ax=ax
    )

    # FIX: recolor the drawn boxes (now in ax.patches, not ax.artists)
    for patch, genotype in zip(ax.patches, order):
        patch.set_facecolor(colors[genotype])
        patch.set_alpha(0.85)
        patch.set_edgecolor("k")
        patch.set_linewidth(2)

    # Overlay scatter points (beige/gold)
    sns.stripplot(
        data=df_vals, x='genotype', y=value_col, order=order,
        dodge=False, alpha=0.9, jitter=0.1, size=8,
        color="#e6d3a3", edgecolor="k", linewidth=0.5, ax=ax
    )

    # Mann–Whitney U test
    ctrl = df_vals.loc[df_vals['genotype'] == 'control', value_col].values
    exp  = df_vals.loc[df_vals['genotype'] == 'experimental', value_col].values
    try:
        _, p = mannwhitneyu(ctrl, exp, alternative='two-sided')
    except ValueError:
        p = 1.0

    # Significance bar
    y_min, y_max = df_vals[value_col].min(), df_vals[value_col].max()
    span = (y_max - y_min) if (y_max > y_min) else 1.0
    y_bar = y_max + 0.1 * span
    _add_sig_bar(ax, 0, 1, y_bar, p, line_h=0.05 * span, text_off=0.01 * span)

    # Labels and style
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticklabels(['Control', 'Experimental'], fontsize=12)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{fname_base}.png", dpi=300)
    plt.savefig(f"{save_dir}/{fname_base}.svg")
    plt.close()



def _seed_from_params(x, base=12345):
    """
    Deterministic seed from continuous params x for reproducible stochastic loss.
    Uses rounded params and SHA256 to avoid Python's randomized hash().
    """
    x_round = np.round(np.asarray(x, dtype=float), 6)
    h = hashlib.sha256(repr(x_round.tolist()).encode("utf-8")).hexdigest()
    return (base + int(h[:8], 16)) % (2**31 - 1)

def de_objective(x, dt, trial_durs, trial_errs, num_agents, num_trials,
                 r_treat, r_all, gamma, penalize_return, opt_n_skip,
                 err_only, q_noise, seed_base):
    """
    Top-level wrapper for loss_function so SciPy can pickle it when using workers>1.
    """
    # derive a deterministic RNG per (eps,beta)
    seed = _seed_from_params(x, base=seed_base)
    rng = np.random.default_rng(seed)
    return loss_function(
        x, dt, trial_durs, trial_errs,
        num_agents, num_trials, r_treat, r_all, gamma,
        penalize_return=penalize_return,
        opt_n_skip=opt_n_skip,
        err_only=err_only,
        q_noise=q_noise,
        rng=rng
    )


# ---- Helpers for valid actions and transitions ----
def get_valid_actions(prev_state, state):
    if state == 0:
        return list(range(NUM_ACTIONS))
    arm = (state - 1) // BINS_PER_ARM
    bin_num = (state - 1) % BINS_PER_ARM + 1
    valid_actions = [8]  # stay
    if bin_num < BINS_PER_ARM:
        valid_actions.append(arm)  # forward
    valid_actions.append((arm + 4) % 8)  # back
    return valid_actions

def get_next_state_scalar(state, action):
    if action == 8:
        return state
    if state == 0:
        return 1 + action * BINS_PER_ARM
    arm = (state - 1) // BINS_PER_ARM
    bin_num = (state - 1) % BINS_PER_ARM + 1
    if action == arm and bin_num < BINS_PER_ARM:
        return state + 1
    elif action == (arm + 4) % 8:
        return 0 if bin_num == 1 else state - 1
    return state

# Precompute valid action masks and next-state lookups for vectorisation
VALID_MASK = np.zeros((TOTAL_STATES, TOTAL_STATES, NUM_ACTIONS), dtype=bool)
for ps in range(TOTAL_STATES):
    for s in range(TOTAL_STATES):
        acts = get_valid_actions(ps, s)
        VALID_MASK[ps, s, acts] = True

NEXT_STATE = np.zeros((TOTAL_STATES, NUM_ACTIONS), dtype=np.int32)
for s in range(TOTAL_STATES):
    for a in range(NUM_ACTIONS):
        NEXT_STATE[s, a] = get_next_state_scalar(s, a)

@njit(fastmath=True)
def _update_q_numba(agent_idx, prev_states, states, actions, rewards, next_states,
                    epsilon, gamma, Q_flat, VM_flat, S, A):
    """
    Numba version of the per-step Q update. Loops over active agents.
    Shapes:
      agent_idx, prev_states, states, actions, rewards, next_states -> (Na,)
      Q_flat: (num_agents, S*S, A)   float32
      VM_flat: (S*S, A)              bool
    """
    Na = actions.shape[0]
    for k in range(Na):
        a_idx = agent_idx[k]
        ps = prev_states[k]
        s  = states[k]
        a  = actions[k]
        ns = next_states[k]

        pair_cur  = ps * S + s
        pair_next = s * S + ns

        # greedy next action under valid mask
        best_val = -1.0e30  # ~ -inf
        best_a = 0
        for aj in range(A):
            if VM_flat[pair_next, aj]:
                val = Q_flat[a_idx, pair_next, aj]
                if val > best_val:
                    best_val = val
                    best_a = aj

        td_target = rewards[k] + gamma * Q_flat[a_idx, pair_next, best_a]
        q_cur     = Q_flat[a_idx, pair_cur, a]
        Q_flat[a_idx, pair_cur, a] = q_cur + epsilon * (td_target - q_cur)


def update_q_vectorized(agent_idx, prev_states, states, actions, rewards, next_states,
                        epsilon, gamma, Q_flat, VM_flat, S):
    """
    Thin wrapper that calls the Numba kernel. Keeps your original function name/signature.
    """
    _update_q_numba(
        agent_idx.astype(np.int32, copy=False),
        prev_states.astype(np.int32, copy=False),
        states.astype(np.int32, copy=False),
        actions.astype(np.int32, copy=False),
        rewards.astype(np.float32, copy=False),
        next_states.astype(np.int32, copy=False),
        np.float32(epsilon),
        np.float32(gamma),
        Q_flat,           # float32 (C-contiguous)
        VM_flat,          # bool
        np.int32(TOTAL_STATES),
        np.int32(NUM_ACTIONS),
    )




def sample_softmax_actions(Q_slice, valid_mask, beta, rng):
    """
    Q_slice: (N, NUM_ACTIONS)
    valid_mask: (N, NUM_ACTIONS)
    Returns sampled actions (N,)
    """
    logits = beta * np.where(valid_mask, Q_slice, -np.inf)
    # Gumbel-max sampling for masked softmax
    g = rng.gumbel(size=logits.shape).astype(np.float32, copy=False)
    chosen = np.argmax(logits + g, axis=1)
    return chosen

def sim_q_learning_agents_vectorized(num_agents, num_trials, r_treat, epsilon, beta, gamma,
                                     r_all, penalize_return, opt_n_skip, q_noise, rng):
    """
    Vectorised simulation over agents.

    Returns (all arrays are shape (num_agents, num_trials)):
        actions_agents : steps taken per agent per trial
        errors_agents  : total errors per agent per trial (entries - 3)
        err_r1_agents  : errors before first treat
        err_r2_agents  : errors between first–second treat
        err_r3_agents  : errors between second–third treat
    """
    S, A = TOTAL_STATES, NUM_ACTIONS

    # Q per agent (float32)
    Q = np.zeros((num_agents, S, S, A), dtype=np.float32)

    if penalize_return:
        large_negative = np.float32(-10.0)
        for prev_state in range(1, S):  # exclude center as "prev"
            prev_arm = (prev_state - 1) // BINS_PER_ARM
            for state in range(1, S):
                curr_arm = (state - 1) // BINS_PER_ARM
                if prev_arm != curr_arm:
                    return_action = prev_arm
                    Q[:, prev_state, state, return_action] = large_negative

    # Treat locations (state indices)
    treat_states = np.array([3 * BINS_PER_ARM, 4 * BINS_PER_ARM, 7 * BINS_PER_ARM], dtype=np.int32)
    num_treats   = treat_states.size

    # Flattened views
    Q_flat  = Q.reshape(num_agents, S * S, A)      # (N, S*S, A) float32
    VM_flat = VALID_MASK.reshape(S * S, A)         # (S*S, A)   bool

    # --- NEW: precompute valid (pair, action) indices once ---
    # Only center has 9 actions; arms have 2 (stay/back). With S=9 this is 225 valid of 729 total.
    valid_pair_idx, valid_act_idx = np.nonzero(VM_flat)  # shapes (M,), (M,)
    M = valid_pair_idx.size

    MAX_ACTIONS = 500

    # Outputs (int32)
    actions_agents = np.zeros((num_agents, num_trials), dtype=np.int32)
    errors_agents  = np.zeros((num_agents, num_trials), dtype=np.int32)
    err_r1_agents  = np.zeros((num_agents, num_trials), dtype=np.int32)
    err_r2_agents  = np.zeros((num_agents, num_trials), dtype=np.int32)
    err_r3_agents  = np.zeros((num_agents, num_trials), dtype=np.int32)

    # phase start arms (blocks of 64 trials)
    start_arm_arr = [2, 5]

    for episode in range(num_trials):
        # Segmented-error bookkeeping
        seg_idx = np.ones(num_agents, dtype=np.int32)  # 1..3
        err1    = np.zeros(num_agents, dtype=np.int32)
        err2    = np.zeros(num_agents, dtype=np.int32)
        err3    = np.zeros(num_agents, dtype=np.int32)

        start_arm   = start_arm_arr[episode // 64]
        START_STATE = np.int32(start_arm * BINS_PER_ARM)

        prev_states = np.full(num_agents, START_STATE, dtype=np.int32)
        states      = np.full(num_agents, START_STATE, dtype=np.int32)

        # collected treats mask per agent
        collected = np.zeros((num_agents, num_treats), dtype=bool)

        done   = np.zeros(num_agents, dtype=bool)
        steps  = np.zeros(num_agents, dtype=np.int32)
        entries= np.zeros(num_agents, dtype=np.int32)

        for _ in range(MAX_ACTIONS):
            if np.all(done):
                break

            active_idx = np.flatnonzero(~done)
            if active_idx.size == 0:
                break

            ps = prev_states[active_idx]
            s  = states[active_idx]

            # Flattened pair index for (prev,current)
            pair_cur = ps * S + s  # int32

            # Gather Q for action selection
            Q_slice = Q_flat[active_idx, pair_cur, :]   # (Na, A) float32
            vm      = VM_flat[pair_cur]                 # (Na, A) bool

            # Sample actions (masked softmax via Gumbel-max)
            actions     = sample_softmax_actions(Q_slice, vm, beta, rng).astype(np.int32, copy=False)
            next_states = NEXT_STATE[s, actions]        # (Na,) int32

            # Reward & treat bookkeeping
            is_treat  = (next_states[:, None] == treat_states[None, :])  # (Na, num_treats)
            coll_sub  = collected[active_idx]
            new_treat = np.any(is_treat & (~coll_sub), axis=1)
            rewards   = np.where(new_treat, np.float32(r_treat), np.float32(r_all)).astype(np.float32, copy=False)

            # Q update
            update_q_vectorized(active_idx, ps, s, actions, rewards, next_states,
                                epsilon, gamma, Q_flat, VM_flat, S)

            # --- NEW: add Gaussian noise ONLY on valid state–action entries ---
            if q_noise > 0:
                # Draw in float32 to avoid upcasts; shape (Na, M)
                noise = rng.standard_normal((active_idx.size, M), dtype=np.float32)
                # Scatter-add onto valid (pair, action) slices for each active agent
                Q_flat[active_idx[:, None], valid_pair_idx[None, :], valid_act_idx[None, :]] += (
                    np.float32(q_noise) * noise
                )

            # Mark collected treats
            if np.any(is_treat):
                coll_sub |= is_treat
                collected[active_idx] = coll_sub

            # Entries/errors bookkeeping
            curr_arm_idx = (next_states - 1) // BINS_PER_ARM
            prev_arm_idx = (s          - 1) // BINS_PER_ARM
            delta        = curr_arm_idx - prev_arm_idx
            entry_step   = (delta > 0)

            if np.any(entry_step):
                entry_full = active_idx[entry_step]
                not_new    = ~new_treat[entry_step]
                if np.any(not_new):
                    idxs = entry_full[not_new]
                    add1 = (seg_idx[idxs] == 1)
                    add2 = (seg_idx[idxs] == 2)
                    add3 = (seg_idx[idxs] == 3)
                    if np.any(add1): err1[idxs[add1]] += 1
                    if np.any(add2): err2[idxs[add2]] += 1
                    if np.any(add3): err3[idxs[add3]] += 1

            # Advance segment on NEW treat
            if np.any(new_treat):
                got_full = active_idx[new_treat]
                seg_idx[got_full] = np.minimum(seg_idx[got_full] + 1, 3)

            # Step the active set
            entries[active_idx]      += entry_step.astype(np.int32, copy=False)
            prev_states[active_idx]   = s
            states[active_idx]        = next_states
            steps[active_idx]        += 1

            # Finished if all treats collected
            done_now = np.all(collected[active_idx], axis=1)
            if np.any(done_now):
                done[active_idx[done_now]] = True

        # Finalise per-trial metrics
        actions_agents[:, episode] = steps
        errors_agents[:, episode]  = (entries - 3)
        err_r1_agents[:, episode]  = err1
        err_r2_agents[:, episode]  = err2
        err_r3_agents[:, episode]  = err3

    return actions_agents, errors_agents, err_r1_agents, err_r2_agents, err_r3_agents


def block_mean_agents(arr, block_size=8):
    """
    Mean per agent over blocks of trials, then average across agents.

    arr: shape (num_agents, num_trials)
    Returns: shape (n_blocks,) = mean errors per day across agents
    """
    n_agents, n_trials = arr.shape
    n_blocks = n_trials // block_size
    arr = arr[:, :n_blocks * block_size]
    arr = arr.reshape(n_agents, n_blocks, block_size)
    daily_means = arr.mean(axis=2)          # (num_agents, n_blocks)
    return daily_means.mean(axis=0)         # (n_blocks,)




def loss_function(params, dt, trial_durs, trial_errs, num_agents, num_trials,
                  r_treat, r_all, gamma, penalize_return=False,
                  opt_n_skip=0, err_only=False, q_noise=0.0, rng=None):
    eps, beta = params

    if rng is None:
        rng = np.random.default_rng(12345)

    (_, errors_agents, _, _, _) = sim_q_learning_agents_vectorized(
        num_agents=num_agents,
        num_trials=num_trials,
        r_treat=r_treat,
        epsilon=eps,
        beta=beta,
        gamma=gamma,
        r_all=r_all,
        penalize_return=penalize_return,
        opt_n_skip=opt_n_skip,
        q_noise=q_noise,
        rng=rng,
    )
    
    # --- NEW: aggregate per agent into daily blocks (8 trials), then average across agents ---
    sim_err_blocks = block_mean_agents(errors_agents, block_size=8)
    
    # Real data: group trials into days (8 trials per day)
    n_days = len(trial_errs) // 8
    trial_errs_clipped = trial_errs[:n_days * 8].reshape(n_days, 8)
    real_err_blocks = trial_errs_clipped.mean(axis=1)   # <-- also use mean here


    # Match length
    min_len = min(len(sim_err_blocks), len(real_err_blocks))
    loss_err = np.mean(np.abs(sim_err_blocks[:min_len] - real_err_blocks[:min_len]))

    if not err_only:
        raise NotImplementedError("Durations removed; only error fitting supported now")

    return loss_err




def de_callback(xk, convergence):
    eps, beta = xk
    print(f"[DE] eps={eps:.3f}, beta={beta:.3f}, conv={convergence:.4f}", flush=True)
    return False


# ------------------ MAIN ------------------
if __name__ == "__main__":

    plt.ioff()

    # switches
    fit = True
    err_only = True
    plot = True

    ndays = 14
    
    n_cpus = 6

    # sim size
    num_agents = 500
    num_trials = 8 * ndays

    # Q-learning parameters (fixed)
    gamma = 0
    r_treat = 1.0
    r_all = -0.1

    # penalty flag
    penalize_return = True

    # skip initial trials in optimisation
    opt_n_skip = 0

    # NEW: Q-table noise hyperparameter
    # 0.0 -> identical to previous behaviour; larger -> more unstable
    q_noise = 0.0  # CHANGE this for experiments; also encoded in save path

    # save directory now includes q_noise
    base_dir = "./../../Output/RL/theoretical_spatial_1hist/"
    sub_dir = (
        f"r_bait={r_treat}, r_else={r_all}, 1 bins, opt after {opt_n_skip} trials, mean_DAY-BASED/"
        + (f"psych_fits_ERR_ONLY_{ndays}_days_gamma={gamma}" if err_only
           else f"psychometric_fits_{ndays}_days_gamma={gamma}")
        + ("_penalizedReturn" if penalize_return else "")
        + f"_qnoise={q_noise}"
    )
    save_dir = os.path.join(base_dir, sub_dir)
    os.makedirs(save_dir, exist_ok=True)

    # load behaviour
    psych_dir = "./../../Output/RL/real_behaviour_spatial"
    df = pd.read_pickle(f"{psych_dir}/pop_basic_data.pkl")
    df = df[np.isin(df.day, np.arange(1, ndays + 1, 1))]

    rng_master = np.random.default_rng(20240914)

    if fit:
        results_list = []

        for geno_id, genotype in enumerate(['control', 'experimental']):
            for mouse_id in range(1, 5):
                mouse_df = df[(df['animal_id'] == mouse_id) & (df.genotype == geno_id + 1)]

                trial_durs = mouse_df.t_trial.values
                trial_errs = mouse_df.num_errors.values

                # Until avg speed is wired in, keep dt=1 like before
                dt = 1.0
                print("UNTIL AVG SPEED OBTAINED, dt DEFINED AS 1.")

                # bounds for (epsilon, beta)
                bounds = [(0.001, 1.0), (0.1, 20.0)]
                
                # fixed base so results are repeatable across processes
                seed_base = 20240914
                
                result_de = differential_evolution(
                    de_objective,
                    bounds,
                    args=(
                        dt, trial_durs.values if hasattr(trial_durs, "values") else np.asarray(trial_durs),
                        trial_errs.values if hasattr(trial_errs, "values") else np.asarray(trial_errs),
                        num_agents, num_trials, r_treat, r_all, gamma,
                        penalize_return, opt_n_skip, err_only, q_noise, seed_base
                    ),
                    strategy="best2bin",
                    tol=0.1,
                    popsize=30,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    init='sobol',
                    updating='deferred',
                    workers=n_cpus,            # parallel across cores
                    callback=de_callback,  # your existing top-level callback
                    polish=False
                )
                
                """************ TRY THIS TO SLOW CONVERGENCE ************"""
                #     strategy="rand2bin",
                #     tol=0.1, # try 0.025 next if still no good
                #     popsize=30, # try 60 next time
                #     mutation=(0.7, 1.9),
                #     recombination=0.7, # KEY: TRY 0.2 AND KEEP OTHERS UNCHANGED
                #     init='sobol',
                #     updating='deferred',
                #     workers=n_cpus,            # parallel across cores
                #     callback=de_callback,  # your existing top-level callback
                #     polish=False,
                #     maxiter=100
                # )
                """********************************************************"""

                de_eps, de_beta = result_de.x
                print(f"Mouse {mouse_id} ({genotype}) | DE Optimized β: {de_beta:.4f}, "
                      f"DE epsilon: {de_eps:.4f} | Loss: {result_de.fun:.4f}")

                results_list.append({
                    "mouse_id": mouse_id,
                    "genotype": genotype,
                    "best_epsilon_DE": float(de_eps),
                    "best_beta_DE": float(de_beta),
                    "loss_DE": float(result_de.fun),
                    "q_noise": float(q_noise)
                })

        results_df = pd.DataFrame(results_list)
        results_df.to_pickle(f"{save_dir}/optimized_mouse_parameters.pkl")

    if plot:
        results_path = f"{save_dir}/optimized_mouse_parameters.pkl"
        if not fit:
            print("Loading precomputed fit results...")
            results_df = pd.read_pickle(results_path)

        for geno_id, genotype in enumerate(['control', 'experimental']):
            for mouse_id in range(1, 5):
                mouse_df = df[(df['animal_id'] == mouse_id) & (df.genotype == geno_id + 1)]
                real_num_errors = mouse_df.num_errors.values

                row = results_df[(results_df.mouse_id == mouse_id) & (results_df.genotype == genotype)]
                eps = float(row.best_epsilon_DE.values[0])
                beta = float(row.best_beta_DE.values[0])

                # --- Run sim ---
                (_, errors_agents, _, _, _) = sim_q_learning_agents_vectorized(
                    num_agents=num_agents,
                    num_trials=num_trials,
                    r_treat=r_treat,
                    epsilon=eps,
                    beta=beta,
                    gamma=gamma,
                    r_all=r_all,
                    penalize_return=penalize_return,
                    opt_n_skip=opt_n_skip,
                    q_noise=q_noise,
                    rng=np.random.default_rng(777 + mouse_id + 10 * geno_id)
                )

                # --- Aggregate into daily blocks (8 trials) ---
                len_day = (min(len(real_num_errors), errors_agents.shape[1]) // 8) * 8
                if len_day < 8:
                    continue  # not enough data for a full day

                n_days = len_day // 8
                days = np.arange(1, n_days + 1)

                # Real daily errors (mean per trial within a day)
                real_blocks = real_num_errors[:len_day].reshape(-1, 8).mean(axis=1)
                
                # RL: mean per agent per day, then mean ± SEM across agents
                rl_day_agents = errors_agents[:, :len_day].reshape(errors_agents.shape[0], -1, 8).mean(axis=2)
                sim_blocks = rl_day_agents.mean(axis=0)
                sem_blocks = sem(rl_day_agents, axis=0)
                
                # --- Plot ---
                fig, ax = plt.subplots()
                ax.plot(days, real_blocks, 'o-', color='red', linewidth=3, label="Real errors (mean/trial)")
                ax.errorbar(days, sim_blocks, yerr=sem_blocks,
                            fmt='s-', color='orange', linewidth=3, capsize=3,
                            label="RL errors (mean±SEM per trial)")
                ax.set_title(
                    f"{mouse_id} ({genotype}) | "
                    + rf"$\beta_{{opt}}={beta:.2f},\ \epsilon_{{opt}}={eps:.2f},\ q_{{noise}}={q_noise:.2f}$"
                )
                ax.set_xlabel("Day (8 trials)")
                ax.set_ylabel("Mean number of errors per trial")
                ax.legend()


                fig.tight_layout()
                plt.savefig(f"{save_dir}/errors_{genotype}_{mouse_id}.png", dpi=300)
                plt.savefig(f"{save_dir}/errors_{genotype}_{mouse_id}.svg")
                plt.close()

        # ---- Summary scatter ----
        results_df['genotype_presentation'] = [g for g in results_df.genotype]
        palette = {"experimental": "red", "control": "blue"}
        plt.figure(figsize=(14, 14))
        sns.scatterplot(
            data=results_df,
            x='best_epsilon_DE',
            y='best_beta_DE',
            hue='genotype_presentation',
            palette=palette,
            s=100
        )
        for _, row in results_df.iterrows():
            plt.text(row['best_epsilon_DE'] + 0.005,
                     row['best_beta_DE'] + 0.005,
                     str(int(row['mouse_id'])), fontsize=14)
        plt.xlabel('Best Epsilon (DE)')
        plt.ylabel('Best Beta (DE)')
        plt.title(rf'Best $\epsilon$ & $\beta$ by Genotype ({ndays} days) | $\gamma={gamma:.2f}$', fontsize=20)
        plt.legend(title='Genotype')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/summary.png", dpi=300)
        plt.savefig(f"{save_dir}/summary.svg")
        plt.close()
        
        
        # ---- Genotype box plots with significance (epsilon & beta) ----

        
       
        
        # Prepare tidy frames and plot
        df_eps  = results_df[['genotype', 'best_epsilon_DE']].rename(columns={'best_epsilon_DE': 'epsilon'})
        df_beta = results_df[['genotype', 'best_beta_DE']].rename(columns={'best_beta_DE': 'beta'})
        
        plot_box_with_sig(df_eps, 'epsilon', 'Best ε (epsilon)', 'box_epsilon', save_dir)
        plot_box_with_sig(df_beta, 'beta', 'Best β (beta)', 'box_beta', save_dir)

        
        # (Optional) combined figure with both subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, (df_vals, valcol, label) in zip(
            axes, [(df_eps, 'epsilon', 'Epsilon (ε)'), (df_beta, 'beta', 'Beta (β)')]
        ):
            order = ['control', 'experimental']
            sns.boxplot(
                data=df_vals, x='genotype', y=valcol, order=order,
                whis=1.5, showcaps=True, boxprops={'facecolor': 'none'}, medianprops={'linewidth': 2}, ax=ax
            )
            sns.stripplot(
                data=df_vals, x='genotype', y=valcol, order=order,
                dodge=False, alpha=0.85, jitter=0.15, size=7, ax=ax
            )
            ctrl = df_vals.loc[df_vals['genotype'] == 'control', valcol].values
            exp  = df_vals.loc[df_vals['genotype'] == 'experimental', valcol].values
            try:
                _, p = mannwhitneyu(ctrl, exp, alternative='two-sided')
            except ValueError:
                p = 1.0
            y_min, y_max = df_vals[valcol].min(), df_vals[valcol].max()
            span = (y_max - y_min) if (y_max > y_min) else 1.0
            y_bar = y_max + 0.06 * span
            # draw significance on this axis
            axes_idx = [0, 1]  # x positions for control/experimental
            ax.set_title(f'{label} (p={p:.3g})')
            # use the same helper but on this axis
            ax.plot([axes_idx[0], axes_idx[0], axes_idx[1], axes_idx[1]],
                    [y_bar, y_bar + 0.03 * span, y_bar + 0.03 * span, y_bar], lw=1.5, c='k')
            ax.text(0.5, y_bar + 0.04 * span, _sigstars(p), ha='center', va='bottom', fontsize=14)
            ax.set_xlabel('')
            ax.set_xticklabels(['Control', 'Experimental'])
        fig.suptitle('Genotype comparisons for ε and β', y=1.02)
        fig.tight_layout()
        fig.savefig(f"{save_dir}/box_epsilon_beta_combined.png", dpi=300)
        fig.savefig(f"{save_dir}/box_epsilon_beta_combined.svg")
        plt.close()



