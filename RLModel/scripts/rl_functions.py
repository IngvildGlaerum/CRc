# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:05:13 2025

@author: Rejwan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk  # Used to detect screen size
import matplotlib as mpl
import sys
sys.path.append("..")
import utility_functions.utility_functions as uf
from matplotlib.patches import Rectangle
from scipy.special import logsumexp

def get_entry_df(sess_data):
    
    df_pos = sess_data['pos']
    
    trials_df = df_pos[df_pos.trial_idx != -1]
    zone_diffs = np.diff(trials_df.zone)
    change_idxs = np.where(zone_diffs != 0)[0]
    entry_idxs = np.concatenate([[0], change_idxs+1])
    entry_df = trials_df.iloc[entry_idxs,:]
    
    return entry_df

def get_full_entry_df(trials_df):
    
    zone_diffs = np.diff(trials_df.zone)
    change_idxs = np.where(zone_diffs != 0)[0]
    entry_idxs = np.concatenate([[0], change_idxs+1])
    entry_df = trials_df.iloc[entry_idxs,:]
    
    end_t_arr = np.full(len(entry_df), np.nan)
    dur_arr = np.full(len(entry_df), np.nan)
    entry_df.insert(1, "end_frame_t", end_t_arr)
    entry_df.insert(2, "zone_dur", dur_arr)
    for t_idx in trials_df.trial.unique().astype(int):
        trial_end_t = \
            trials_df[trials_df.trial == t_idx].time.iloc[-1]
        trial_entry_df = entry_df[entry_df.trial == t_idx]
        trial_end_frames = np.concatenate([trial_entry_df.time.iloc[1:],
                                           [trial_end_t]])
        entry_df.loc[trial_entry_df.index, 'end_frame_t'] = trial_end_frames
        entry_df.loc[trial_entry_df.index, 'zone_dur'] = \
            trial_end_frames - trial_entry_df.time
        
    return entry_df

# Function to get screen DPI and resolution
def get_screen_dpi():
    root = tk.Tk()
    dpi = root.winfo_fpixels('1i')  # Get pixels per inch
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return dpi, screen_width, screen_height

def set_plotting_params(BASE_DPI=96):

    # Get screen properties
    dpi, screen_width, screen_height = get_screen_dpi()
    
    # Define a base DPI for standardization
    # BASE_DPI = 100  # Adjust sizes based on this reference DPI
    scaling_factor = dpi / BASE_DPI  # Scale elements dynamically
    
    # Dynamically adjust rcParams based on scaling factor
    mpl.rcParams.update({
        'figure.figsize': (20 * scaling_factor, 11.25 * scaling_factor), # Scale figure
        'figure.titlesize': 20 * scaling_factor, # scale suptitle size
        'axes.titlesize': 20 * scaling_factor,  # Scale title size
        'axes.labelsize': 15 * scaling_factor,  # Scale x/y label size
        'xtick.labelsize': 15 * scaling_factor,  # Scale tick labels
        'ytick.labelsize': 15 * scaling_factor,
        'legend.fontsize': 11 * scaling_factor,  # Scale legend size
        'lines.linewidth': 2 * scaling_factor,  # Scale line width
        'grid.linewidth': 1 * scaling_factor,  # Scale grid lines
        'lines.markersize': 3 * scaling_factor,
        'xtick.major.size': 6 * scaling_factor,  # Major tick length
        'xtick.major.width': 1.2 * scaling_factor,  # Major tick width
        'ytick.major.size': 6 * scaling_factor,
        'ytick.major.width': 1.2 * scaling_factor,
        'xtick.minor.size': 3 * scaling_factor,  # Minor tick length
        'xtick.minor.width': 0.8 * scaling_factor,  # Minor tick width
        'ytick.minor.size': 3 * scaling_factor,
        'ytick.minor.width': 0.8 * scaling_factor,
    })





#%% define functions <for r: +1 to -1 version>

def softmax_action_selection(state, Q, beta, num_arms=8):
    """Optimized Softmax Action Selection using log-sum-exp trick."""
    Q_s = beta * Q[state]  # Scale Q-values by beta
    log_probs = Q_s - logsumexp(Q_s)  # Compute log-softmax
    probabilities = np.exp(log_probs)  # Convert back to probabilities
    probabilities /= np.sum(probabilities)
    return np.random.choice(num_arms, p=probabilities)

def run_episode(Q, epsilon, beta, gamma, start_arm):
    """Simulate one trial of the radial arm maze."""
    state = start_arm
    reward_list = np.array([-1, -1, 1, 1, -1, -1, 1, -1])  # reward setup
    total_reward = 0
    state_list = [start_arm]
    
    action_count = 0
    while np.sum(reward_list) != -8:  # Continue until all treats are found
        action = softmax_action_selection(state, Q, beta)
        reward = reward_list[action]
        reward_list[action] = -1  # Mark arm as visited
        
        # Q-learning update rule
        best_future_value = np.max(Q[action])
        Q[state, action] += epsilon * (
            reward + gamma * best_future_value - Q[state, action])
        
        state = action  # Move to the new state
        total_reward += reward
        action_count += 1
        state_list.append(state)
    
    return total_reward, Q.copy(), action_count, state_list

def run_episode_no_learning(Q, epsilon, beta, gamma, start_arm):
    """Simulate one trial of the radial arm maze."""
    state = start_arm
    reward_list = np.array([-1, -1, 1, 1, -1, -1, 1, -1])  # reward setup
    total_reward = 0
    state_list = [start_arm]
    
    action_count = 0
    while np.sum(reward_list) != -8 and action_count < 100:  # Continue until all treats are found
        action = softmax_action_selection(state, Q, beta)
        reward = reward_list[action]
        reward_list[action] = -1  # Mark arm as visited
        
        state = action  # Move to the new state
        total_reward += reward
        action_count += 1
        state_list.append(state)
    
    return total_reward, Q.copy(), action_count, state_list

# for optimisation of beta fit to experimental data
def sim_mean_std_q(epsilon, beta, gamma, start_arm, n_agents=100,
                   n_trials=25, num_arms=8):
    
    Q_history = np.empty((n_agents, n_trials, num_arms, num_arms))
    for agent in range(n_agents):
        Q_init = np.zeros((num_arms, num_arms))
        for trial in range(n_trials):
            _, Q_history[agent, trial], _, _ = run_episode(Q_init, epsilon,
                                                           beta, gamma, 
                                                           start_arm)
            
    std_q = np.std(Q_history, axis=-1)
    mean_std_q = np.mean(std_q, axis=-1)
    agent_avg_mean_std_q = np.mean(mean_std_q, axis=0)
    agent_std_mean_std_q = np.std(mean_std_q, axis=0)

    return agent_avg_mean_std_q, agent_std_mean_std_q  

# for optimisation of beta fit to experimental data
def sim_num_entries(epsilon, beta, gamma, start_arm, n_agents=100,
                    n_trials=25, num_arms=8):
    
    # Q_history = np.empty((n_agents, n_trials, num_arms, num_arms))
    num_entries = np.empty((n_agents, n_trials))
    for agent in range(n_agents):
        Q_init = np.zeros((num_arms, num_arms))
        for trial in range(n_trials):
            _, _, num_entries[agent, trial], _ = run_episode(Q_init, epsilon,
                                                             beta, gamma, 
                                                             start_arm)
            
    mean_entries = np.mean(num_entries, axis=0)
    std_entries = np.std(num_entries, axis=0)

    return mean_entries, std_entries
    

def run_episode_real(Q, epsilon, gamma, start_arm, entries):
    """Simulate one trial of the radial arm maze."""
    state = start_arm
    reward_list = np.array([-1, -1, 1, 1, -1, -1, 1, -1])  # reward setup
    total_reward = 0
    
    action_count = 0
    # first element in entries is start arm so move on
    for action in entries[1:]:
        reward = reward_list[action]
        reward_list[action] = -1  # Mark arm as visited
        
        # Q-learning update rule
        best_future_value = np.max(Q[action])
        Q[state, action] += epsilon * (
            reward + gamma * best_future_value - Q[state, action])
        
        state = action  # Move to the new state
        total_reward += reward
        action_count += 1
    
    return total_reward, Q.copy(), action_count

# Function to animate Q-table evolution
def animate_q_table(Q_history, num_arms, arm_labels):
    vmax = np.max([np.abs(np.min(Q_history)), np.max(Q_history)])
    vmin = -1 * vmax
    fig, ax = plt.subplots()
    cax = ax.imshow(Q_history[0], cmap='coolwarm', vmin=vmin, vmax=vmax)
    plt.colorbar(cax)
    ax.set_xticks(range(num_arms))
    ax.set_yticks(range(num_arms))
    ax.set_xticklabels(arm_labels)
    ax.set_yticklabels(arm_labels)
    ax.set_xlabel("State Choice")
    ax.set_ylabel("Current State")
    
    def update(frame):
        cax.set_array(Q_history[frame])
        ax.set_title(f"Episode {frame+1}")
        return cax,
    
    ani = animation.FuncAnimation(fig, update, frames=len(Q_history), 
                                  interval=100, blit=False)
    
    # Ensure animation object persists
    return ani

def animate_q_table_single(Q_history, num_arms, arm_labels, title_prefix):
    """
    Animate a single agent's Q-table over trials.
    
    Q_history shape: (num_trials, num_arms, num_arms)
    """
    num_trials, _, _ = Q_history.shape  # Unpack dimensions
    
    vmax = np.max([np.abs(np.min(Q_history)), np.max(Q_history)])
    vmin = -1 * vmax
    
    # Create a figure and axis
    fig, ax = plt.subplots(dpi=100)
    
    # Initialize heatmap
    cax = ax.imshow(Q_history[0], cmap='coolwarm', vmin=vmin, vmax=vmax)
    
    # Set axis labels
    ax.set_xticks(range(num_arms))
    ax.set_yticks(range(num_arms))
    ax.set_xticklabels(arm_labels)
    ax.set_yticklabels(arm_labels)
    
    fig.colorbar(cax, ax=ax)
    fig.suptitle(f"{title_prefix} | Episode 1")
    
    extended_frames = 10  # Number of extra frames for the last Q-table
    
    def update(frame):
        """ Update heatmap per frame """
        if frame < num_trials:
            cax.set_data(Q_history[frame])
            fig.suptitle(f"{title_prefix} | Episode {frame+1}")
        return cax,
    
    ani = animation.FuncAnimation(fig, update, frames=num_trials + extended_frames, 
                                  interval=200, blit=False)
    
    return ani



def animate_q_table_multi(Q_history_multi, num_arms, arm_labels, title_prefix):
    """
    Animate multiple Q-tables in a 3x10 subplot layout.
    
    Q_history_multi shape: (num_agents, num_trials, num_arms, num_arms)
    """
    num_agents, num_trials, _, _ = Q_history_multi.shape  # Unpack dimensions
    
    vmax = np.max([np.abs(np.min(Q_history_multi)), np.max(Q_history_multi)])
    vmin = -1 * vmax

    # Create a 3x10 subplot figure
    fig, axes = plt.subplots(5, 10, dpi=50)
    axes = axes.flatten()  # Flatten for easy indexing

    # Initialize heatmaps with lower colormap resolution
    heatmaps = []
    for i in range(num_agents):
        cax = axes[i].imshow(Q_history_multi[i, 0], cmap='coolwarm', vmin=vmin, 
                             vmax=vmax)
        heatmaps.append(cax)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    # Remove empty subplots if num_agents < 50
    for j in range(num_agents, len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle(f"{title_prefix} | Episode 1")

    plt.tight_layout()
    
    
    # Add a single colorbar outside the grid (top-right of the figure)
    cbar_ax = fig.add_axes([0.75, 0.98, 0.2, 0.05])  # [lft, btm, width, height]
    fig.colorbar(heatmaps[0], cax=cbar_ax, orientation='horizontal', aspect=30)
    
    extended_frames = 10  # Number of extra frames for the last Q-table


    def update(frame):
        """ Efficiently update all heatmaps per frame """
        if frame < num_trials:
            for i in range(num_agents):
                heatmaps[i].set_data(Q_history_multi[i, frame])  # Use set_data() instead of set_array()
            fig.suptitle(f"{title_prefix} | Episode {frame+1}")
        return heatmaps

    ani = animation.FuncAnimation(fig, update, 
                                  frames=num_trials + extended_frames, 
                                  interval=200, blit=False)

    return ani



def q_learning_real_data(entry_df, epsilon, gamma, save_dir, num_trials, 
                         start_arm, num_arms=8):    
    
    # Initialize Q-table
    Q = np.zeros((num_arms, num_arms))  # Q-values for each (state, action) pair
    
    # Preallocate arrays for efficiency
    Q_history = np.zeros((num_trials, num_arms, num_arms))
    reward_history = np.empty(num_trials)
    num_entries = np.empty(num_trials)
    entry_arr = []
    # Run multiple episodes to train agent
    for episode in range(num_trials):
        entries = entry_df[entry_df.trial_idx == episode + 1]#.zone.values
        # merge entries very close together e.g. 3 0 3 in less than 2 seconds
        entries_nz = entries[entries.zone != 0]
        entry_t_diff = entries_nz.frame_time.iloc[1:].values - entries_nz.end_frame_t.iloc[:-1].values
        bool_keep = (entry_t_diff > 1) | (np.diff(entries_nz.zone) != 0)
        bool_keep = np.insert(bool_keep, 0, True) # keep first entry
        valid_entries = entries_nz.zone[bool_keep].values.astype(int)
        if valid_entries[0] != start_arm+1:
            valid_entries = np.insert(valid_entries, 0, start_arm+1)
    
        # reindex for python
        valid_entries -= 1
        # add entries to array for behavioural raster plots later
        entry_arr.append(valid_entries)
        
        trial_output = run_episode_real(Q, epsilon, gamma, start_arm, 
                                        valid_entries)
        reward_history[episode] = trial_output[0]
        Q_history[episode] = trial_output[1] 
        num_entries[episode] = trial_output[2]
    
    return reward_history, Q_history, num_entries, entry_arr

    
def plot_q_learning_real_data(reward_history, Q_history, num_entries, 
                              entry_arr, file_name, figtitle, save_dir,
                              file_suffix):
    
    # set save directories 
    save_dir_err = f"{save_dir}/errors"
    os.makedirs(save_dir_err, exist_ok=True)
    save_dir_rew = f"{save_dir}/rewards"
    os.makedirs(save_dir_rew, exist_ok=True)
    save_dir_combi = f"{save_dir}/combi_plots"
    os.makedirs(save_dir_combi, exist_ok=True)
    save_dir_raw = f"{save_dir}/raw_data"
    os.makedirs(save_dir_raw, exist_ok=True)
    save_dir_mean_std_q = f"{save_dir}/mean_std_q"
    os.makedirs(save_dir_mean_std_q, exist_ok=True)
    # get errors using -1 x (total reward - 3)
    error_history = -1 * (reward_history - 3)
    
    # save q table
    
    np.save(f"{save_dir_raw}/q_table_{file_name}.npy", Q_history)
    ########## plot mean_s of std_a of Q(s,a) for each agent per trial
    
    std_q = np.std(Q_history, axis=-1)
    mean_std_q = np.mean(std_q, axis=-1)
    plt.plot(mean_std_q, marker='.')
    plt.title(figtitle)
    plt.savefig(f"{save_dir_mean_std_q}/{file_name}.png")
    plt.close()
    
    
    # ########## plot mean reward 
    plt.plot(reward_history, marker='.')
    
    plt.ylabel("Total reward")
    plt.xlabel("Trial no.")
    
    plt.title(figtitle)
    plt.savefig(f"{save_dir_rew}/reward_history_{file_suffix}.png")
    plt.close()
    
    ############# plot mean errors
    plt.plot(error_history, marker='.')
    
    plt.ylabel("Total errors")
    plt.xlabel("Trial no.")
    
    plt.title(figtitle)
    plt.savefig(f"{save_dir_err}/error_history_{file_suffix}.png")
    plt.close()
    
    ############# plot mean errors and number of entries
    fig, ax1 = plt.subplots()
    
    # Plot mean errors on primary y-axis
    ax1.set_xlabel("Trial no.")
    ax1.set_ylabel("Total reward", color='b')
    ax1.plot(reward_history, marker='.', color='b', label="Trial tot. rew.")
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create a secondary y-axis for num_entries
    ax2 = ax1.twinx()
    ax2.set_ylabel("no. entries", color='r')
    ax2.plot(num_entries, marker='.', color='r', label="no. entries")
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Set title and save plot
    plt.title(figtitle)
    fig.tight_layout()  # Ensures labels don't overlap
    plt.savefig(f"{save_dir_combi}/rewards_and_entries_{file_suffix}.png")
    plt.close()
    
    
    
    
def plot_q_table_single_trial(Q_history, i, save_dir, num_arms, arm_labels,
                              figtitle, filename):
    ############## plot q table after 25 trials
    save_dir_qtab = f"{save_dir}/q_tables_final"
    os.makedirs(save_dir_qtab, exist_ok=True)
    fig, ax = plt.subplots()
    cax = ax.imshow(Q_history[i], cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(cax)
    ax.set_xticks(range(num_arms))
    ax.set_yticks(range(num_arms))
    ax.set_xticklabels(arm_labels)
    ax.set_yticklabels(arm_labels)
    ax.set_xlabel("State Choice")
    ax.set_ylabel("Current State")
    
    fig.suptitle(f"{figtitle} | Trial {i+1}")
    
    plt.savefig(f"{save_dir_qtab}/{filename}.png")
    plt.close()
    
    

"""entry+1 because arms indexed offset by 1"""
def plot_raster_q_table(entry_arr, file_name, rast_save_path, figtitle, 
                        tick_w=0.75, tick_h=0.8):
    
    ######### plot behavioural raster plot for random sample of agents
    fig, ax = plt.subplots(1, 1)
    agent_entries = entry_arr
    max_entries = np.max([len(agent_entries[i]) for i in range(25)])
    for trial_idx in range(25):
        trial_entries = agent_entries[trial_idx]
        # Shade zones within this trial
        for i, entry in enumerate(trial_entries):
            rect = Rectangle(
                (i, trial_idx - 0.4),
                tick_w, # width of rectangle
                tick_h,  # height of the rectangle (spanning the entire trial)
                facecolor=uf.zone_colours[entry+1],
                alpha=1
            )
            ax.add_patch(rect)
    
    # Set limits to remove space before the zeros
    ax.set_xlim(left=0, right=max_entries)
    ax.set_ylim(bottom=-.5, top=24.5)
    ax.set_yticks(np.arange(0, 25, 1))
    ax.set_yticklabels(np.arange(1, 26, 1))  # Labels starting from 1
    ax.invert_yaxis()
    
    ax.set_ylabel("Trial no.")
    ax.set_xlabel("Action no.")
    
    plt.title(figtitle)
    plt.savefig(f"{rast_save_path}.png")
    plt.close()