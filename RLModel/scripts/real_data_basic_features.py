# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:14:18 2025

@author: rejwanfs
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import utility_functions.utility_functions as uf
import rl_functions as rlf
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns

########### set plotting parameters ############
rlf.set_plotting_params()
plt.ioff()
plt.close('all')
################################################

#%% Load general metadata and define base directories for saving

# base load dir
load_dir = "./../../Data"

# base save dir create
save_dir = "./../../Output/RL/real_behaviour_spatial"
os.makedirs(save_dir, exist_ok=True)

plot = True
plot_mean_std_q = True

# set global variables 
start_arm_mapping = {1: 2, 
                     2: 5}

# --- NEW: treat locations (arms) ---
TREAT_ARMS = {3, 4, 7}

num_arms = 8
arm_labels = list(range(1,9))

num_days = 14

genotype_map = {2: 'cre+', 1: 'cre-'}

file_geno_names = {1: 'control', 2: 'experimental'}


#%% load session data and get entries
# create empty array to fill with data that will get concatenated into datafrme
basic_data_arr = []
# invalid_entries = []
# list to collect trajectory data
trajectory_data = []

for geno_id, genotype in enumerate(['control', 'experimental']):
    for mouse_id in range(1,5):
        
        # load mouse data
        mouse_df = pd.read_pickle(f"{load_dir}/{genotype}_{mouse_id}")
        
        # loop through sessions and fill basic data arr
        for day in range(1,num_days+1):
            
            for session in range(2):
                
                sess_data = mouse_df[(mouse_df.day == day) & 
                                     (mouse_df.session == session)]
                
                sess_trial_block = sess_data.trial_block.values[0]
                
                start_arm = start_arm_mapping[sess_trial_block] # maybe need to subtract 1
                
                entry_df = rlf.get_full_entry_df(sess_data)
        
                # extract number of trials
                n_real_trials = entry_df.trial.nunique()
                
                # loop over trials, get length <s>, distance traveled <m>, and # errors
                mouse_data = []
                for trial_idx in entry_df.trial.unique().astype(int):
                    
                    trial_t = sess_data[sess_data.trial == trial_idx].time
                    
                    # calculate number of errors
                    """get code from RL to remove spurious re-entry"""
                    entries = entry_df[entry_df.trial == trial_idx]#.zone.values
                    # merge entries very close together e.g. 3 0 3 in less than 2 seconds
                    entries_nz = entries[entries.zone != 0]
                    entry_t_diff = (entries_nz.time.iloc[1:].values 
                                    - entries_nz.end_frame_t.iloc[:-1].values)
                    bool_keep = (entry_t_diff > 1) | (np.diff(entries_nz.zone) != 0)
                    bool_keep = np.insert(bool_keep, 0, True) # keep first entry
                    valid_entries = entries_nz.zone[bool_keep].values.astype(int)
                    
                    # add arm 2 if it was missed
                    if valid_entries[0] != start_arm:
                        valid_entries = np.insert(valid_entries, 0, start_arm)
                    # num errors is len - 3 <baited first entry> - 1 <start arm is initial>
                    # --- DIRECT error counting w.r.t. treats in {3,4,7} ---
                    err_r1 = err_r2 = err_r3 = 0
                    collected = set()
                    segment = 1  # 1: before 1st treat, 2: 1st→2nd, 3: 2nd→3rd
                    
                    # iterate after the (possibly inserted) start arm
                    for arm in valid_entries[1:]:
                        is_new_treat = (arm in TREAT_ARMS) and (arm not in collected)
                        if is_new_treat:
                            collected.add(arm)
                            if len(collected) >= 3:
                                # all three treats acquired; stop counting
                                break
                            segment = min(segment + 1, 3)
                        else:
                            # any entry that is NOT the next new treat is an error in the current segment
                            if segment == 1:
                                err_r1 += 1
                            elif segment == 2:
                                err_r2 += 1
                            else:
                                err_r3 += 1
                    
                    # total errors = sum of segment errors (works even if <3 treats were collected)
                    trial_num_errors = err_r1 + err_r2 + err_r3

                    
                    # calculate total trial length
                    tot_trial_length = trial_t.iloc[-1] - trial_t.iloc[0]
                    """subtract total frame time of rows containing nans?"""
                    


                    # add data to mouse_data along with mouse id, date, genotype, etc
                    """or do that at the end after the loop and concatnating"""
                    mouse_data.append([trial_idx, trial_num_errors, tot_trial_length, err_r1, err_r2, err_r3])
                    
                    # save trajectory with meta info
                    trajectory_data.append({
                        "animal_id": mouse_id,
                        "genotype": geno_id + 1,
                        "day": day,
                        "session": session,
                        "trial_idx": trial_idx,
                        "entries": valid_entries.tolist()
                    })
                    
                    
                    
                    # frac = 50 / len(trial_x) # whatever fraction is 50 frames
                    # smth_x = lowess(trial_x, trial_t, frac=frac, it=1, return_sorted=False)
                    # smth_y = lowess(trial_y, trial_t, frac=frac, it=3, return_sorted=False)
                    
                # append to data arr
                """add genotype, day, type, etc"""
                genotype_arr = np.full(n_real_trials, geno_id+1, dtype=int)
                mouse_id_arr = np.full(n_real_trials, mouse_id, dtype=int)
                trial_block_arr = np.full(n_real_trials, sess_trial_block, dtype=int)
                day_arr = np.full(n_real_trials, day, dtype=int)
                meta_concat = np.vstack([mouse_id_arr, genotype_arr, trial_block_arr, day_arr]).T
                mouse_data_concat = np.vstack(mouse_data)
                # concat both meta and trial results and append to population data
                basic_data_arr.append(np.hstack([meta_concat, mouse_data_concat]))
                

# concatenate data array and create dataframe from it
concat_basic_data = np.vstack(basic_data_arr)
df_basic_data = pd.DataFrame(
    concat_basic_data,
    columns=[
        'animal_id', 'genotype', 'experiment_type', 'day', 'trial_idx',
        'num_errors', 't_trial', 'err_r1', 'err_r2', 'err_r3'
    ],
    dtype=float
)

# get performance using 100 * ( 1 - num_errors/(num_errors+3) )
performance = 100 * (
    1 - df_basic_data.num_errors / (df_basic_data.num_errors + 3))
df_basic_data.insert(len(df_basic_data.columns), 'performance', performance)
# convert trajectory list to dataframe
df_entries = pd.DataFrame(trajectory_data, 
    columns=['animal_id', 'genotype', 'day', 'session', 'trial_idx', 'entries'])

# save dataframes
df_basic_data.to_pickle(f"{save_dir}/pop_basic_data.pkl")
df_entries.to_pickle(f"{save_dir}/pop_entry_data.pkl")

#%% plot scatter
df = df_basic_data
# remove 65935
# plt.ioff()
# plot all data
sns.pairplot(df, hue='genotype', vars=['num_errors', 't_trial'], 
             kind='scatter', palette={1: 'blue', 2: 'red'}, 
             plot_kws={'alpha': 0.4, 's': 20}, diag_kind='hist', 
             diag_kws={'common_norm': False})
plt.savefig(f"{save_dir}/pop_basic_data_pairplot.png")
plt.close()

# constrain to day 1
for i in range(1,15):
    sns.pairplot(df[df.day==i], hue='genotype', 
                 vars=['num_errors', 't_trial'], 
                 kind='scatter', palette={1: 'blue', 2: 'red'}, 
                 diag_kind='hist', diag_kws={'common_norm': False},
                 plot_kws={'alpha': 0.4, 's': 20})
    plt.savefig(f"{save_dir}/pop_basic_data_pairplot_day_{i}.png")
    plt.close()


# %% plot psychometric curves for each mouse
# Ensure seaborn style for better visualization
sns.set_style("white")
sns.set_palette('dark', 6)


# Get unique animals
unique_animals = df['animal_id'].unique()

# get trial indexing resetting each day
# 1..8 repeating regardless of animal/day
df['trial_in_day'] = ((df['trial_idx'] - 1) % 8) + 1

# Loop through each animal
for geno_id, genotype in enumerate(['control', 'experimental']):
    for mouse_id in range(1,5):
    
        animal = int(mouse_id)
        
        animal_df = df[(df['animal_id'] == animal) & (df.genotype == geno_id+1)]
        animal_df['day'] = animal_df['day'].apply(int).apply(str)
        # genotype = genotype_map[animal_df.genotype.iloc[0]]
        
        
        # first create scatter plot of trial length and performance
        sns.scatterplot(data=animal_df, x='t_trial', y='performance', hue='day',
                        s=50, alpha=0.4)
        plt.ylim([-5,105])
        plt.ylabel("Performance (%)")
        plt.xlabel("Trial Length (sec)")
        plt.savefig(f"{save_dir}/{genotype}_{animal}_psychometric_scatter.png")
        plt.close()
        
        # Create figure with 2 rows, 7 columns for 14 days
        fig, axes = plt.subplots(2, 7, sharex=True, sharey=True)
        axes = axes.flatten()  # Flatten for easy indexing
        
        # Loop through days (1 to 6)
        for day in range(1, 15):
            ax = axes[day - 1]  # Get correct subplot
            
            # Subset data for this day
            day_df = animal_df[animal_df['day'] == str(day)]
            
            # Plot trial length (t_trial) on primary y-axis
            sns.lineplot(x='trial_in_day', y='t_trial', data=day_df, 
                         ax=ax, color='blue', marker='o')
    
            # Create secondary y-axis for performance
            ax2 = ax.twinx()
            sns.lineplot(x='trial_in_day', y='performance', data=day_df, 
                         ax=ax2, color='red', marker='s')
    
            # Formatting
            ax.set_title(f"Day {day}")
            ax.set_xlabel("Trial Number")
            ax.set_ylabel("Trial Length (sec)", color='blue')
            ax2.set_ylabel("Performance (%)", color='red')
            # ax.set_xlim([1,25])
            ax2.set_ylim([-5,105])
            
            # Adjust y-axis colors for clarity
            ax.tick_params(axis='y', colors='blue')
            ax2.tick_params(axis='y', colors='red')
    
            # # Remove overlapping legends
            # ax.legend(loc="upper left", fontsize=8)
            # ax2.legend(loc="upper right", fontsize=8)
    
        # note genotype and mouse id
        plt.suptitle(f"{animal} | ({genotype})")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"{save_dir}/{genotype}_{animal}_psychometric_curves.png")
        plt.close()    
        
# %% NEW: per-animal segmented-error plots (r1, r2, r3) – summed per day
for geno_id, genotype in enumerate(['control', 'experimental']):
    for mouse_id in range(1, 5):
        animal_df = df_basic_data[
            (df_basic_data['animal_id'] == mouse_id) &
            (df_basic_data['genotype'] == (geno_id + 1))
        ].copy()
        if animal_df.empty:
            continue

        # ensure integer day and aggregate by day
        animal_df['day'] = animal_df['day'].astype(int)
        daily = (animal_df
                 .groupby('day')
                 .agg(err_r1_sum=('err_r1', 'sum'),
                      err_r2_sum=('err_r2', 'sum'),
                      err_r3_sum=('err_r3', 'sum'))
                 .reset_index()
                 .sort_values('day'))

        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
        for ax, col, title in zip(
            axes,
            ['err_r1_sum', 'err_r2_sum', 'err_r3_sum'],
            ['Errors before 1st treat (sum)',
             'Errors between 1st–2nd (sum)',
             'Errors between 2nd–3rd (sum)']
        ):
            ax.plot(daily['day'], daily[col], 'o-', linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel('Day')
            ax.set_ylabel('Errors (sum)')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, num_days + 1))

        plt.suptitle(f'{mouse_id} ({genotype}) – Summed reward-segment errors by day')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{save_dir}/{genotype}_{mouse_id}_reward_segment_errors_sum_by_day.png")
        plt.close()




