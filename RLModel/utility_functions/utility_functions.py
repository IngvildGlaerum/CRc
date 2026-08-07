# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:08:41 2024

@author: rejwan
"""

import pandas as pd
import numpy as np
import re
import os

#### GLOBALS ####
processed_data_dir = "./../../data/processed"
expt_type_dict = {0: 'Open field', 1: 'RAM habituation', 2: 'Trial block 1',
                  3: 'Trial block 2'}
zone_colours = {0: 'white', 1: '#e73133', 2: '#4b8bbf', 3: '#5fb75c', 
                4: '#ff8c19', 5: '#ddcf6e', 6: '#af673d', 7: '#f88ec5', 
                8: '#a3a3a3'}
#################

def load_session_data(mouse_id, date):
    
    return pd.read_pickle(f"{processed_data_dir}/{mouse_id}_{date}.pkl")

def load_sessionwide_metadata():
    
    return pd.read_pickle(f"{processed_data_dir}/../session_metadata.pkl")

def translate_meta_to_str(meta_df, mouse_id, date):
    """
    This finds the entry in the metadata and return the name of the 
    experimental type as well as the day <latter is easy to get, but efficient
    code by doing it here>

    Parameters
    ----------
    meta_df : DataFrame
        Metadata of all sessions.
    mouse_id : TYPE
        DESCRIPTION.
    date : TYPE
        DESCRIPTION.

    Returns
    -------
    expt_type : TYPE
        DESCRIPTION.
    day : TYPE
        DESCRIPTION.
    day_corrected : TYPE
        If trial block 2, then for ease of access, returns the day - 3 so we 
        can easily denote that day 4 is the first day of trial block 2, say.

    """
    
    session_meta = meta_df[(meta_df.animal_id == mouse_id) & 
                           (meta_df.date == date)]
    expt_type_idx = session_meta.experiment_type.values[0]
    day = session_meta.day.values[0]
    expt_type = expt_type_dict[expt_type_idx]
    
    day_corrected = np.nan
    if expt_type_idx == 3:
        day_corrected = day - 3
        
    
    return expt_type, day, day_corrected


def find_directory(base_path, mouse_id, date):
    """
    Code to programmatically find the right directory for the raw data. 
    So far, the purpose is for enabling raw data to be extracted to be later
    filtered and downsampled into LFP data 

    Parameters
    ----------
    base_path : TYPE
        DESCRIPTION.
    mouse_id : TYPE
        DESCRIPTION.
    date : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # Simplified regex pattern
    pattern = re.compile(rf"{mouse_id}_{date}_[0-9]{{2}}-[0-9]{{2}}-[0-9]{{2}}_A$")
    
    # Search for the first matching directory
    for dir_name in os.listdir(base_path):
        if pattern.match(dir_name):
            return dir_name  # Return immediately if found
    
    return None  # Return None if no match is found


