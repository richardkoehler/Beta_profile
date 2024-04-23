""" Beta profile sheet to Excel """

import os
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy
from cycler import cycler
from scipy.signal import hann


# PyPerceive Imports
from PerceiveImport.classes import main_class
from .. utils import find_folders as find_folders
from ..utils import io as io
from ..preprocessing import tfr as tfr


LFP_GROUPS = {
    "Right": ["RingR", "SegmIntraR", "SegmInterR"],
    "Left": ["RingL", "SegmIntraL", "SegmInterL"]
}

PICK_CHANNELS = {
    "Ring": ["01", "12", "23"],
    "Segm": ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C", "1A2A", "1B2B", "1C2C"]
}

ALL_CHANNELS = ["01", "12", "23", "1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C", "1A2A", "1B2B", "1C2C"]

BETA_RANGES = ["low_beta", "high_beta", "beta"]

HEMISPHERES = ["Right", "Left"]

def write_beta_profile(
        sub:str,
        session:str,
        condition:str,
):
    """
    1) for Ring and Segm channels separately
    2) Rank channels by peak_4Hz_power within beta 13-35 Hz
    3) for the maximal beta channel: extract peak CF and peak_4Hz_power
    4) Calculate the relative 4 Hz peak power of channels with lower beta power
    
    """
    beta_profile_all = {}

    for hem in HEMISPHERES: 

        load_peak_details = tfr.main_tfr(
            sub=sub,
            session=session,
            condition=condition,
            hemisphere=hem
        )

        peak_details = load_peak_details[0]

        # only look at beta 13-35 Hz
        beta_peak_details = peak_details.loc[peak_details.f_range == "beta"]

        # Rank beta channels separately for Rings and Segments
        groups = ["Ring", "Segm"]

        for group in groups:

            channels = PICK_CHANNELS[group]

            group_peak_details = beta_peak_details.loc[beta_peak_details.channel.isin(channels)]
            group_peak_details_copy = group_peak_details.copy()

            # rank peak_4_hz_power
            group_peak_details_copy["beta_rank"] = group_peak_details_copy["peak_4Hz_power"].rank(
                ascending=False
            ) # rank 1 means maximal beta power

            max_beta_power = group_peak_details_copy["peak_4Hz_power"].max()
            group_peak_details_copy["rel_beta_peak_power"] = group_peak_details_copy["peak_4Hz_power"] / max_beta_power

            # re-order the dataframe by the beta rank
            group_peak_details_copy = group_peak_details_copy.sort_values(by="beta_rank")

            beta_profile_all[f"{hem}_{group}"] = group_peak_details_copy

    # save as excel file
    io.save_df_to_excel_sheets(
        sub=sub,
        filename=f"beta_profile_{session}_{condition}",
        file=beta_profile_all,
    )

    return beta_profile_all







