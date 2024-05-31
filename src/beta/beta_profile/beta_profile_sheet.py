""" Beta profile sheet to Excel """

# import os
# import matplotlib.pyplot as plt
# import mne
# import scipy
# from cycler import cycler
# from scipy.signal import hann
import numpy as np
import pandas as pd


# PyPerceive Imports
from PerceiveImport.classes import main_class
from ..utils import find_folders as find_folders
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

def get_maximal_beta_peak_CF(ring_channels=None, beta_peak_details=None):
    """
    This function takes all beta peak values from the Ring Channels (originally from the tfr.main_tfr() function)
        - with already calculated 4 Hz Peak power values around the highest peaks of each channels
        - it then ranks the 4Hz peak power values
        - and outputs only the Peak CF of the maximal 4Hz peak power from all these channels
    """

    # only keep the ring channels
    group_peak_details = beta_peak_details.loc[beta_peak_details.channel.isin(ring_channels)]
    group_peak_details_copy = group_peak_details.copy()

    # rank peak_4_hz_power
    group_peak_details_copy["beta_rank"] = group_peak_details_copy["peak_4Hz_power"].rank(
        ascending=False
    ) # rank 1 means maximal beta power

    # Get the CF of the maximal beta from the channel with maximal peak 4Hz Power around their own highest beta peak
    max_Ring_peak_CF = group_peak_details_copy.loc[group_peak_details_copy["beta_rank"] == 1.0]
    return max_Ring_peak_CF.peak_CF.values[0]



def calculate_peak_4Hz_power_around_CF(max_Ring_peak_CF=None, f=None, psd=None):
    """
    From a fixed CF the peak power ± 2 Hz will be calculated 

    Input: 
    
    """
    # calculate the psd average of ± 2 Hz around the max_Ring_peak_CF
    peak_index = np.where(f == max_Ring_peak_CF)
    peak_index = peak_index[0].item()

    # go -2 and +3 indices
    index_low_cut = peak_index-2
    index_high_cut = peak_index+3   # +4 because the ending index is left out when slicing a numpy array

    return np.mean(psd[index_low_cut:index_high_cut])


def write_beta_profile(
        sub:str,
        session:str,
        condition:str,
        beta_range:str
):
    """
    Input:
        beta_range: "beta", "low_beta", "high_beta" -> always start with "beta", 
            -> use high beta e.g. in case the "higest peak" is at the border 13 Hz, but an actual peak is visible in higher freq ranges

    1) for Ring and Segm channels separately
    2) Rank channels by peak_4Hz_power within beta 13-35 Hz
    3) extract the maximal beta Ring channel and get the CF of the highest beta peak = fixed CF
      -> this CF will be used for all 4 Hz Peak Power calculations
    3) for each channel: extract the peak_4Hz_power around that fixed CF and rank (Ring, Segm separately)
    4) Calculate the relative 4 Hz peak power of channels with lower beta power
    
    """
    
    beta_profile_all = {
        "Right_Ring": pd.DataFrame(),
        "Right_Segm": pd.DataFrame(),
        "Left_Ring": pd.DataFrame(),
        "Left_Segm": pd.DataFrame()
    }
    beta_result_to_excel = {}

    for hem in HEMISPHERES: 

        load_peak_details = tfr.main_tfr(
            sub=sub,
            session=session,
            condition=condition,
            hemisphere=hem
        )

        peak_details = load_peak_details[0]
        LFP_data = load_peak_details[1]
        frequencies = LFP_data.frequencies.values[0]

        # only look at the beta range of interest e.g. 13-35 Hz
        beta_peak_details = peak_details.loc[peak_details.f_range == beta_range]

        ################## 1. Get the Peak CF of the maximal peak of all RING channels depending on their 4 Hz power of their highest Peak in the given freq band ###################
        ring_channels = PICK_CHANNELS["Ring"]

        max_Ring_peak_CF = get_maximal_beta_peak_CF(
            ring_channels=ring_channels, 
            beta_peak_details=beta_peak_details,
        )

        ################### 2. calculate the 4 Hz power around the max_Ring_peak_CF for all other channels ###################
        groups = ["Ring", "Segm"]
        for group in groups:

            channels = PICK_CHANNELS[group]

            for chan in channels:

                chan_lfp = LFP_data.loc[LFP_data.channel == chan]
                psd = chan_lfp.filtered_psd.values[0]

                # 4 Hz power around the max Ring peak CF -> all around the same CF
                power_4Hz_range_around_peak = calculate_peak_4Hz_power_around_CF(max_Ring_peak_CF=max_Ring_peak_CF,
                                                                                 f=frequencies,
                                                                                 psd=psd)

                # new dataframe row with channel, max_ring_peak_cf, power 4Hz range
                beta_profile_single_dict = {
                    "channel": [chan],
                    "f_range": [beta_range],
                    "ring_max_beta_CF": [max_Ring_peak_CF],
                    "peak_4Hz_power": [power_4Hz_range_around_peak]
                }

                single_to_df = pd.DataFrame(beta_profile_single_dict)
                beta_profile_all[f"{hem}_{group}"] = pd.concat([beta_profile_all[f"{hem}_{group}"], single_to_df])

            # rank all power 4 Hz values
            beta_profile_all_copy = beta_profile_all[f"{hem}_{group}"].copy()
            beta_profile_all_copy["beta_rank"] = beta_profile_all_copy["peak_4Hz_power"].rank(
                ascending=False
            ) # rank 1 means maximal beta 

            # calculate rel beta peak power relative to the maximal value
            max_beta_power = beta_profile_all_copy["peak_4Hz_power"].max()
            beta_profile_all_copy["rel_beta_peak_power"] = beta_profile_all_copy["peak_4Hz_power"] / max_beta_power

            # re-order the dataframe by the beta rank
            beta_profile_all_copy = beta_profile_all_copy.sort_values(by="beta_rank")

            beta_result_to_excel[f"{hem}_{group}"] = beta_profile_all_copy

    # save as excel file
    io.save_df_to_excel_sheets(
        sub=sub,
        filename=f"{beta_range}_profile_{session}_{condition}",
        file=beta_result_to_excel,
    )

    return beta_result_to_excel



# def write_beta_profile(
#         sub:str,
#         session:str,
#         condition:str,
# ):
#     """
#       OLD VERSION: this function writes Excel File with 4Hz peak power values around the CF of the highest peak within each channel
#       NO FIXED CF TO THE MAX RING PEAK
#
#     1) for Ring and Segm channels separately
#     2) Rank channels by peak_4Hz_power within beta 13-35 Hz
#     3) for the maximal beta channel: extract peak CF and peak_4Hz_power
#     4) Calculate the relative 4 Hz peak power of channels with lower beta power
    
#     """
#     beta_profile_all = {}

#     for hem in HEMISPHERES: 

#         load_peak_details = tfr.main_tfr(
#             sub=sub,
#             session=session,
#             condition=condition,
#             hemisphere=hem
#         )

#         peak_details = load_peak_details[0]

#         # only look at beta 13-35 Hz
#         beta_peak_details = peak_details.loc[peak_details.f_range == "beta"]

#         # Rank beta channels separately for Rings and Segments
#         groups = ["Ring", "Segm"]

#         for group in groups:

#             channels = PICK_CHANNELS[group]

#             group_peak_details = beta_peak_details.loc[beta_peak_details.channel.isin(channels)]
#             group_peak_details_copy = group_peak_details.copy()

#             # rank peak_4_hz_power
#             group_peak_details_copy["beta_rank"] = group_peak_details_copy["peak_4Hz_power"].rank(
#                 ascending=False
#             ) # rank 1 means maximal beta power

#             max_beta_power = group_peak_details_copy["peak_4Hz_power"].max()
#             group_peak_details_copy["rel_beta_peak_power"] = group_peak_details_copy["peak_4Hz_power"] / max_beta_power

#             # re-order the dataframe by the beta rank
#             group_peak_details_copy = group_peak_details_copy.sort_values(by="beta_rank")

#             beta_profile_all[f"{hem}_{group}"] = group_peak_details_copy

#     # save as excel file
#     io.save_df_to_excel_sheets(
#         sub=sub,
#         filename=f"beta_profile_{session}_{condition}",
#         file=beta_profile_all,
#     )

#     return beta_profile_all