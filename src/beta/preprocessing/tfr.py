""" Fast Fourier """

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

def band_pass_filter(signal:None):
    """ 
    create a butterworth filter 5-95 Hz
    
    """

    # sampling frequency: 250 Hz
    fs = 250

    # set filter parameters for band-pass filter
    filter_order = 5 # in MATLAB spm_eeg_filter default=5 Butterworth
    frequency_cutoff_low = 5 # 5Hz high-pass filter
    frequency_cutoff_high = 95 # 95 Hz low-pass filter

    # create the filter
    b, a = scipy.signal.butter(filter_order, (frequency_cutoff_low, frequency_cutoff_high), btype='bandpass', output='ba', fs=fs)

    return scipy.signal.filtfilt(b, a, signal) 


def fourier_transform(signal:None):
    """
    Fourier transform with 50% overlap, window length of 1 second
    
    """
    fs = 250
    window = fs # window length = 1 sec; frequencies will be from 0 to 125 Hz, 125Hz = Nyquist = fs/2
    noverlap = window // 2 # 50% overlap of windows

    window = hann(window, sym=False)

    # compute spectrogram with Fourier Transforms

    f,time_sectors,Sxx = scipy.signal.spectrogram(x=signal, fs=fs, window=window, noverlap=noverlap,  scaling='density', mode='psd', axis=0)
    # f = frequencies 0-125 Hz (Maximum = Nyquist frequency = sfreq/2)
    # time_sectors = sectors 0.5 - 20.5 s in 0.5 steps (in total 21 time sectors)
    # Sxx = 126 arrays with 21 values each of PSD [µV^2/Hz], for each frequency bin PSD values of each time sector
    # Sxx = 126 frequency rows, 21 time sector columns

    # average all 21 Power spectra of all time sectors 
    average_Sxx = np.mean(Sxx, axis=1) # axis = 1 -> mean of each column: in total 21x126 mean values for each frequency
                

    #################### CALCULATE THE STANDARD ERROR OF MEAN ####################
    # SEM = standard deviation / square root of sample size
    Sxx_std = np.std(Sxx, axis=1) # standard deviation of each frequency row
    Sxx_sem = Sxx_std / np.sqrt(Sxx.shape[1]) # sample size = 21 time vectors -> sem with 126 values

    return {
        "freq": f,
        "time_sectors": time_sectors,
        "Sxx": Sxx,
        "average_Sxx": average_Sxx,
        "Sxx_sem": Sxx_sem
    }



def pick_channels_of_interest(
        sub:str,
        session:str,
        condition:str,
        hemisphere:str,):
    """
    Pick either Ring or Segm lfp group: Input "Ring" or "Segm"

    This function picks the channels of interest either for Ring or Segm LFP groups
        - Ring: 0-1, 1-2, 2-3
        - Segm: 1A-1B, 1A-1C, 1B-1C, 2A-2B, 2A-2C, 2B-2C, 1A-2A, 1B-2B, 1C-2C

    """
    structured_signals_dataframe = pd.DataFrame()

    # load the MNE object
    mne_object = io.extract_data_from_py_perceive(
        sub=sub,
        session=session,
        condition=condition,
        hemisphere=hemisphere
    )

    for group in LFP_GROUPS[hemisphere]:

        if "Ring" in group:
            ring_or_segm = "Ring"
        
        elif "Segm" in group:
            ring_or_segm = "Segm"

        lfp_data = mne_object[group]

        # get new channel names
        ch_names = lfp_data.info.ch_names

        #################### PICK CHANNELS ####################
        include_channel_list = [] # this will be a list with all channel names selected
        
        for names in ch_names:
            
            # add all channel names that contain the picked channels: e.g. 02, 13, etc given in the input pickChannels
            for picked in PICK_CHANNELS[ring_or_segm]:
                if picked in names:
                    include_channel_list.append(names)

        # Error Checking: 
        if len(include_channel_list) == 0:
            print("Channel names don't exist")
            continue

        # pick channels of interest: mne.pick_channels() will output the indices of included channels in an array
        ch_names_indices = mne.pick_channels(ch_names, include=include_channel_list)

    
        for i, ch in enumerate(ch_names):
            
            # only get picked channels
            if i not in ch_names_indices:
                continue

            short_ch_name = [substring for substring in ch.split("_") if substring in PICK_CHANNELS[ring_or_segm]]

            signal_structured = {
                "original_channel": [ch],
                "channel": [short_ch_name[0]],
                "lfp": [lfp_data.get_data()[i, :]],
                "lfp_group": [group]
            }

            signals_df = pd.DataFrame(signal_structured)
            structured_signals_dataframe = pd.concat([structured_signals_dataframe, signals_df], ignore_index=True)
        
    
    return structured_signals_dataframe



def power_average_in_freq(psd:None, f:None):
    """

    Input: 
        - f: frequencies
        - lfp: band-pass filtered power spectrum
    
    """                
    # create booleans for each frequency-range for low beta, high beta, beta
    low_beta_frequency = (f >= 13) & (f <= 20)
    high_beta_frequency = (f >= 21) & (f <= 35)
    beta_frequency = (f >= 13) & (f <= 35)
    
    low_beta_power = psd[low_beta_frequency] # all psd values within a frequency band
    low_beta_power = np.mean(low_beta_power)

    high_beta_power = psd[high_beta_frequency] # all psd values within a frequency band
    high_beta_power = np.mean(high_beta_power)

    beta_power = psd[beta_frequency] # all psd values within a frequency band
    beta_power = np.mean(beta_power)

    return {
        "low_beta": low_beta_power,
        "high_beta": high_beta_power,
        "beta": beta_power
    }


def peak_detection(psd:None, f:None
        
):
    """

    
    """

    peak_details = pd.DataFrame()
    # find all peaks: peaks is a tuple -> peaks[0] = index of frequency?, peaks[1] = dictionary with keys("peaks_height") 
    peaks = scipy.signal.find_peaks(psd, height=0.1) # height: peaks only above 0.1 will be recognized

    # Error checking: if no peaks found, continue
    if len(peaks) == 0:
        print("no peaks found")
    
    else: 

        peaks_height = peaks[1]["peak_heights"] # np.array of y-value of peaks = power
        peaks_pos = f[peaks[0]] # np.array of indeces on x-axis of peaks = frequency

        # get all peak positions and heights within each frequency range
        low_beta_peak_pos = peaks_pos[(peaks_pos >= 13) & (peaks_pos <= 20)]
        low_beta_peak_height = peaks_height[(peaks_pos >= 13) & (peaks_pos <= 20)]

        high_beta_peak_pos = peaks_pos[(peaks_pos >= 21) & (peaks_pos <= 35)]
        high_beta_peak_height = peaks_height[(peaks_pos >= 21) & (peaks_pos <= 35)]

        beta_peak_pos = peaks_pos[(peaks_pos >= 13) & (peaks_pos <= 35)]
        beta_peak_height = peaks_height[(peaks_pos >= 13) & (peaks_pos <= 35)]        


        for range in BETA_RANGES:
            if range == "low_beta":
                peak_pos = low_beta_peak_pos
                peak_height = low_beta_peak_height
            
            elif range == "high_beta":
                peak_pos = high_beta_peak_pos
                peak_height = high_beta_peak_height
            
            elif range == "beta":
                peak_pos = beta_peak_pos
                peak_height = beta_peak_height

            # Error checking: check first, if there is a peak in the frequency range
            if len(peak_height) == 0:
                print(f"No peak in {range}")
                highest_peak_height = None
                highest_peak_pos = None
                power_4Hz_range_around_peak = None

            else: 
                # select only the highest peak within the freq range
                highest_peak_height = peak_height.max()
                # get the index of the highest peak y value to get the corresponding peak position x
                ix = np.where(peak_height == highest_peak_height)
                highest_peak_pos = peak_pos[ix].item()

                ######## calculate psd average of +- 2 Hz from highest Peak ########
                # 1) find psd values from -2Hz until +2Hz from highest Peak by slicing and indexing the numpy array of all chosen psd values
                peak_index = np.where(psd == highest_peak_height) # np.where output is a tuple: index, dtype
                peak_index_value = peak_index[0].item() # only take the index value of the highest Peak psd value in all chosen psd

                # 2) go -2 and +3 indeces 
                index_low_cut = peak_index_value-2
                index_high_cut = peak_index_value+3   # +4 because the ending index is left out when slicing a numpy array

                # 3) slice the numpy array of all chosen psd values, only get values from -2 until +2 Hz from highest Peak
                power_4Hz_range_around_peak = np.mean(psd[index_low_cut:index_high_cut]) # array only of psd values -2 until +2Hz around Peak = 5 values                      

            peak_dict = {
                "f_range": [range],
                "peak_CF": [highest_peak_pos],
                "peak_power": [highest_peak_height],
                "peak_4Hz_power": [power_4Hz_range_around_peak]
            }

            peaks_df = pd.DataFrame(peak_dict)
            peak_details = pd.concat([peak_details, peaks_df], ignore_index=True)
    
    return peak_details



def main_tfr(
        sub:str,
        session:str,
        condition:str,
        hemisphere:str
):
    """

    1) Load all relevant Ring and Segm BSSU time series
    2) Band-pass filter 5-95 Hz
    3) Fourier Transform to obtain Spectral power: 1sec window length, 50% window overlap
    4) Extract features within beta, low beta and high beta: 
        - average power in freq band
        - highest peak CF and peak power
    
    """

    beta_profile = pd.DataFrame()
    lfp_psd_data = pd.DataFrame()

    # get the path to write the Excel file and plots into the Teams Folder
    sub_path = io.load_sub_path(sub=sub)

    # load the LFP data through PyPerceive and extract the LFP data in 3 LFP groups
    structured_lfp_data = pick_channels_of_interest(
        sub=sub,
        session=session,
        condition=condition,
        hemisphere=hemisphere
    )

    # for each channel, perform the further pre-processing
    for ch in ALL_CHANNELS:

        # get the LFP from the channel
        ch_lfp = structured_lfp_data.loc[structured_lfp_data.channel == ch]
        ch_lfp = ch_lfp.lfp.values[0]

        filtered_signal = band_pass_filter(ch_lfp)

        fourier_transformed_lfp = fourier_transform(filtered_signal)

        f = fourier_transformed_lfp["freq"]
        psd = fourier_transformed_lfp["average_Sxx"]

        lfp_per_ch = {
            "channel": [ch],
            "unfiltered_lfp": [ch_lfp],
            "filtered_lfp": [filtered_signal],
            "frequencies": [f],
            "filtered_psd": [psd]
        }

        lfp_per_ch_df = pd.DataFrame(lfp_per_ch)
        lfp_psd_data = pd.concat([lfp_psd_data, lfp_per_ch_df], ignore_index=True)

        # extract power average in frequencies
        power_average = power_average_in_freq(f=f, psd=psd)

        # extract peak parameters
        all_peaks = peak_detection(psd=psd, f=f)

        for range in BETA_RANGES:

            power_av_in_range = power_average[range]
            peak_details_in_range = all_peaks.loc[all_peaks["f_range"] == range]

            # check if peaks exist
            if len(peak_details_in_range["peak_CF"].values) == 0:
                peak_CF = None
                peak_power = None
                peak_4Hz_power = None

            else:
                peak_CF = peak_details_in_range["peak_CF"].values[0]
                peak_power = peak_details_in_range["peak_power"].values[0]
                peak_4Hz_power = peak_details_in_range["peak_4Hz_power"].values[0]

            beta_profile_per_ch_range = {
                "channel": [ch],
                "f_range": [range],
                "power_in_f_range": [power_av_in_range],
                "peak_CF": [peak_CF],
                "peak_power": [peak_power],
                "peak_4Hz_power": [peak_4Hz_power]
            }

            single_beta_profile = pd.DataFrame(beta_profile_per_ch_range)
            beta_profile = pd.concat([beta_profile, single_beta_profile], ignore_index=True)

    
    return beta_profile, lfp_psd_data




