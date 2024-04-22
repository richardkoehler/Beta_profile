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

            signal_structured = {
                "channel": [ch],
                "lfp": [lfp_data.get_data()[i, :]],
                "lfp_group": [group]
            }

            signals_df = pd.DataFrame(signal_structured)
            structured_signals_dataframe = pd.concat([structured_signals_dataframe, signals_df], ignore_index=True)
        
    
    return structured_signals_dataframe








def spectrogram_Psd_onlyONEsession(incl_sub: str, 
                                   incl_session: list, 
                                   incl_condition: list, 
                                   pickChannels: list, 
                                   hemisphere: str, 
                                   filter: str):
    """

    Input: 
        - incl_sub: str e.g. "024"
        - incl_session: list ["postop", "fu3m", "fu12m", "fu18m", "fu24m"]
        - incl_condition: list e.g. ["m0s0", "m1s0"]
        - pickChannels: list of bipolar channels, depending on which incl_contact was chosen
                        Ring: ['03', '13', '02', '12', '01', '23']
                        SegmIntra: ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                        SegmInter: ['1A2A', '1B2B', '1C2C']
        - hemisphere: str e.g. "Right"
        - normalization: str "rawPSD", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - filter: str "unfiltered", "band-pass"

    
    1) load data from main_class.PerceiveData using the input values.

    2) pick channels
    
    3) if filter == "band-pass": band-pass filter by a Butterworth Filter of fifth order (5-95 Hz).
    
    4) Calculate the raw psd values of every channel for each timepoint by using scipy.sinal.scpectrogram.
        - Compute a spectrogram with consecutive Fourier transforms.
        - hanning window (scipy.signal.hann):
            - sampling frequency: 250 Hz
            - window samples: 250
            - sym=False
            - noverlap: 0.5 (50% overlap of windows)

        output variables:
        - f = frequencies 0-125 Hz (Maximum = Nyquist frequency = sfreq/2)
        - time_sectors = sectors 0.5 - 20.5 s in 0.5 steps (21 time sectors)
        - Sxx = 126 frequency rows (arrays with 21 PSD values [µV^2/Hz] of each time sector, 21 time sector columns
    
    5) Normalization variants: calculate different normalized PSD values 
        - normalized to total sum of PSD from each power spectrum
        - normalized to sum of PSD from 1-100 Hz
        - normalized to sum of PSD from 40-90 Hz


    Depending on normalization variation: 
    
    6) For each frequency band alpha (8-12 Hz), low beta (13-20 Hz), high beta (21-35 Hz), beta (13-35 Hz), gamma (40-90 Hz) the highest Peak values (frequency and psd) will be seleted and saved in a DataFrame.

    7) The raw or noramlized PSD values will be plotted and the figure will be saved as:
        f"\sub{incl_sub}_{hemisphere}_normalizedPsdToTotalSum_seperateTimepoints_{pickChannels}.png"
    
    8) All frequencies and relative psd values, as well as the values for the highest PEAK in each frequency band will be returned as a Dataframe in a dictionary: 
    
    return {
        "rawPsdDataFrame":rawPSDDataFrame,
        "normPsdToTotalSumDataFrame":normToTotalSumPsdDataFrame,
        "normPsdToSum1_100Hz": normToSum1_100Hz,
        "normPsdToSum40_90Hz":normToSum40_90Hz,
        "psdAverage_dict": psdAverage_dict,
        "highestPeakRawPSD": highestPeakRawPsdDF,
    }
    Watchout: I changed filenames -> now also including filter information!!!

    """

    # sns.set()
    plt.style.use('seaborn-whitegrid')  

    

    # add error correction for sub and task??
    
    f_rawPsd_dict = {} # dictionary with tuples of frequency and psd for each channel and timepoint of a subject
    f_normPsdToTotalSum_dict = {}
    f_normPsdToSum1to100Hz_dict = {}
    f_normPsdToSum40to90Hz_dict = {}
    psdAverage_dict = {}
    highest_peak_dict = {}

    # loop through all normalizations to get all values
    normalization_list = ["rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"]

    for n, norm in enumerate(normalization_list):

        # set layout for figures: using the object-oriented interface
        fig = plt.figure() # subplot(rows, columns, panel number), figsize(width,height)
        ax = fig.add_subplot()
        

        # Create a list of 15 colors and add it to the cycle of matplotlib 
        cycler_colors = cycler("color", ["blue", "navy", "deepskyblue", "purple", "green", "darkolivegreen", "magenta", "orange", "red", "darkred", "chocolate", "gold", "cyan",  "yellow", "lime"])
        plt.rc('axes', prop_cycle=cycler_colors)


        for t, tp in enumerate(incl_session):
            # t is indexing time_points, tp are the time_points

            for c, cond in enumerate(incl_condition):

                for cont, contact in enumerate(incl_contact[f"{hemisphere}"]): 
                    # tk is indexing task, task is the input task

                    # avoid Attribute Error, continue if attribute doesn´t exist
                    if getattr(mainclass_sub.survey, tp) is None:
                        continue

                
                    # apply loop over channels
                    temp_data = getattr(mainclass_sub.survey, tp) # gets attribute e.g. of tp "postop" from modality_class with modality set to survey
                    
                    # avoid Attribute Error, continue if attribute doesn´t exist
                    if getattr(temp_data, cond) is None:
                        continue
                
                    # try:
                    #     temp_data = getattr(temp_data, cond)
                    #     temp_data = temp_data.rest.data[tasks[tk]]
                    
                    # except AttributeError:
                    #     continue

                    temp_data = getattr(temp_data, cond) # gets attribute e.g. "m0s0"
                    temp_data = getattr(temp_data.rest, contact)
                    temp_data = temp_data.run1.data # gets the mne loaded data from the perceive .mat BSSu, m0s0 file with task "RestBSSuRingR"
        

                    #################### CREATE A BUTTERWORTH FILTER ####################
                    # sampling frequency: 250 Hz
                    fs = temp_data.info['sfreq']

                    # only if filter == "band-pass"
                    if filter == "band-pass":

                        # set filter parameters for band-pass filter
                        filter_order = 5 # in MATLAB spm_eeg_filter default=5 Butterworth
                        frequency_cutoff_low = 5 # 5Hz high-pass filter
                        frequency_cutoff_high = 95 # 95 Hz low-pass filter

                        # create the filter
                        b, a = scipy.signal.butter(filter_order, (frequency_cutoff_low, frequency_cutoff_high), btype='bandpass', output='ba', fs=fs)
        
                    else:
                        print("no filter applied")
                    
                    
                    # get new channel names
                    ch_names = temp_data.info.ch_names


                    #################### PICK CHANNELS ####################
                    include_channelList = [] # this will be a list with all channel names selected
                    exclude_channelList = []

                    for n, names in enumerate(ch_names):
                        
                        # add all channel names that contain the picked channels: e.g. 02, 13, etc given in the input pickChannels
                        for picked in pickChannels:
                            if picked in names:
                                include_channelList.append(names)


                        # exclude all bipolar 0-3 channels, because they do not give much information
                        # if "03" in names:
                        #     exclude_channelList.append(names)
                        
                    # Error Checking: 
                    if len(include_channelList) == 0:
                        continue

                    # pick channels of interest: mne.pick_channels() will output the indices of included channels in an array
                    ch_names_indices = mne.pick_channels(ch_names, include=include_channelList)

                    
                    for i, ch in enumerate(ch_names):
                        
                        # only get picked channels
                        if i not in ch_names_indices:
                            continue

                        #################### FILTER ####################
                        signal = {}
                        if filter == "band-pass":
                            # filter the signal by using the above defined butterworth filter
                            signal["band-pass"] = scipy.signal.filtfilt(b, a, temp_data.get_data()[i, :]) 
                        
                        elif filter == "unfiltered": 
                            signal["unfiltered"] = temp_data.get_data()[i, :]

                        #################### PERFORM FOURIER TRANSFORMATION AND CALCULATE POWER SPECTRAL DENSITY ####################

                        window = 250 # with sfreq 250 frequencies will be from 0 to 125 Hz, 125Hz = Nyquist = fs/2
                        noverlap = 0.5 # 50% overlap of windows

                        window = hann(window, sym=False)

                        # compute spectrogram with Fourier Transforms
                        
                        f,time_sectors,Sxx = scipy.signal.spectrogram(x=signal[f"{filter}"], fs=fs, window=window, noverlap=noverlap,  scaling='density', mode='psd', axis=0)
                        # f = frequencies 0-125 Hz (Maximum = Nyquist frequency = sfreq/2)
                        # time_sectors = sectors 0.5 - 20.5 s in 0.5 steps (in total 21 time sectors)
                        # Sxx = 126 arrays with 21 values each of PSD [µV^2/Hz], for each frequency bin PSD values of each time sector
                        # Sxx = 126 frequency rows, 21 time sector columns

                        # average all 21 Power spectra of all time sectors 
                        average_Sxx = np.mean(Sxx, axis=1) # axis = 1 -> mean of each column: in total 21x126 mean values for each frequency
                                    

                        #################### CALCULATE THE STANDARD ERROR OF MEAN ####################
                        # SEM = standard deviation / square root of sample size
                        Sxx_std = np.std(Sxx, axis=1) # standard deviation of each frequency row
                        semRawPsd = Sxx_std / np.sqrt(Sxx.shape[1]) # sample size = 21 time vectors -> sem with 126 values

                        # store frequency, time vectors and psd values in a dictionary, together with session timepoint and channel
                        f_rawPsd_dict[f'{tp}_{ch}'] = [tp, ch, f, time_sectors, average_Sxx, semRawPsd] 
                    

                        #################### NORMALIZE PSD IN MULTIPLE WAYS ####################
                        
                        #################### NORMALIZE PSD TO TOTAL SUM OF THE POWER SPECTRUM (ALL FREQUENCIES) ####################

                        normToTotalSum_psd = (average_Sxx/np.sum(average_Sxx))*100 # in percentage               
                        # calculate the SEM of psd values 
                        semNormToTotalSum_psd = (semRawPsd/np.sum(average_Sxx))*100

                        # store frequencies and normalized psd values and sem of normalized psd in a dictionary
                        f_normPsdToTotalSum_dict[f'{tp}_{ch}'] = [tp, ch, f, time_sectors, normToTotalSum_psd, semNormToTotalSum_psd]


                        #################### NORMALIZE PSD TO SUM OF PSD BETWEEN 1-100 Hz  ####################
                    
                        # get raw psd values from 1 to 100 Hz by indexing the numpy arrays f and px
                        rawPsd_1to100Hz = average_Sxx[1:100]

                        # sum of rawPSD between 1 and 100 Hz
                        psdSum1to100Hz = rawPsd_1to100Hz.sum()

                        # raw psd divided by sum of psd between 1 and 100 Hz
                        normPsdToSum1to100Hz = (average_Sxx/psdSum1to100Hz)*100

                        # calculate the SEM of psd values 
                        semNormPsdToSum1to100Hz = (semRawPsd/psdSum1to100Hz)*100

                        # store frequencies and normalized psd values and sem of normalized psd in a dictionary
                        f_normPsdToSum1to100Hz_dict[f'{tp}_{ch}'] = [tp, ch, f, time_sectors, normPsdToSum1to100Hz, semNormPsdToSum1to100Hz]


                        #################### NORMALIZE PSD TO SUM OF PSD BETWEEN 40-90 Hz  ####################
                    
                        # get raw psd values from 40 to 90 Hz (gerundet) by indexing the numpy arrays f and px
                        rawPsd_40to90Hz = average_Sxx[40:90] 

                        # sum of rawPSD between 40 and 90 Hz
                        psdSum40to90Hz = rawPsd_40to90Hz.sum()

                        # raw psd divided by sum of psd between 40 and 90 Hz
                        normPsdToSum40to90Hz = (average_Sxx/psdSum40to90Hz)*100
                    
                        # calculate the SEM of psd values 
                        semNormPsdToSum40to90Hz = (semRawPsd/psdSum40to90Hz)*100

                        # store frequencies and normalized psd values and sem of normalized psd in a dictionary
                        f_normPsdToSum40to90Hz_dict[f'{tp}_{ch}'] = [tp, ch, f, time_sectors, normPsdToSum40to90Hz, semNormPsdToSum40to90Hz]


                        #################### PSD average and PEAK DETECTION ####################
                        # depending on what normalization or raw was chosen: define variables for psd, sem and ylabel accordingly
                        if norm == "rawPsd":
                            chosenPsd = average_Sxx
                            chosenSem = semRawPsd
                            chosen_ylabel = "uV^2/Hz+-SEM"
                            chosen_ylim = [0, 9]
                        
                        elif norm == "normPsdToTotalSum":
                            chosenPsd = normToTotalSum_psd
                            chosenSem = semNormToTotalSum_psd
                            chosen_ylabel = "PSD to total sum[%]+-SEM"
                            chosen_ylim = [0, 14]

                        elif norm == "normPsdToSum1_100Hz":
                            chosenPsd = normPsdToSum1to100Hz
                            chosenSem = semNormPsdToSum1to100Hz
                            chosen_ylabel = "PSD to sum 1-100 Hz[%]+-SEM"
                            chosen_ylim = [0, 14]

                        elif norm == "normPsdToSum40_90Hz":
                            chosenPsd = normPsdToSum40to90Hz
                            chosenSem = semNormPsdToSum40to90Hz
                            chosen_ylabel = "PSD to sum 40-90 Hz[%]+-SEM"
                            chosen_ylim = [0, 200]
                        
                        else:
                            chosenPsd = average_Sxx
                            chosenSem = semRawPsd
                            chosen_ylabel = "uV^2/Hz+-SEM"
                        # else statement is necessary to ensure the definition variable is not only locally 
                            

                        #################### PSD AVERAGE OF EACH FREQUENCY BAND DEPENDING ON CHOSEN PSD NORMALIZATION ####################
                        
                        # create booleans for each frequency-range for alpha, low beta, high beta, beta and gamma
                        alpha_frequency = (f >= 8) & (f <= 12) # alpha_range will output a boolean of True values within the alpha range
                        lowBeta_frequency = (f >= 13) & (f <= 20)
                        highBeta_frequency = (f >= 21) & (f <= 35)
                        beta_frequency = (f >= 13) & (f <= 35)
                        narrowGamma_frequency = (f >= 40) & (f <= 90)

                        # make a list with all boolean masks of each frequency, so I can loop through
                        range_allFrequencies = [alpha_frequency, lowBeta_frequency, highBeta_frequency, beta_frequency, narrowGamma_frequency]

                        # loop through frequency ranges and get all psd values of each frequency band
                        for count, boolean in enumerate(range_allFrequencies):

                            frequency = []
                            if count == 0:
                                frequency = "alpha"
                            elif count == 1:
                                frequency = "lowBeta"
                            elif count == 2:
                                frequency = "highBeta"
                            elif count == 3:
                                frequency = "beta"
                            elif count == 4:
                                frequency = "narrowGamma"
                            


                            # get all frequencies and chosen psd values within each frequency range
                            # frequencyInFreqBand = f[range_allFrequencies[count]] # all frequencies within a frequency band
                            psdInFreqBand = chosenPsd[range_allFrequencies[count]] # all psd values within a frequency band

                            psdAverage = np.mean(psdInFreqBand)

                            # store averaged psd values of each frequency band in a dictionary
                            psdAverage_dict[f'{tp}_{ch}_psdAverage_{norm}_{frequency}'] = [tp, ch, frequency, norm, psdAverage]



                        #################### PEAK DETECTION PSD DEPENDING ON CHOSEN PSD NORMALIZATION ####################
                        # find all peaks: peaks is a tuple -> peaks[0] = index of frequency?, peaks[1] = dictionary with keys("peaks_height") 
                        peaks = scipy.signal.find_peaks(chosenPsd, height=0.1) # height: peaks only above 0.1 will be recognized

                        # Error checking: if no peaks found, continue
                        if len(peaks) == 0:
                            continue

                        peaks_height = peaks[1]["peak_heights"] # np.array of y-value of peaks = power
                        peaks_pos = f[peaks[0]] # np.array of indeces on x-axis of peaks = frequency

                        # set the x-range for each frequency band
                        alpha_range = (peaks_pos >= 8) & (peaks_pos <= 12) # alpha_range will output a boolean of True values within the alpha range
                        lowBeta_range = (peaks_pos >= 13) & (peaks_pos <= 20)
                        highBeta_range = (peaks_pos >= 21) & (peaks_pos <= 35)
                        beta_range = (peaks_pos >= 13) & (peaks_pos <= 35)
                        narrowGamma_range = (peaks_pos >= 40) & (peaks_pos <= 90)

                        # make a list with all boolean masks of each frequency, so I can loop through
                        frequency_ranges = [alpha_range, lowBeta_range, highBeta_range, beta_range, narrowGamma_range]

                        # loop through frequency ranges and get the highest peak of each frequency band
                        for count, boolean in enumerate(frequency_ranges):

                            frequency = []
                            if count == 0:
                                frequency = "alpha"
                            elif count == 1:
                                frequency = "lowBeta"
                            elif count == 2:
                                frequency = "highBeta"
                            elif count == 3:
                                frequency = "beta"
                            elif count == 4:
                                frequency = "narrowGamma"
                            
                            # get all peak positions and heights within each frequency range
                            peaksinfreq_pos = peaks_pos[frequency_ranges[count]]
                            peaksinfreq_height = peaks_height[frequency_ranges[count]]

                            # Error checking: check first, if there is a peak in the frequency range
                            if len(peaksinfreq_height) == 0:
                                continue

                            # select only the highest peak within the alpha range
                            highest_peak_height = peaksinfreq_height.max()

                            ######## calculate psd average of +- 2 Hz from highest Peak ########
                            # 1) find psd values from -2Hz until + 2Hz from highest Peak by slicing and indexing the numpy array of all chosen psd values
                            peakIndex = np.where(chosenPsd == highest_peak_height) # np.where output is a tuple: index, dtype
                            peakIndexValue = peakIndex[0].item() # only take the index value of the highest Peak psd value in all chosen psd

                            # 2) go -2 and +3 indeces 
                            indexlowCutt = peakIndexValue-2
                            indexhighCutt = peakIndexValue+3   # +3 because the ending index is left out when slicing a numpy array

                            # 3) slice the numpy array of all chosen psd values, only get values from -2 until +2 Hz from highest Peak
                            psdArray5HzRangeAroundPeak = chosenPsd[indexlowCutt:indexhighCutt] # array only of psd values -2 until +2Hz around Peak = 5 values

                            # 4) Average of 5Hz Array
                            highest_peak_height_5Hzaverage = np.mean(psdArray5HzRangeAroundPeak)                       



                            # get the index of the highest peak y value to get the corresponding peak position x
                            ix = np.where(peaksinfreq_height == highest_peak_height)
                            highest_peak_pos = peaksinfreq_pos[ix].item()

                            # plot only the highest peak within each frequency band
                            plt.scatter(highest_peak_pos, highest_peak_height, color="k", s=15, marker='D')

                            # store highest peak values of each frequency band in a dictionary
                            highest_peak_dict[f'{tp}_{ch}_highestPEAK_{norm}_{frequency}'] = [tp, ch, frequency, norm, highest_peak_pos, highest_peak_height, highest_peak_height_5Hzaverage]




                        #################### PLOT THE CHOSEN PSD DEPENDING ON NORMALIZATION INPUT ####################

                        # the title of each plot is set to the timepoint e.g. "postop"
                        plt.title(tp, fontsize=15) 

                        # get y-axis label and limits
                        # axes[t].get_ylabel()
                        # axes[t].get_ylim()

                        # .plot() method for creating the plot, axes[0] refers to the first plot, the plot is set on the appropriate object axes[t]
                        plt.plot(f, chosenPsd, label=f"{ch}_{cond}")  # or np.log10(px) 
                        # colors of each line in different color, defined at the beginning
                        # axes[t].plot(f, chosenPsd, label=f"{ch}_{cond}", color=colors[i])

                        # make a shadowed line of the sem
                        plt.fill_between(f, chosenPsd-chosenSem, chosenPsd+chosenSem, color='lightgray', alpha=0.5)



        #################### PLOT SETTINGS ####################
        fig.suptitle(f"PowerSpectra sub{incl_sub} {hemisphere} hemisphere, Filter: {filter}", ha="center", fontsize= 20)
        plt.subplots_adjust(wspace=0, hspace=0)
    
        font = {"size": 20}

        # ax.legend(loc= 'upper right') # Legend will be in upper right corner
        ax.grid() # show grid

        # different xlim depending on filtered or unfiltered signal
        if filter == "band-pass":
            plt.xlim([3, 50]) # no ylim for rawPSD and normalization to sum 40-90 Hz

        elif filter == "unfiltered":
            plt.xlim([-2, 50])

        # ax.set(xlim=[-5, 60] ,ylim=[0,7]) for normalizations to total sum or to sum 1-100Hz set ylim to zoom in
        plt.xlabel("Frequency", fontsize=12)
        plt.ylabel(chosen_ylabel, fontsize=12)
        plt.ylim(chosen_ylim)

        plt.axvline(x=8, color='black', linestyle='--')
        plt.axvline(x=13, color='black', linestyle='--')
        plt.axvline(x=20, color='black', linestyle='--')
        plt.axvline(x=35, color='black', linestyle='--')
    
    

        ###### LEGEND ######
        legend = plt.legend(loc= 'upper right', edgecolor="black") #bbox_to_anchor=(1.5, -0.1)) 
        # frame the legend with black edges amd white background color 
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor("white")

        fig.tight_layout()

        fig.savefig(figures_path + f"\\PSDspectrogram_sub{incl_sub}_{hemisphere}_{norm}_{filter}.png", bbox_inches ="tight")
                            

    #################### WRITE DATAFRAMES TO STORE VALUES ####################
    # write raw PSD Dataframe
    rawPSDDataFrame = pd.DataFrame(f_rawPsd_dict)
    rawPSDDataFrame.rename(index={0: "session", 1: "bipolarChannel", 2: "frequency", 3: "time_sectors", 4: "rawPsd", 5: "SEM_rawPsd"}, inplace=True) # rename the rows
    rawPSDDataFrame = rawPSDDataFrame.transpose() # Dataframe with 5 columns and rows for each single power spectrum

    # write DataFrame of normalized PSD to total Sum
    normPsdToTotalSumDataFrame = pd.DataFrame(f_normPsdToTotalSum_dict) # Dataframe of normalised to total sum psd: columns=single bipolar channel of one session
    normPsdToTotalSumDataFrame.rename(index={0: "session", 1: "bipolarChannel", 2: "frequency", 3: "time_sectors", 4: "normPsdToTotalSum", 5: "SEM_normPsdToTotalSum"}, inplace=True) # rename the rows
    normPsdToTotalSumDataFrame = normPsdToTotalSumDataFrame.transpose() # Dataframe with 5 columns and rows for each single power spectrum

    # write DataFrame of normalized PSD to Sum of PSD between 1 and 100 Hz
    normPsdToSum1to100HzDataFrame = pd.DataFrame(f_normPsdToSum1to100Hz_dict) # Dataframe of normalised to total sum psd: columns=single bipolar channel of one session
    normPsdToSum1to100HzDataFrame.rename(index={0: "session", 1: "bipolarChannel", 2: "frequency", 3: "time_sectors", 4: "normPsdToSumPsd1to100Hz", 5: "SEM_normPsdToSumPsd1to100Hz"}, inplace=True) # rename the rows
    normPsdToSum1to100HzDataFrame = normPsdToSum1to100HzDataFrame.transpose() # Dataframe with 5 columns and rows for each single power spectrum

    # write DataFrame of normalized PSD to Sum of PSD between 1 and 100 Hz
    normPsdToSum40to90DataFrame = pd.DataFrame(f_normPsdToSum40to90Hz_dict) # Dataframe of normalised to total sum psd: columns=single bipolar channel of one session
    normPsdToSum40to90DataFrame.rename(index={0: "session", 1: "bipolarChannel", 2: "frequency", 3: "time_sectors", 4: "normPsdToSum40to90Hz", 5: "SEM_normPsdToSum40to90Hz"}, inplace=True) # rename the rows
    normPsdToSum40to90DataFrame = normPsdToSum40to90DataFrame.transpose() # Dataframe with 5 columns and rows for each single power spectrum



    # write DataFrame of averaged psd values in each frequency band depending on the chosen normalization
    psdAverageDF = pd.DataFrame(psdAverage_dict) # Dataframe with 5 rows and columns for each single power spectrum
    psdAverageDF.rename(index={0: "session", 1: "bipolarChannel", 2: "frequencyBand", 3: "absoluteOrRelativePSD", 4: "averagedPSD"}, inplace=True) # rename the rows
    psdAverageDF = psdAverageDF.transpose() # Dataframe with 4 columns and rows for each single power spectrum


    # write DataFrame of frequency and psd values of the highest peak in each frequency band
    highestPEAKDF = pd.DataFrame(highest_peak_dict) # Dataframe with 5 rows and columns for each single power spectrum
    highestPEAKDF.rename(index={0: "session", 1: "bipolarChannel", 2: "frequencyBand", 3: "absoluteOrRelativePSD", 4: "PEAK_frequency", 5: "PEAK_amplitude", 6: "PEAK_5HzAverage"}, inplace=True) # rename the rows
    highestPEAKDF = highestPEAKDF.transpose() # Dataframe with 6 columns and rows for each single power spectrum


    # save Dataframes as csv in the results folder
    
    # rawPSDDataFrame.to_json(os.path.join(results_path,f"SPECTROGRAMrawPSD_{hemisphere}_{filter}"), sep=",")
    # normPsdToTotalSumDataFrame.to_json(os.path.join(results_path,f"SPECTROGRAMnormPsdToTotalSum_{hemisphere}_{filter}"), sep=",")
    # normPsdToSum1to100HzDataFrame.to_json(os.path.join(results_path,f"SPECTROGRAMnormPsdToSum_1to100Hz_{hemisphere}_{filter}"), sep=",")
    # normPsdToSum40to90DataFrame.to_json(os.path.join(results_path,f"SPECTROGRAMnormPsdToSum_40to90Hz_{hemisphere}_{filter}"), sep=",")
    psdAverageDF.to_json(os.path.join(results_path,f"SPECTROGRAMpsdAverageFrequencyBands_{hemisphere}_{filter}.json"))
    highestPEAKDF.to_json(os.path.join(results_path,f"SPECTROGRAM_highestPEAK_FrequencyBands_{hemisphere}_{filter}.json"))

    # concatenate the PSD Dataframes to one and take out the Duplicated columns
    PSD_Dataframe = pd.concat([rawPSDDataFrame, normPsdToTotalSumDataFrame, normPsdToSum1to100HzDataFrame, normPsdToSum40to90DataFrame], axis=1)
    PSD_Dataframe = PSD_Dataframe.loc[:,~PSD_Dataframe.columns.duplicated()]
    PSD_Dataframe.to_json(os.path.join(results_path,f"SPECTROGRAMPSD_{hemisphere}_{filter}.json"))


    return {
        f"PSD_Dataframe": PSD_Dataframe,
        f"rawPsdDataFrame_{filter}":rawPSDDataFrame,
        f"normPsdToTotalSumDataFrame_{filter}":normPsdToTotalSumDataFrame,
        f"normPsdToSum1to100HzDataFrame_{filter}":normPsdToSum1to100HzDataFrame,
        f"normPsdToSum40to90HzDataFrame_{filter}":normPsdToSum40to90DataFrame,
        f"averagedPSD_{filter}": psdAverageDF,
        f"highestPEAK_{filter}": highestPEAKDF,
    }



