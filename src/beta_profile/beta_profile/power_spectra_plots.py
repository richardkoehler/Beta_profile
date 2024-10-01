""" Time Series and Power Spectra Plots """

import matplotlib.pyplot as plt
from cycler import cycler

from ..utils import find_folders as find_folders
from ..utils import io as io
from . import tfr_preprocessing as tfr_preprocessing


COLORS = [
    "blue",
    "orange",
    "green",
    "darkred",
    "purple",
    "lime",
    "navy",
    "gold",
    "red",
    "deepskyblue",
    "yellow",
    "magenta",
    "cyan",
    "darkolivegreen",
    "chocolate",
]

PICK_CHANNELS = {
    "Ring_neighbours": ["01", "12", "23"],
    "Ring_sandwich": ["02", "13"],
    "Segm": ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C", "1A2A", "1B2B", "1C2C"],
}

SESSION_FILENAME = {
    "fu3m": "3MFU",
    "fu12m": "12MFU",
    "fu18m": "18MFU",
    "fu24m": "24MFU",
    "fu36m": "36MFU",
}

CONDITION_FILENAME = {
    "m0s0": "MedOFF-StimOFF",
    "m0s1": "MedOFF-StimON",
    "m1s0": "MedON-StimOFF",
    "m1s1": "MedON-StimON",
}


def plot_power_spectra(sub: str, session: str, condition: str, hemisphere: str):
    """
    Plot Power spectra of band-pass filtered LFP in 2 separate plots:
        - Ring Plot
        - Segm Plot

    """
    # load psd
    beta_profile = tfr_preprocessing.main_tfr(
        sub=sub, session=session, condition=condition, hemisphere=hemisphere
    )

    peak_details = beta_profile[0]
    power_details = beta_profile[1]

    # plot separately Ring and Segm
    ch_group = ["Ring_neighbours", "Ring_sandwich", "Segm"]
    for group in ch_group:

        channels = PICK_CHANNELS[group]

        # figure settings
        fig = (
            plt.figure()
        )  # subplot(rows, columns, panel number), figsize(width,height)
        # ax.grid()

        fig.suptitle(
            f"Power Spectra {hemisphere} hemisphere ({group}), \n{SESSION_FILENAME[session]}, {CONDITION_FILENAME[condition]}",
            ha="center",
            fontsize=20,
        )
        plt.subplots_adjust(wspace=0, hspace=0)

        # Create a list of 15 colors and add it to the cycle of matplotlib
        cycler_colors = cycler("color", COLORS)
        plt.rc("axes", prop_cycle=cycler_colors)

        for chan in channels:

            # get f and filtered psd
            power_details_ch = power_details.loc[power_details.channel == chan]
            filtered_psd = power_details_ch.filtered_psd.values[0]
            frequencies = power_details_ch.frequencies.values[0]

            # get the beta peak parameters
            peak_details_ch = peak_details.loc[peak_details.channel == chan]
            beta_peak_details_ch = peak_details_ch.loc[
                peak_details_ch.f_range == "beta"
            ]

            # check if peak exists
            if len(beta_peak_details_ch.peak_CF.values) == 0:
                print(f"No Beta Peak in channel {chan}")
                beta_peak_CF = 0
                beta_peak_power = 0

            else:
                beta_peak_CF = beta_peak_details_ch.peak_CF.values[0]
                beta_peak_power = beta_peak_details_ch.peak_power.values[0]

            # plot each channel
            plt.plot(frequencies, filtered_psd, label=chan)
            plt.scatter(beta_peak_CF, beta_peak_power, color="k", s=15, marker="D")

        # ax.set(xlim=[-5, 60] ,ylim=[0,7]) for normalizations to total sum or to sum 1-100Hz set ylim to zoom in
        plt.xlabel("Frequency", fontsize=12)
        plt.ylabel("PSD", fontsize=12)

        plt.axvline(x=13, color="black", linestyle="--")
        plt.axvline(x=20, color="black", linestyle="--")
        plt.axvline(x=35, color="black", linestyle="--")

        ###### LEGEND ######
        legend = plt.legend(
            loc="upper right", edgecolor="black"
        )  # bbox_to_anchor=(1.5, -0.1))
        # frame the legend with black edges amd white background color
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor("white")

        fig.tight_layout()

        # save figure
        io.save_fig_jpeg(
            sub=sub,
            filename=f"Power_Spectra_{hemisphere}_{group}_{session}_{condition}",
            figure=fig,
        )
