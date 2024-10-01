""" ECG artifact cleaning """

import matplotlib.pyplot as plt

from ..utils import io as io
from . import tfr_preprocessing as tfr_preprocessing


HEMISPHERES = ["Right", "Left"]


PICK_CHANNELS = {
    "Ring": ["01", "12", "23", "02", "13"],
    "SegmInter": ["1A2A", "1B2B", "1C2C"],
    "SegmIntra": ["1A1B", "1A1C", "1B1C", "2A2B", "2A2C", "2B2C"],
}

CH_GROUPS = ["Ring", "SegmInter", "SegmIntra"]


def plot_ieeg_data(sub: str, hemisphere: str, condition: str, session: str):
    """
    Function to plot the iEEG data. This function can be used when you have already extracted the data into a 2D array.

    Input:
        - ieeg_data: np.array -> 2D array shape: (n_channels, n_samples)
    """

    try:
        plt.style.use("seaborn-whitegrid")
    except OSError as e:
        if "'seaborn-whitegrid' is not a valid package style" not in str(e):
            raise e
        plt.style.use("seaborn-v0_8-whitegrid")

    # load psd
    beta_profile = tfr_preprocessing.main_tfr(
        sub=sub, session=session, condition=condition, hemisphere=hemisphere
    )

    power_details = beta_profile[1]

    for group in CH_GROUPS:

        channels = PICK_CHANNELS[group]
        fig_size = (40, 30)

        fig, axes = plt.subplots(
            len(channels), 1, figsize=fig_size
        )  # subplot(rows, columns, panel number), figsize(width,height)
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.suptitle(
            f"Unfiltered time series sub-{sub}, {hemisphere} hemisphere, {session} session, {condition}, {group}",
            ha="center",
            fontsize=40,
        )

        for i, ch in enumerate(channels):

            ch_unfiltered_lfp = power_details.loc[power_details.channel == ch]
            ch_unfiltered_lfp = ch_unfiltered_lfp.unfiltered_lfp.values[0]

            #################### PLOT THE CHOSEN PSD DEPENDING ON NORMALIZATION INPUT ####################

            axes[i].set_title(f"Channel {ch}", fontsize=30)
            axes[i].plot(ch_unfiltered_lfp, label=f"{ch}", color="k", linewidth=0.5)

        for ax in axes:
            ax.set_xlabel("timestamp", fontsize=30)
            ax.set_ylabel("amplitude", fontsize=30)
            ax.tick_params(axis="both", which="major", labelsize=30)

        for ax in axes.flat[:-1]:
            ax.set(xlabel="")

        fig.tight_layout()

        # save figure
        filename = f"unfiltered_time_series_sub-{sub}_{hemisphere}_{condition}_{group}"
        io.save_fig_jpeg(sub=sub, filename=filename, figure=fig)
