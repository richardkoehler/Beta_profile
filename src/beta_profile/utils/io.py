""" Loads and saves data"""

import os

import pandas as pd

# PyPerceive Imports
from PerceiveImport.classes import main_class

from ..utils import find_folders as find_folders

LFP_GROUPS = {
    "Right": ["RingR", "SegmIntraR", "SegmInterR"],
    "Left": ["RingL", "SegmIntraL", "SegmInterL"],
}


def load_sub_path(sub: str):
    """
    Loading the path to the diectory:
        - ercept_Data_structured/beta_data/sub-XXX

    Input:
        - sub: "024"
    """

    sub_path = find_folders.get_onedrive_path(folder="beta_data")
    sub_path = os.path.join(sub_path, f"sub-{sub}")

    return sub_path


def check_or_create_sub_path(sub: str):
    """
    Checks if the sub-XXX folder exists and creates it if it doesn't
    """

    sub_path = load_sub_path(sub=sub)

    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

    return sub_path


def load_py_perceive_object(sub: str, session: str, condition: str, hemisphere: str):
    """
    Loading the MNE object of the BrainSense Survey of the given input through PyPerceive

    """
    return main_class.PerceiveData(
        sub=sub,
        incl_modalities=["survey"],
        incl_session=[session],
        incl_condition=[condition],
        incl_task=["rest"],
        incl_contact=LFP_GROUPS[hemisphere],
    )


def extract_data_from_py_perceive(
    sub: str,
    session: str,
    condition: str,
    hemisphere: str,
):
    """
    This function first checks if the data exists and then extracts the LFP of interest from the PyPerceive MNE object
    """

    lfp_group_data = {}

    # load the MNE object
    mainclass_object = load_py_perceive_object(
        sub=sub, session=session, condition=condition, hemisphere=hemisphere
    )

    for lfp_group in LFP_GROUPS[hemisphere]:

        # check if attributes exist
        if getattr(mainclass_object.survey, session) is None:
            print(f"session {session} doesn't exist for sub-{sub}")

        else:
            lfp_data = getattr(mainclass_object.survey, session)

        if getattr(lfp_data, condition) is None:
            print(
                f"condition {condition} doesn't exist for sub-{sub}, session {session}"
            )

        else:
            lfp_data = getattr(lfp_data, condition)
            lfp_data = getattr(lfp_data.rest, lfp_group)
            lfp_data = (
                lfp_data.run1.data
            )  # gets the mne loaded data from the perceive .mat BSSu, m0s0 file

            # save in a dictionary with keys "RingR", "SegmIntraR", "SegmInterR"
            lfp_group_data[lfp_group] = lfp_data

    return lfp_group_data


def save_fig_jpeg(sub: str, filename: str, figure=None):
    """
    Input:
        - path: str
        - filename: str
        - figure: must be a plt figure

    """
    path = check_or_create_sub_path(sub=sub)

    figure.savefig(
        os.path.join(path, f"{filename}.jpg"),
        bbox_inches="tight",
        format="jpeg",
        dpi=300,
    )

    print(f"Figure {filename}.jpg", f"\nwere written in: {path}.")


def save_df_as_excel(sub: str, filename: str, file: pd.DataFrame, sheet_name: str):
    """ """

    path = check_or_create_sub_path(sub=sub)
    filepath = os.path.join(path, f"{filename}.xlsx")

    file.to_excel(filepath, sheet_name=sheet_name, index=False)


def save_df_to_excel_sheets(sub: str, filename: str, file: dict):
    """ """
    sheet_names = ["Right_Ring", "Right_Segm", "Left_Ring", "Left_Segm"]

    path = check_or_create_sub_path(sub=sub)
    filepath = os.path.join(path, f"{filename}.xlsx")

    # write each dataframe to separate Excel sheet
    with pd.ExcelWriter(filepath) as writer:

        for sheet in sheet_names:
            file[sheet].to_excel(writer, sheet_name=sheet, index=False)
