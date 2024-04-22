"""  """

import os
import numpy as np
import pandas as pd
import sys


def get_onedrive_path(folder: str = 'onedrive', sub: str = None):
    """
    Device and OS independent function to find
    the synced-OneDrive folder where data is stored
    Folder has to be in ['onedrive', 'Percept_Data_structured', 'sourcedata']
    """

    folder_options = ['onedrive', 'sourcedata', 'beta_data']

    # Error checking, if folder input is in folder options
    if folder.lower() not in folder_options:
        raise ValueError(f'given folder: {folder} is incorrect, ' f'should be {folder_options}')

    # from your cwd get the path and stop at 'Users'
    path = os.getcwd()

    while os.path.dirname(path)[-5:] != 'Users':
        path = os.path.dirname(path)  # path is now leading to Users/username

    # get the onedrive folder containing "onedrive" and "charit" and add it to the path
    onedrive_f = [f for f in os.listdir(path) if np.logical_and('onedrive' in f.lower(), 'charit' in f.lower())]

    path = os.path.join(path, onedrive_f[0])  # path is now leading to Onedrive folder

    # add the folder DATA-Test to the path and from there open the folders depending on input folder
    datapath = os.path.join(path, 'Percept_Data_structured')
    if folder == 'onedrive':
        return datapath

    elif folder == 'sourcedata':
        return os.path.join(datapath, 'sourcedata')
    
    elif folder == 'beta_data':
        return os.path.join(datapath, 'beta_data')
    
    
def get_onedrive_path_mac(folder: str = 'onedrive', sub: str = None):
    """
    Device and OS independent function to find
    the synced-OneDrive folder where data is stored
    Folder has to be in ['onedrive', 'Percept_Data_structured', 'sourcedata']
    """

    folder_options = ['onedrive', 'sourcedata', 'beta_data']

    # Error checking, if folder input is in folder options
    if folder.lower() not in folder_options:
        raise ValueError(f'given folder: {folder} is incorrect, ' f'should be {folder_options}')

    # from your cwd get the path and stop at 'Users'
    path = os.getcwd()

    while os.path.dirname(path)[-5:] != 'Users':
        path = os.path.dirname(path)  # path is now leading to Users/username

    # get the onedrive folder containing "charit" and add it to the path

    path = os.path.join(path, 'Charité - Universitätsmedizin Berlin')

    # onedrive_f = [
    #     f for f in os.listdir(path) if np.logical_and(
    #         'onedrive' in f.lower(),
    #         'shared' in f.lower())
    #         ]
    # print(onedrive_f)

    # path = os.path.join(path, onedrive_f[0]) # path is now leading to Onedrive folder

    # add the folder DATA-Test to the path and from there open the folders depending on input folder
    datapath = os.path.join(path, 'AG Bewegungsstörungen - Percept - Percept_Data_structured')
    if folder == 'onedrive':
        return datapath

    elif folder == 'sourcedata':
        return os.path.join(datapath, 'sourcedata')
    
    elif folder == 'beta_data':
        return os.path.join(datapath, 'beta_data')


def chdir_repository(repository: str):
    """
    repository: "Py_Perceive", "Beta_profile"

    """

    #######################     USE THIS DIRECTORY FOR IMPORTING PYPERCEIVE REPO  #######################

    # create a path to the BetaSenSightLongterm folder
    # and a path to the code folder within the BetaSenSightLongterm Repo
    jennifer_user_path = os.getcwd()
    while jennifer_user_path[-14:] != 'jenniferbehnke':
        jennifer_user_path = os.path.dirname(jennifer_user_path)

    repo_dict = {
        "Py_Perceive": os.path.join(jennifer_user_path, 'code', 'PyPerceive_project', 'PyPerceive', 'code'),
        "Beta_profile": os.path.join(jennifer_user_path, 'code', 'Beta_profile_project', 'Beta_profile')
    }

    # directory to PyPerceive code folder
    project_path = repo_dict[repository]
    sys.path.append(project_path)

    # # change directory to PyPerceive code path within BetaSenSightLongterm Repo
    os.chdir(project_path)

    return os.getcwd()
